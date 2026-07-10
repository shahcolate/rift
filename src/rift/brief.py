"""Model upgrade brief: the exec-readable one-pager, from ANY comparison.

``rift report comparison.json --format html`` renders the payload that
``rift compare --output`` saved into the same visual language as the
``rift demo`` memo — verdict up top, accuracy and $/correct charts,
action items — but with narration derived from the numbers instead of a
prepared scenario. One reconstruction path (:func:`load_comparison`)
also feeds the markdown and terminal formats, so every renderer reads
the same data.

The verdict is rules-based and conservative: it claims only what the
statistics support (significance direction + cost CI), and every brief
carries the reproduce command and the error-count disclosure.
"""

from __future__ import annotations

import json
from pathlib import Path

from .comparator import DriftResult
from .demo import (
    DemoAct,
    DemoScript,
    VerdictCard,
    _fmt_pct,
    _safe_pct_change,
    export_demo_html,
    export_demo_markdown,
)
from ._errors import OperationalError
from .runner import RunResult


class ComparisonLoadError(OperationalError):
    """The file isn't a ``rift compare --output`` payload."""


def _drift_from_dict(data: dict) -> DriftResult:
    """Rebuild a DriftResult (and nested subgroups) from its asdict form.

    Filters to known fields the same way :meth:`RunResult.load` does, so
    payloads saved by an older or newer Rift still load.
    """
    fields = DriftResult.__dataclass_fields__  # type: ignore[attr-defined]
    kwargs = {k: v for k, v in data.items() if k in fields}
    kwargs["subgroups"] = {
        name: _drift_from_dict(sub)
        for name, sub in (data.get("subgroups") or {}).items()
    }
    return DriftResult(**kwargs)


def load_comparison(path: str | Path) -> tuple[DriftResult, RunResult, RunResult, dict]:
    """Load a ``rift compare --output`` JSON.

    Returns ``(drift, baseline_run, challenger_run, extras)`` where
    ``extras`` carries the optional sections (power, replication,
    preregistration, ...) verbatim.
    """
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        raise ComparisonLoadError(f"Comparison file not found: {path}") from None
    except ValueError as e:
        raise ComparisonLoadError(f"Not valid JSON: {path} ({e})") from None
    if not isinstance(data, dict) or "drift" not in data \
            or "baseline" not in data or "challenger" not in data:
        raise ComparisonLoadError(
            f"{path} is not a comparison payload — expected the JSON written "
            "by `rift compare --output` (keys: drift, baseline, challenger)."
        )
    try:
        drift = _drift_from_dict(data["drift"])
        base = RunResult.from_dict(data["baseline"])
        chal = RunResult.from_dict(data["challenger"])
    except (KeyError, TypeError) as e:
        raise ComparisonLoadError(
            f"{path} has an unrecognized shape ({e}). Re-save it with the "
            "current Rift's `compare --output`."
        ) from None
    extras = {k: v for k, v in data.items()
              if k not in ("drift", "baseline", "challenger")}
    return drift, base, chal, extras




def _verdict(drift: DriftResult, n_err: int) -> VerdictCard:
    """Rules-based verdict: claim exactly what the statistics support."""
    cost_up = (drift.cost_delta_ci_defined
               and drift.cost_delta_ci_lower > 0)
    cost_down = (drift.cost_delta_ci_defined
                 and drift.cost_delta_ci_upper < 0)
    ci_pct = int(round(getattr(drift, "ci_level", 0.95) * 100))

    actions: list[str] = []
    if n_err:
        actions.append(
            f"Re-run until the {n_err} errored case(s) are zero before "
            "citing these numbers — an errored case scores 0 and is "
            "indistinguishable from a wrong answer."
        )

    if drift.significant and drift.delta < 0:
        headline = (
            f"Statistically significant accuracy regression: "
            f"{drift.delta * 100:+.2f}pp (p={drift.p_value:.4f})."
        )
        recommendation = (
            "Do not promote the challenger on this workload without a fix "
            "or an explicit accept of the regression."
        )
        actions += [
            "Inspect the regressed cases in the saved comparison JSON "
            f"({len(drift.regressed_cases)} case(s)).",
            "If the workload matters, gate the rollout: the reproduce "
            "command below exits 1 on this regression.",
        ]
    elif drift.significant and drift.delta > 0:
        headline = (
            f"Statistically significant accuracy improvement: "
            f"{drift.delta * 100:+.2f}pp (p={drift.p_value:.4f})."
        )
        recommendation = "Accuracy supports the upgrade on this workload."
        actions.append(
            "Check the cost verdict below before promoting — an accuracy "
            "win at a step-change in $/correct is a procurement decision, "
            "not an engineering one."
        )
    else:
        headline = (
            f"No statistically significant accuracy change "
            f"(Δ={drift.delta * 100:+.2f}pp, p={drift.p_value:.4f})."
        )
        recommendation = (
            "Treat accuracy as unchanged at this sample size; decide on "
            "cost and the behavioral probes."
        )
        actions.append(
            "If the observed delta would matter in production, this suite "
            "is underpowered to confirm it — add cases or trials before "
            "deciding (see the power analysis in the saved comparison)."
        )

    if cost_up:
        actions.append(
            f"$/correct rose and the {ci_pct}% CI excludes zero — budget "
            "impact is statistically supported, not noise."
        )
    elif cost_down:
        actions.append(
            f"$/correct fell and the {ci_pct}% CI excludes zero — the "
            "challenger is cheaper per correct answer on this workload."
        )

    confidence = (
        f"Verdict at α={1 - getattr(drift, 'ci_level', 0.95):.2g} via "
        f"{drift.test_used}; CIs are paired-bootstrap at the {ci_pct}% "
        "level. List-price cost only — restate for your serving "
        "configuration (batch/fast/cache tiers) before budgeting."
    )
    reproduce = (
        f"rift compare --baseline {drift.baseline_model} "
        f"--challenger {drift.challenger_model} --suite {drift.suite_name}"
    )
    return VerdictCard(
        headline=headline,
        recommendation=recommendation,
        action_items=actions,
        confidence_note=confidence,
        reproduce_cmd=reproduce,
    )


def build_brief_script(drift: DriftResult, base_run: RunResult,
                       chal_run: RunResult) -> DemoScript:
    """A DemoScript for an arbitrary comparison — the 'upgrade brief'.

    Reuses the demo's exporters so the brief and the demo memo share one
    visual language; only the narration differs, and here it is derived
    from the statistics rather than a prepared story.
    """
    from .reporter import _error_counts

    n_err_base, n_err_chal = _error_counts(base_run, chal_run)
    delta_pp = drift.delta * 100.0
    cpc_pct = _safe_pct_change(drift.baseline_cost_per_correct,
                               drift.challenger_cost_per_correct)
    ci_pct = int(round(getattr(drift, "ci_level", 0.95) * 100))

    acc_md = (
        "## Accuracy\n\n"
        f"- Baseline mean: **{drift.baseline_mean:.4f}**\n"
        f"- Challenger mean: **{drift.challenger_mean:.4f}**\n"
        f"- Delta: **{drift.delta:+.4f} ({delta_pp:+.2f}pp)**\n"
        f"- p-value: {drift.p_value:.4f}  ·  test: {drift.test_used}\n"
        f"- {ci_pct}% CI: [{drift.ci_lower:+.4f}, {drift.ci_upper:+.4f}]\n"
    )
    cost_md = (
        "## Cost\n\n"
        f"- Baseline $/correct: **{drift.baseline_cost_per_correct:.4f}**\n"
        f"- Challenger $/correct: **{drift.challenger_cost_per_correct:.4f}**\n"
        f"- Δ $/correct: **{drift.cost_normalized_delta_usd:+.4f}** "
        f"({_fmt_pct(cpc_pct)})\n"
        + (
            f"- {ci_pct}% CI on Δ $/correct: "
            f"[{drift.cost_delta_ci_lower:+.4f}, "
            f"{drift.cost_delta_ci_upper:+.4f}]\n"
            if drift.cost_delta_ci_defined else
            "- CI on Δ $/correct: undefined (a run had zero correct cases)\n"
        )
        + "\nList price only — see the serving-configuration note in the "
        "verdict.\n"
    )

    acts = [
        DemoAct(title="Accuracy", body_md=acc_md),
        DemoAct(title="Cost", body_md=cost_md),
    ]
    if drift.subgroups:
        rows = "\n".join(
            f"| {tag} | {d.n_cases} | {d.baseline_mean:.3f} | "
            f"{d.challenger_mean:.3f} | {d.delta:+.3f} | "
            f"[{d.ci_lower:+.3f}, {d.ci_upper:+.3f}] | {d.p_value:.4f} |"
            for tag, d in sorted(drift.subgroups.items())
        )
        acts.append(DemoAct(
            title="Subgroups",
            body_md=("## Subgroups\n\n"
                     f"| Subgroup | n | Baseline | Challenger | Δ | {ci_pct}% CI | p |\n"
                     "|---|---|---|---|---|---|---|\n" + rows + "\n\n"
                     "Raw p-values; apply the BH-corrected q from the full "
                     "report before citing any single subgroup.\n"),
        ))

    headline_numbers = {
        "accuracy_delta": f"{delta_pp:+.2f}pp",
        "p_value": f"{drift.p_value:.4f}",
        "cost_per_correct_delta": _fmt_pct(cpc_pct),
    }
    if n_err_base or n_err_chal:
        headline_numbers["errored_cases"] = (
            f"{n_err_base} baseline / {n_err_chal} challenger"
        )

    return DemoScript(
        title=(f"Model upgrade brief: {drift.baseline_model} → "
               f"{drift.challenger_model}"),
        subtitle=(f"Paired drift comparison on '{drift.suite_name}' "
                  f"({drift.n_cases} cases). Generated by rift report."),
        baseline_model=drift.baseline_model,
        challenger_model=drift.challenger_model,
        suite_name=drift.suite_name,
        n_cases=drift.n_cases,
        acts=acts,
        verdict=_verdict(drift, n_err_base + n_err_chal),
        headline_numbers=headline_numbers,
        sources=["rift compare --output payload"],
        badge_label="live comparison",
    )


def export_brief_html(drift: DriftResult, base_run: RunResult,
                      chal_run: RunResult, path: str | Path) -> None:
    """Write the self-contained HTML brief."""
    script = build_brief_script(drift, base_run, chal_run)
    export_demo_html(script, path, base_run, chal_run, drift)


def export_brief_markdown(drift: DriftResult, base_run: RunResult,
                          chal_run: RunResult, path: str | Path) -> None:
    """Write the markdown brief (Notion/Slack/email-ready)."""
    script = build_brief_script(drift, base_run, chal_run)
    export_demo_markdown(script, path)
