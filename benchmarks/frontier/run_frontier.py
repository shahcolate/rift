"""Frontier head-to-head benchmark driver.

Measures two frontier models against each other — by default Claude
Fable 5 (baseline) vs GPT-5.6 Sol (challenger) — with the verdict bound
to the pre-registered primary endpoint in ``preregistration.yaml``:
pooled accuracy on the frontier panel (reasoning + extraction +
hard_reasoning, all binary-scored and judge-free). Everything else in
the report is exploratory.

Two modes:

* **live** — hits the real provider APIs. Requires the vendors' keys
  (ANTHROPIC_API_KEY / OPENAI_API_KEY for the default pair). A whole-run
  cost estimate is checked against ``--max-cost`` before the first
  request. Run JSONs are saved per (suite, model) so the capture is
  auditable and replayable.

* **replay** — rebuilds the full report (including the pre-registration
  verdict) from the run JSONs of a previous live capture, keyless. Use
  this to reproduce a published report exactly.

The GitHub Actions workflow ``.github/workflows/frontier-benchmark.yml``
runs live mode on demand with repo secrets — useful when your local
network can't reach the provider APIs.

Usage:

    python benchmarks/frontier/run_frontier.py --mode live \
        --baseline fable-5 --challenger gpt-5.6-sol

    python benchmarks/frontier/run_frontier.py --mode replay \
        --from-dir benchmarks/frontier/results/2026-07-12
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import sys
from pathlib import Path

import click

from rift.comparator import compare_by_subgroup, compare_runs, variance_components
from rift.config import SuiteConfig, load_suite, resolve_model
from rift.observatory import BudgetTracker, estimate_stage_cost
from rift.preregistration import PreregOutcome, Preregistration, evaluate, load_preregistration
from rift.reporter import generate_markdown_report
from rift.runner import RunResult, run_suite

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PREREG_PATH = HERE / "preregistration.yaml"

PANEL_NAME = "frontier_panel_v1"
# Binary-scored, judge-free suites pooled into the pre-registered primary
# endpoint. The tuple order is part of the panel definition (it fixes the
# pooled case order) — do not reorder without re-registering.
POOLED_SUITES = ("reasoning", "extraction", "hard_reasoning")
# Continuous-scored suites: exploratory only, never pooled.
EXPLORATORY_SUITES = ("summarization", "code_generation")
# LLM-judged suites: exploratory and opt-in (--judged). Cross-vendor
# caveat: the pinned judge shares a vendor with one contestant, so any
# house-style bias is NOT symmetric the way it is in a same-vendor
# comparison. Disclosed in the report whenever these run.
JUDGED_SUITES = ("open_ended_qa",)
JUDGE_MODEL = "claude-sonnet-4-6"


def _die(msg: str) -> None:
    """Operational failure: message to stderr, exit 2 (never 1 — the
    gate code is reserved for confirmed regressions in `rift compare`)."""
    print(msg, file=sys.stderr)
    sys.exit(2)


def pool_pairs(
    baseline_runs: dict[str, RunResult], challenger_runs: dict[str, RunResult]
) -> tuple[list[float], list[float], list[float], list[float], list[list[str]], int]:
    """Concatenate the pooled suites into one paired score/cost vector set.

    Cases are matched by ``case_index`` within each suite; a pair where
    either side errored is excluded (an outage is not drift — same rule
    as the observatory). Returns ``(baseline_scores, challenger_scores,
    baseline_costs, challenger_costs, tags_per_case, n_excluded)`` where
    each tag is ``suite:<name>`` so ``compare_by_subgroup`` can split the
    pooled result back out per suite.
    """
    b_scores: list[float] = []
    c_scores: list[float] = []
    b_costs: list[float] = []
    c_costs: list[float] = []
    tags: list[list[str]] = []
    excluded = 0
    for name in POOLED_SUITES:
        b_cases = {c.case_index: c for c in baseline_runs[name].cases}
        c_cases = {c.case_index: c for c in challenger_runs[name].cases}
        for idx in sorted(b_cases.keys() & c_cases.keys()):
            b, c = b_cases[idx], c_cases[idx]
            if b.error or c.error:
                excluded += 1
                continue
            b_scores.append(b.score)
            c_scores.append(c.score)
            b_costs.append(b.cost_usd)
            c_costs.append(c.cost_usd)
            tags.append([f"suite:{name}"])
    return b_scores, c_scores, b_costs, c_costs, tags, excluded


def _pooled_run(model_label: str, runs: dict[str, RunResult],
                include_errored: bool = False) -> RunResult:
    """Assemble a synthetic RunResult over the pooled panel.

    Built from the same cases the pooled drift test uses (errored cases
    dropped unless ``include_errored``) so the totals the report renders
    are consistent with the statistics.
    """
    cases = []
    fingerprints: set[str] = set()
    for name in POOLED_SUITES:
        for c in sorted(runs[name].cases, key=lambda c: c.case_index):
            if c.error and not include_errored:
                continue
            cases.append(c)
            if c.provider_fingerprint:
                fingerprints.add(c.provider_fingerprint)
    started = min((r.started_at for r in runs.values() if r.started_at), default="")
    completed = max((r.completed_at for r in runs.values() if r.completed_at), default="")
    return RunResult(
        model=model_label,
        suite_name=PANEL_NAME,
        scoring_method="pooled_binary",
        cases=cases,
        started_at=started,
        completed_at=completed,
        metadata={
            "pooled_from": list(POOLED_SUITES),
            "fingerprints": sorted(fingerprints),
        },
    )


def _fingerprints(run: RunResult) -> list[str]:
    meta = run.metadata.get("fingerprints")
    if meta:
        return list(meta)
    return sorted({c.provider_fingerprint for c in run.cases if c.provider_fingerprint})


def _verdict_sentence(outcome: PreregOutcome) -> str:
    """One-sentence headline bound to the pre-registered primary endpoint."""
    qualifier = "" if outcome.honored else " (PLAN DISHONORED — see violations)"
    if outcome.direction == "two_sided":
        if outcome.primary_significant:
            side = "challenger ahead" if outcome.primary_delta > 0 else "baseline ahead"
            return (f"**Significant difference on the primary endpoint** "
                    f"({side}; {outcome.detail}).{qualifier}")
        return (f"**No significant difference on the primary endpoint** "
                f"({outcome.detail}).{qualifier}")
    if outcome.adverse_confirmed:
        return f"**Pre-registered adverse outcome CONFIRMED** ({outcome.detail}).{qualifier}"
    return f"**Pre-registered adverse outcome not confirmed** ({outcome.detail}).{qualifier}"


def estimate_total_cost(model_strs: list[str], suites: list[SuiteConfig],
                        trials: int) -> float:
    """Whole-plan pre-flight estimate (list price, both models, all suites)."""
    total = 0.0
    for suite in suites:
        for m in model_strs:
            cfg = resolve_model(m)
            total += estimate_stage_cost(cfg.model, suite, provider=cfg.provider) * trials
    return total


async def _run_live(args, suites: dict[str, SuiteConfig],
                    out_dir: Path) -> tuple[dict[str, RunResult], dict[str, RunResult]]:
    from rich.console import Console

    from rift.keys import ensure_provider_keys

    providers = [resolve_model(args.baseline).provider,
                 resolve_model(args.challenger).provider]
    if args.judged:
        providers.append(resolve_model(JUDGE_MODEL).provider)
    ensure_provider_keys(providers, Console())

    budget = BudgetTracker(args.max_cost)
    baseline_runs: dict[str, RunResult] = {}
    challenger_runs: dict[str, RunResult] = {}
    for name, suite in suites.items():
        for model_str, bucket in ((args.baseline, baseline_runs),
                                  (args.challenger, challenger_runs)):
            cfg = resolve_model(model_str)
            est = estimate_stage_cost(cfg.model, suite, provider=cfg.provider) * args.trials
            if not budget.allows(est):
                budget.skipped.append(f"{name}/{model_str}")
                continue
            run = await run_suite(
                suite, cfg, concurrency=args.concurrency,
                cache_dir=args.cache_dir, trials=args.trials,
            )
            budget.add(run.total_cost_usd)
            run.save(out_dir / name / f"{model_str}.json", strip_io=args.strip_io)
            bucket[name] = run
    if budget.skipped:
        # A partial head-to-head is misleading; record what was skipped
        # loudly. The report generation below will fail cleanly if a
        # pooled suite is missing.
        print(f"⚠️  budget cap ${args.max_cost:.2f} hit — skipped: "
              f"{', '.join(budget.skipped)}", file=sys.stderr)
    return baseline_runs, challenger_runs


def _load_replay(args, suites: dict[str, SuiteConfig],
                 from_dir: Path) -> tuple[dict[str, RunResult], dict[str, RunResult]]:
    baseline_runs: dict[str, RunResult] = {}
    challenger_runs: dict[str, RunResult] = {}
    for name in suites:
        for model_str, bucket in ((args.baseline, baseline_runs),
                                  (args.challenger, challenger_runs)):
            path = from_dir / name / f"{model_str}.json"
            if not path.exists():
                if name in POOLED_SUITES:
                    have = sorted(str(p.relative_to(from_dir))
                                  for p in from_dir.glob("*/*.json"))
                    _die(
                        f"replay: missing pooled-suite run {path}\n"
                        f"  runs present under {from_dir}:\n    "
                        + ("\n    ".join(have) if have else "(none)")
                    )
                continue  # exploratory suite absent from this capture — skip
            bucket[name] = RunResult.load(path)
    return baseline_runs, challenger_runs


def render_report(args, prereg: Preregistration, outcome: PreregOutcome,
                  pooled_drift, pooled_base: RunResult, pooled_chal: RunResult,
                  n_excluded: int,
                  baseline_runs: dict[str, RunResult],
                  challenger_runs: dict[str, RunResult],
                  suite_drifts: dict[str, object],
                  replication: dict | None,
                  n_trials: int,
                  stamp: str) -> str:
    s: list[str] = []
    suites_run = list(baseline_runs)
    s.append(f"# Frontier head-to-head — {args.baseline} vs {args.challenger}\n")
    s.append(f"_Generated {stamp} · mode: `{args.mode}` · trials per case: "
             f"{n_trials} · suites: {', '.join(f'`{n}`' for n in suites_run)}_\n")
    if args.mode == "replay":
        s.append(
            "> ♻️ **Keyless replay.** Rebuilt from the saved run JSONs in "
            f"`{args.from_dir}` — no API calls were made. The capture dates "
            "of the underlying runs are listed under Integrity.\n"
        )

    s.append(
        "> **Serving-configuration disclosure.** All $ figures are "
        "standard-mode list price (`rift.pricing`); both vendors sell the "
        "same models at other prices (batch discounts, fast/priority "
        "tiers, cached-input rates). Model *configuration* also differs "
        "by default — e.g. Claude Fable 5 ships always-on thinking billed "
        "as output tokens, and its tokenizer counts ~30% more tokens than "
        "Opus-tier models for identical text — so compare **$/correct**, "
        "never per-token rates. Each side ran with its provider defaults "
        "plus the per-suite `model_params` recorded in the run JSONs.\n"
    )

    # --- Primary endpoint (confirmatory) ---
    s.append("## Pre-registered primary endpoint (confirmatory)\n")
    s.append(_verdict_sentence(outcome) + "\n")
    s.append(
        f"Plan (`preregistration.yaml`, committed before the first "
        f"capture): primary `{prereg.primary}`, direction "
        f"`{prereg.direction}`, α={prereg.alpha}, min_cases="
        f"{prereg.min_cases}, panel `{prereg.suite}` = "
        + " + ".join(f"`{n}`" for n in POOLED_SUITES) + "."
    )
    if prereg.hypothesis:
        s.append(f"\n> Hypothesis: {prereg.hypothesis.strip()}")
    s.append("")
    if outcome.violations:
        s.append("**Protocol violations:**\n")
        s.extend(f"- ⚠️ {v}" for v in outcome.violations)
        s.append("")
    if n_excluded:
        s.append(
            f"_{n_excluded} case pair(s) excluded from the pooled test "
            "because one side errored (an outage is not drift)._\n"
        )
    s.append(generate_markdown_report(pooled_drift, pooled_base, pooled_chal))
    s.append("")

    # --- Scorecard ---
    s.append("## Scorecard — every suite, both models\n")
    s.append(
        "| Suite | Model | Mean | Correct | Errors | In tok | Out tok "
        "| Spend | $/correct |\n"
        "|-------|-------|------|---------|--------|--------|---------"
        "|-------|-----------|"
    )
    for name in suites_run:
        for label, run in ((args.baseline, baseline_runs[name]),
                           (args.challenger, challenger_runs[name])):
            n_correct = sum(1 for c in run.cases if c.score >= 0.999)
            n_err = run.metadata.get("n_errors", 0)
            cpc = run.cost_per_correct()
            cpc_s = "∞" if cpc == float("inf") else f"${cpc:.4f}"
            s.append(
                f"| `{name}` | `{label}` | {run.mean_score:.3f} | "
                f"{n_correct}/{len(run.cases)} | {n_err} | "
                f"{run.total_input_tokens:,} | {run.total_output_tokens:,} | "
                f"${run.total_cost_usd:.4f} | {cpc_s} |"
            )
    s.append("")

    # --- Token profile ---
    s.append(f"## Token profile — `{args.challenger}` / `{args.baseline}`\n")
    s.append(
        "Divergent input ratios point at tokenizer differences; divergent "
        "output ratios point at response-length / thinking-budget "
        "differences. Both feed directly into $/correct.\n"
    )
    s.append("| Suite | Input ratio | Output ratio | Total ratio |\n"
             "|-------|-------------|--------------|-------------|")
    for name in suites_run:
        b, c = baseline_runs[name], challenger_runs[name]
        bi, bo = b.total_input_tokens, b.total_output_tokens
        ci, co = c.total_input_tokens, c.total_output_tokens
        in_r = f"{ci / bi:.3f}×" if bi else "—"
        out_r = f"{co / bo:.3f}×" if bo else "—"
        tot_r = f"{(ci + co) / (bi + bo):.3f}×" if (bi + bo) else "—"
        s.append(f"| `{name}` | {in_r} | {out_r} | {tot_r} |")
    s.append("")

    # --- Exploratory per-suite drift ---
    s.append("## Exploratory: per-suite drift\n")
    s.append(
        "> Hypothesis-generating, not confirmatory — the verdict above is "
        "bound to the pooled primary endpoint only. Per-suite p-values are "
        "uncorrected.\n"
    )
    for name, drift in suite_drifts.items():
        b, c = baseline_runs[name], challenger_runs[name]
        n_err = b.metadata.get("n_errors", 0) + c.metadata.get("n_errors", 0)
        s.append(f"### `{name}`\n")
        if n_err:
            s.append(
                f"> ⚠️ {n_err} errored case(s) in this suite are scored 0 "
                "here (matching `rift compare`); the pooled primary "
                "excludes them instead.\n"
            )
        s.append(generate_markdown_report(drift, b, c))
        s.append("")

    # --- Replication ---
    if replication and replication.get("mean_trials", 0) > 1:
        s.append("## Replication / noise floor\n")
        s.append(
            f"Pooled panel, {n_trials} trials per case: "
            f"within-case SD {replication['mean_within_sd']:.4f}, "
            f"ICC {replication['icc']:.3f}, noise floor "
            f"{replication['noise_floor']:.4f}. A pooled delta within "
            "~2× the noise floor may not survive a re-run.\n"
        )

    # --- Integrity ---
    s.append("## Integrity\n")
    s.append("| Suite | Model | Served fingerprint(s) | Errors | Captured |\n"
             "|-------|-------|------------------------|--------|----------|")
    for name in suites_run:
        for label, run in ((args.baseline, baseline_runs[name]),
                           (args.challenger, challenger_runs[name])):
            fps = ", ".join(f"`{f}`" for f in _fingerprints(run)) or "—"
            s.append(
                f"| `{name}` | `{label}` | {fps} | "
                f"{run.metadata.get('n_errors', 0)} | "
                f"{run.started_at or '—'} |"
            )
    s.append("")
    if any(n in suites_run for n in JUDGED_SUITES):
        s.append(
            f"> ⚖️ **Judge disclosure.** LLM-judged suites used "
            f"`{JUDGE_MODEL}` — an Anthropic model judging an "
            "Anthropic-vs-OpenAI comparison. Unlike a same-vendor "
            "comparison, any house-style judge bias does NOT land on both "
            "sides equally. Treat judged numbers as exploratory color, "
            "and validate the judge (`rift validate-judge`) before "
            "quoting them.\n"
        )
    return "\n".join(s)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Frontier head-to-head benchmark (pre-registered)."
    )
    ap.add_argument("--mode", choices=("live", "replay"), default="live")
    ap.add_argument("--baseline", default="fable-5")
    ap.add_argument("--challenger", default="gpt-5.6-sol")
    ap.add_argument("--trials", type=int, default=1,
                    help="Replication trials per case (noise floor).")
    ap.add_argument("--concurrency", type=int, default=2,
                    help="Per-model parallel requests. Lower if you hit 429s.")
    ap.add_argument("--judged", action="store_true",
                    help="Include LLM-judged suites (exploratory; judge "
                         "shares a vendor with one contestant — disclosed).")
    ap.add_argument("--strip-io", action="store_true",
                    help="Blank input/output text in saved run JSONs.")
    ap.add_argument("--max-cost", type=float, default=25.0,
                    help="USD cap: refuse to start if the whole-plan "
                         "estimate exceeds it; stop starting new stages "
                         "once actual spend reaches it.")
    ap.add_argument("--out-dir", default=None,
                    help="Results dir (default: benchmarks/frontier/"
                         "results/<UTC date>).")
    ap.add_argument("--from-dir", default=None,
                    help="Replay: results dir of a previous live capture.")
    ap.add_argument("--cache-dir", default=str(ROOT / ".rift" / "cache"))
    ap.add_argument("--preregister", default=str(PREREG_PATH))
    args = ap.parse_args(argv)

    try:
        _run(args)
    except click.ClickException as e:
        # Rift's operational errors are ClickExceptions (clean message +
        # exit code). argparse scripts don't get Click's rendering for
        # free, so mirror it.
        e.show()
        sys.exit(e.exit_code)


def _run(args) -> None:
    prereg = load_preregistration(args.preregister)

    suite_names = list(POOLED_SUITES) + list(EXPLORATORY_SUITES)
    if args.judged:
        suite_names += list(JUDGED_SUITES)
    suites: dict[str, SuiteConfig] = {}
    for name in suite_names:
        suite = load_suite(name)
        if args.judged and name in JUDGED_SUITES:
            # Pin the judge in the suite config so the capture is
            # reproducible regardless of RIFT_JUDGE_MODEL in the caller's
            # environment.
            suite.judge_model = JUDGE_MODEL
        suites[name] = suite

    stamp = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    if args.mode == "live":
        est = estimate_total_cost([args.baseline, args.challenger],
                                  list(suites.values()), args.trials)
        if est > args.max_cost:
            _die(
                f"pre-flight estimate ${est:.2f} exceeds --max-cost "
                f"${args.max_cost:.2f}; raise the cap or trim the plan "
                "(fewer trials, --judged off). Nothing was run."
            )
        out_dir = Path(args.out_dir) if args.out_dir else HERE / "results" / stamp
        baseline_runs, challenger_runs = asyncio.run(
            _run_live(args, suites, out_dir)
        )
        missing = [n for n in POOLED_SUITES
                   if n not in baseline_runs or n not in challenger_runs]
        if missing:
            _die(
                f"pooled suite(s) not captured (budget cap?): "
                f"{', '.join(missing)} — no report written; the primary "
                "endpoint cannot be evaluated on a partial panel."
            )
    else:
        if not args.from_dir:
            _die("replay mode requires --from-dir <results dir>")
        from_dir = Path(args.from_dir)
        if not from_dir.is_dir():
            _die(f"replay: {from_dir} is not a directory")
        out_dir = from_dir
        baseline_runs, challenger_runs = _load_replay(args, suites, from_dir)

    # Primary (confirmatory): pooled panel at the pre-registered alpha.
    b_scores, c_scores, b_costs, c_costs, tags, n_excluded = pool_pairs(
        baseline_runs, challenger_runs
    )
    pooled_drift = compare_runs(
        baseline_scores=b_scores,
        challenger_scores=c_scores,
        baseline_model=args.baseline,
        challenger_model=args.challenger,
        suite_name=PANEL_NAME,
        alpha=prereg.alpha,
        baseline_costs=b_costs,
        challenger_costs=c_costs,
    )
    pooled_drift.subgroups = compare_by_subgroup(
        baseline_scores=b_scores,
        challenger_scores=c_scores,
        tags_per_case=tags,
        subgroup_prefix="suite:",
        baseline_model=args.baseline,
        challenger_model=args.challenger,
        suite_name=PANEL_NAME,
        alpha=prereg.alpha,
        baseline_costs=b_costs,
        challenger_costs=c_costs,
    )
    outcome = evaluate(prereg, pooled_drift, n_cases=pooled_drift.n_cases,
                       baseline_model=args.baseline,
                       challenger_model=args.challenger)
    pooled_base = _pooled_run(args.baseline, baseline_runs)
    pooled_chal = _pooled_run(args.challenger, challenger_runs)

    # Exploratory: per-suite drift at the same alpha, full vectors
    # (errored cases score 0, matching `rift compare`).
    suite_drifts = {}
    for name in baseline_runs:
        b, c = baseline_runs[name], challenger_runs[name]
        suite_drifts[name] = compare_runs(
            baseline_scores=b.scores,
            challenger_scores=c.scores,
            baseline_model=args.baseline,
            challenger_model=args.challenger,
            suite_name=name,
            alpha=prereg.alpha,
            baseline_costs=[case.cost_usd for case in b.cases],
            challenger_costs=[case.cost_usd for case in c.cases],
        )

    # Trials come from the runs themselves (metadata stamped by
    # run_suite when trials > 1) so a replay reports the capture's real
    # replication level without the caller re-passing --trials.
    all_runs = list(baseline_runs.values()) + list(challenger_runs.values())
    n_trials = max([r.metadata.get("trials", 1) for r in all_runs] + [1])
    replication = None
    if n_trials > 1:
        pooled_trials = [
            case.trial_scores
            for runs in (baseline_runs, challenger_runs)
            for name in POOLED_SUITES
            for case in runs[name].cases
            if case.trial_scores
        ]
        if pooled_trials:
            replication = variance_components(pooled_trials)

    report = render_report(
        args, prereg, outcome, pooled_drift, pooled_base, pooled_chal,
        n_excluded, baseline_runs, challenger_runs, suite_drifts,
        replication, n_trials, stamp,
    )
    report_path = out_dir / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)

    print(f"results: {out_dir}")
    print(f"report:  {report_path}")
    print(_verdict_sentence(outcome).replace("**", ""))


if __name__ == "__main__":
    main()
