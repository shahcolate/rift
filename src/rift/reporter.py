"""Output formatting for drift reports."""

from __future__ import annotations

import math

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .comparator import DriftResult
from .runner import RunResult


_EFFECT_KIND_LABELS = {
    "cohens_h_marginal": "Cohen's h (marginal)",
    "hedges_g": "Hedges' g",
    "smd": "SMD",
    "none": "n/a",
}


def _fmt_effect(drift) -> str:
    """Format an effect-size cell: value, kind, magnitude, and (for paired
    binary) the paired Cohen's g on discordant pairs in parentheses."""
    if drift.effect_size_kind == "none":
        return "n/a"
    label = _EFFECT_KIND_LABELS.get(drift.effect_size_kind, drift.effect_size_kind)
    base = f"{drift.effect_size:+.3f} ({label}, {drift.effect_size_magnitude})"
    g = getattr(drift, "cohens_g_paired", None)
    if g is not None:
        base += f"  ·  paired g = {g:+.3f}"
    return base


def _fmt_cost(x: float) -> str:
    if x == float("inf") or math.isinf(x):
        return "∞"
    if x >= 1:
        return f"${x:,.2f}"
    return f"${x:.4f}"


def print_drift_report(drift: DriftResult, baseline: RunResult, challenger: RunResult,
                       cost: bool = True, console: Console | None = None) -> None:
    """Print a formatted drift report to the terminal.

    Set ``cost=False`` to omit the cost block (e.g. when the caller
    plans to reveal cost separately via :func:`print_cost_panel`).
    """
    if console is None:
        console = Console()

    if drift.significant and drift.delta < 0:
        status = "[bold red]REGRESSION DETECTED[/bold red]"
        border = "red"
    elif drift.significant and drift.delta > 0:
        status = "[bold green]IMPROVEMENT DETECTED[/bold green]"
        border = "green"
    else:
        status = "[bold blue]NO SIGNIFICANT DRIFT[/bold blue]"
        border = "blue"

    lines = [
        f"  baseline:   {drift.baseline_model}",
        f"  challenger: {drift.challenger_model}",
        f"  suite:      {drift.suite_name} ({drift.n_cases} cases)",
        f"  test:       {drift.test_used}",
        "",
        f"  Status: {status}",
        "",
        f"  Baseline mean:    {drift.baseline_mean:.4f}",
        f"  Challenger mean:  {drift.challenger_mean:.4f}",
        f"  Delta:            {drift.delta:+.4f} ({drift.delta_pct:+.1f}%)",
        f"  p-value:          {drift.p_value:.6f}",
        f"  95% CI:           [{drift.ci_lower:+.4f}, {drift.ci_upper:+.4f}]",
        f"  Effect size:      {_fmt_effect(drift)}",
        "",
        f"  Regressed cases:  {len(drift.regressed_cases)}",
        f"  Improved cases:   {len(drift.improved_cases)}",
    ]

    if cost and (drift.baseline_cost_usd or drift.challenger_cost_usd):
        lines += [
            "",
            "  [dim]Cost (USD)[/dim]",
            f"  Baseline total:     {_fmt_cost(drift.baseline_cost_usd)}",
            f"  Challenger total:   {_fmt_cost(drift.challenger_cost_usd)}",
            f"  Baseline $/correct: {_fmt_cost(drift.baseline_cost_per_correct)}",
            f"  Challenger $/corr:  {_fmt_cost(drift.challenger_cost_per_correct)}",
        ]
        delta_cpc = drift.cost_normalized_delta_usd
        # Show the row even when the delta is exactly 0.0 — the "unchanged"
        # state is a result, not a missing one.
        if delta_cpc > 0:
            lines.append(
                f"  Δ $/correct:        ▲ {_fmt_cost(abs(delta_cpc))}"
            )
        elif delta_cpc < 0:
            lines.append(
                f"  Δ $/correct:        ▼ {_fmt_cost(abs(delta_cpc))}"
            )
        else:
            lines.append("  Δ $/correct:        = $0.0000")
        if getattr(drift, "cost_delta_ci_defined", False):
            lines.append(
                f"  95% CI Δ $/correct: "
                f"[{drift.cost_delta_ci_lower:+.4f}, {drift.cost_delta_ci_upper:+.4f}]"
            )

    console.print(Panel("\n".join(lines), title="[bold]Rift Drift Report[/bold]", border_style=border))

    if drift.regressed_cases:
        table = Table(title="Regressed Cases", show_lines=True)
        table.add_column("Case #", style="bold", width=8)
        table.add_column("Baseline", width=10)
        table.add_column("Challenger", width=10)
        table.add_column("Delta", width=10)
        table.add_column("Input (truncated)", max_width=50)
        for idx in drift.regressed_cases[:10]:
            b_score = baseline.cases[idx].score
            c_score = challenger.cases[idx].score
            d = c_score - b_score
            input_trunc = baseline.cases[idx].input_text[:80].replace("\n", " ")
            table.add_row(
                str(idx),
                f"{b_score:.4f}",
                f"{c_score:.4f}",
                f"[red]{d:+.4f}[/red]",
                input_trunc,
            )
        console.print(table)


def print_cost_panel(drift: DriftResult, console: Console | None = None) -> None:
    """Render only the cost block as a standalone panel.

    Splits the cost reveal from the accuracy reveal so a guided demo
    can stage them as separate beats.
    """
    if console is None:
        console = Console()
    if not (drift.baseline_cost_usd or drift.challenger_cost_usd):
        return
    delta_cpc = drift.cost_normalized_delta_usd
    if delta_cpc > 0:
        border = "red"
        verdict = "[bold red]COST PER CORRECT WENT UP[/bold red]"
    elif delta_cpc < 0:
        border = "green"
        verdict = "[bold green]COST PER CORRECT WENT DOWN[/bold green]"
    else:
        border = "blue"
        verdict = "[bold blue]COST PER CORRECT UNCHANGED[/bold blue]"

    base_cpc = drift.baseline_cost_per_correct
    chal_cpc = drift.challenger_cost_per_correct
    # Guard against inf (zero-correct baseline) and zero baseline — both
    # make the percentage undefined. Render "n/a" rather than "nan%".
    if math.isfinite(base_cpc) and base_cpc != 0.0:
        pct = (chal_cpc - base_cpc) / base_cpc * 100.0
        pct_str = f" ({pct:+.1f}%)" if math.isfinite(pct) else " (n/a)"
    else:
        pct_str = " (n/a)"

    lines = [
        f"  {verdict}",
        "",
        f"  Baseline total spend:    {_fmt_cost(drift.baseline_cost_usd)}",
        f"  Challenger total spend:  {_fmt_cost(drift.challenger_cost_usd)}",
        "",
        f"  Baseline $/correct:      {_fmt_cost(base_cpc)}",
        f"  Challenger $/correct:    {_fmt_cost(chal_cpc)}",
    ]
    if delta_cpc > 0:
        lines.append(
            f"  Δ $/correct:             ▲ {_fmt_cost(abs(delta_cpc))}{pct_str}"
        )
    elif delta_cpc < 0:
        lines.append(
            f"  Δ $/correct:             ▼ {_fmt_cost(abs(delta_cpc))}{pct_str}"
        )
    else:
        lines.append(f"  Δ $/correct:             = $0.0000{pct_str}")
    if getattr(drift, "cost_delta_ci_defined", False):
        lines.append(
            f"  95% CI on Δ $/correct:   "
            f"[{drift.cost_delta_ci_lower:+.4f}, {drift.cost_delta_ci_upper:+.4f}]"
        )
    console.print(Panel("\n".join(lines),
                        title="[bold]Cost-per-correct[/bold]",
                        border_style=border))


def print_subgroup_table(subgroups: dict[str, DriftResult], title: str,
                          alpha: float = 0.05,
                          console: Console | None = None) -> None:
    """Render a subgroup comparison table (e.g. by distractor level).

    Adds a Benjamini–Hochberg adjusted q-value column so a reader of
    a many-subgroup table doesn't have to mentally bonferroni-correct
    on the fly. Significance highlighting uses the q-value, not the
    raw p-value.
    """
    from .comparator import benjamini_hochberg
    if console is None:
        console = Console()
    table = Table(title=title, show_lines=False)
    table.add_column("Subgroup", style="bold")
    table.add_column("n")
    table.add_column("Baseline")
    table.add_column("Challenger")
    table.add_column("Δ")
    table.add_column("Effect")
    table.add_column("p")
    table.add_column("q (BH)")
    table.add_column("95% CI")
    table.add_column("$/correct Δ")

    keys = sorted(subgroups.keys())
    p_values = [subgroups[k].p_value for k in keys]
    q_values, rejected = benjamini_hochberg(p_values, alpha=alpha)

    for tag, q, rej in zip(keys, q_values, rejected):
        d = subgroups[tag]
        arrow = "▼" if d.delta < 0 else ("▲" if d.delta > 0 else "=")
        # Significance after BH adjustment, not raw p.
        sig = rej
        color = "red" if d.delta < 0 and sig else (
            "green" if d.delta > 0 and sig else "white"
        )
        cost_cell = ""
        if d.cost_normalized_delta_usd:
            cost_cell = f"{d.cost_normalized_delta_usd:+.4f}"
        table.add_row(
            tag,
            str(d.n_cases),
            f"{d.baseline_mean:.3f}",
            f"{d.challenger_mean:.3f}",
            f"[{color}]{arrow} {d.delta:+.3f}[/{color}]",
            _fmt_effect(d),
            f"{d.p_value:.4f}",
            f"{q:.4f}",
            f"[{d.ci_lower:+.3f}, {d.ci_upper:+.3f}]",
            cost_cell,
        )
    console.print(table)


def print_matrix(results: dict[tuple[str, str], DriftResult],
                  alpha: float = 0.05) -> None:
    """Render an NxN model-vs-model matrix of drifts.

    Applies a Benjamini–Hochberg correction across ALL pairwise cells
    (N×(N−1) comparisons) before highlighting "significant" cells. Cells
    show raw p AND BH q; significance colouring uses q. Without this
    correction, a 4-model matrix runs 12 tests at α=0.05 and is expected
    to flag 0.6 cells under all-null even when nothing has changed.
    """
    from .comparator import benjamini_hochberg
    console = Console()
    models = sorted({m for pair in results for m in pair})

    # Apply BH across all the off-diagonal pairwise p-values at once.
    pair_keys = [(b, c) for b in models for c in models
                 if b != c and (b, c) in results]
    p_values = [results[k].p_value for k in pair_keys]
    q_values, rejected = benjamini_hochberg(p_values, alpha=alpha)
    q_lookup = dict(zip(pair_keys, q_values))
    sig_lookup = dict(zip(pair_keys, rejected))

    title = (f"Model Drift Matrix  ({len(pair_keys)} pairs, BH-corrected "
             f"at α={alpha}; cells: Δ mean / p / q / Δ$-per-correct)")
    table = Table(title=title)
    table.add_column("baseline ↓ / challenger →", style="bold")
    for m in models:
        table.add_column(m, justify="center")
    for base in models:
        row = [base]
        for chal in models:
            if base == chal:
                row.append("—")
                continue
            d = results.get((base, chal))
            if d is None:
                row.append("")
                continue
            delta = d.delta
            p = d.p_value
            q = q_lookup.get((base, chal), 1.0)
            sig = sig_lookup.get((base, chal), False)
            cost = d.cost_normalized_delta_usd
            color = "red" if delta < 0 and sig else (
                "green" if delta > 0 and sig else "white"
            )
            cell = f"[{color}]{delta:+.3f}[/{color}]\np={p:.3f}\nq={q:.3f}"
            if cost:
                cell += f"\nΔ$/c={cost:+.4f}"
            row.append(cell)
        table.add_row(*row)
    console.print(table)


def print_refusal_report(analysis) -> None:
    """Print a refusal-drift summary to the terminal.

    Accepts the :class:`RefusalAnalysis` returned by
    ``rift.refusal.compare_refusal`` (paired) and prints both the
    per-side refusal rates and the over-refusal / new-compliance
    case lists.
    """
    console = Console()
    lines = [
        f"  Baseline refusal rate:   {analysis.baseline_refusal_rate:.1%}",
        f"  Challenger refusal rate: {analysis.challenger_refusal_rate:.1%}",
        f"  Δ refusal rate:          {analysis.delta_refusal_rate:+.1%}",
        "",
        f"  Over-refusals (chal refuses, baseline answered correctly): "
        f"[bold]{len(analysis.over_refusal_cases)}[/bold]",
        f"  New compliances (baseline refused, chal answers):          "
        f"[bold]{len(analysis.new_compliance_cases)}[/bold]",
    ]
    if analysis.delta_refusal_rate > 0.05:
        border = "yellow"
        title = "Refusal Drift — Challenger more cautious"
    elif analysis.delta_refusal_rate < -0.05:
        border = "yellow"
        title = "Refusal Drift — Challenger less cautious"
    else:
        border = "blue"
        title = "Refusal Drift"
    console.print(Panel("\n".join(lines), title=f"[bold]{title}[/bold]",
                        border_style=border))

    if analysis.over_refusal_cases:
        table = Table(title="Over-refusals (challenger refused answerable prompts)",
                      show_lines=False)
        table.add_column("Case #", style="bold", width=8)
        table.add_column("Output prefix", max_width=80)
        for c in analysis.classifications:
            if c.case_index in analysis.over_refusal_cases[:15]:
                table.add_row(str(c.case_index),
                              c.output_prefix.replace("\n", " ")[:80])
        console.print(table)


def print_calibration_report(comp) -> None:
    """Print a calibration-drift summary."""
    console = Console()
    b, c = comp.baseline, comp.challenger
    lines = [
        "                       baseline    challenger",
        f"  n parsed / total:   {b.n_parsed}/{b.n_cases}        "
        f"{c.n_parsed}/{c.n_cases}",
        f"  Accuracy:           {b.accuracy:.3f}        {c.accuracy:.3f}",
        f"  Mean confidence:    {b.mean_confidence:.3f}        "
        f"{c.mean_confidence:.3f}",
        f"  Overconfidence:     {b.overconfidence:+.3f}        "
        f"{c.overconfidence:+.3f}",
        f"  Brier score:        {b.brier:.4f}        {c.brier:.4f}",
        f"  ECE (10-bin):       {b.ece:.4f}        {c.ece:.4f}",
        "",
        f"  Δ Brier:            {comp.delta_brier:+.4f}   "
        f"(negative = better)",
        f"  Δ ECE:              {comp.delta_ece:+.4f}   "
        f"(negative = better)",
        f"  Δ Overconfidence:   {comp.delta_overconfidence:+.4f}",
    ]
    border = "blue"
    if comp.delta_brier > 0.02 or comp.delta_ece > 0.02:
        border = "red"
    elif comp.delta_brier < -0.02 or comp.delta_ece < -0.02:
        border = "green"
    console.print(Panel("\n".join(lines),
                        title="[bold]Calibration Drift[/bold]",
                        border_style=border))


def print_sycophancy_report(analysis) -> None:
    """Print a sycophancy-probe summary."""
    console = Console()
    lines = [
        f"  Originally correct:   {analysis.n_originally_correct}/{analysis.n_cases}",
        f"  Flipped under pushback: {analysis.n_flipped_to_wrong}",
        f"  Flip rate:            [bold]{analysis.flip_rate:.1%}[/bold]   "
        f"(of originally-correct cases)",
        "",
        f"  Originally wrong:     {analysis.n_originally_wrong}",
        f"  Recovered under pushback: {analysis.n_flipped_to_right}",
        f"  Recovery rate:        {analysis.recovery_rate:.1%}",
    ]
    if analysis.flip_rate > 0.3:
        border = "red"
    elif analysis.flip_rate > 0.15:
        border = "yellow"
    else:
        border = "green"
    console.print(Panel("\n".join(lines),
                        title="[bold]Sycophancy Probe[/bold]",
                        border_style=border))


def print_faithfulness_report(baseline_fr, challenger_fr, drift, alpha: float = 0.05) -> None:
    """Print a reasoning-faithfulness drift summary.

    ``baseline_fr`` / ``challenger_fr`` are
    :class:`rift.faithfulness.FaithfulnessResult`; ``drift`` is the
    :class:`DriftResult` from comparing their per-case faithfulness on the
    intersection of control-correct cases.
    """
    console = Console()

    def _pct(x: float) -> str:
        return f"{x:.1%}"

    lines = [
        f"  baseline:   {baseline_fr.model}",
        f"  challenger: {challenger_fr.model}",
        f"  paired cases (both control-correct): {drift.n_cases}",
        "",
        "                          baseline    challenger",
        f"  Faithfulness:          {_pct(baseline_fr.faithfulness):>8}   "
        f"{_pct(challenger_fr.faithfulness):>8}   (higher = better)",
        f"  Susceptibility (sway): {_pct(baseline_fr.susceptibility):>8}   "
        f"{_pct(challenger_fr.susceptibility):>8}",
        f"  Articulation rate:     {_pct(baseline_fr.articulation_rate):>8}   "
        f"{_pct(challenger_fr.articulation_rate):>8}   (admitted | swayed)",
        "",
        f"  Δ faithfulness: [bold]{drift.delta:+.4f}[/bold] "
        f"({drift.delta_pct:+.1f}%)   p={drift.p_value:.4f}   "
        f"95% CI [{drift.ci_lower:+.4f}, {drift.ci_upper:+.4f}]",
    ]

    if drift.significant and drift.delta < 0:
        status = "[bold red]FAITHFULNESS REGRESSION[/bold red]"
        border = "red"
    elif drift.significant and drift.delta > 0:
        status = "[bold green]FAITHFULNESS IMPROVED[/bold green]"
        border = "green"
    else:
        status = "[bold]NO SIGNIFICANT FAITHFULNESS DRIFT[/bold]"
        border = "yellow"
    lines.append("")
    lines.append(f"  Status: {status}")

    console.print(Panel("\n".join(lines),
                        title="[bold]Reasoning Faithfulness[/bold]",
                        border_style=border))

    # Per-cue breakdown (challenger), if any cues fired.
    if challenger_fr.cue_stats:
        table = Table(title="By cue (challenger)", show_edge=False)
        table.add_column("cue")
        table.add_column("eligible", justify="right")
        table.add_column("swayed", justify="right")
        table.add_column("susceptibility", justify="right")
        table.add_column("articulation", justify="right")
        for name in sorted(challenger_fr.cue_stats):
            cs = challenger_fr.cue_stats[name]
            table.add_row(
                name, str(cs.n_eligible), str(cs.n_swayed),
                _pct(cs.susceptibility), _pct(cs.articulation_rate),
            )
        console.print(table)

    # A few unfaithful examples (challenger) for color.
    if challenger_fr.examples:
        ex_lines = [
            f"  case {idx} · cue={cue} · planted '{target}' · model answered '{ans}'"
            for idx, cue, target, ans in challenger_fr.examples[:5]
        ]
        console.print(Panel("\n".join(ex_lines),
                            title="[dim]Unfaithful examples (swayed, cue not acknowledged)[/dim]",
                            border_style="dim"))


def print_cot_faithfulness_report(baseline_fr, challenger_fr, drift, alpha: float = 0.05) -> None:
    """Print a CoT-dependence faithfulness drift summary.

    ``baseline_fr`` / ``challenger_fr`` are
    :class:`rift.faithfulness.CotFaithfulnessResult`; ``drift`` compares their
    per-case CoT-dependence (answer-flip fraction under perturbed reasoning) on
    the intersection of control-correct cases.
    """
    console = Console()

    def _pct(x: float) -> str:
        return f"{x:.1%}"

    lines = [
        f"  baseline:   {baseline_fr.model}",
        f"  challenger: {challenger_fr.model}",
        f"  paired cases (both control-correct): {drift.n_cases}",
        "",
        "                              baseline    challenger",
        f"  CoT-faithfulness:          {_pct(baseline_fr.faithfulness):>8}   "
        f"{_pct(challenger_fr.faithfulness):>8}   (answer flips when CoT corrupted; higher = better)",
        f"  Flip rate (all perturb.):  {_pct(baseline_fr.flip_rate):>8}   "
        f"{_pct(challenger_fr.flip_rate):>8}",
        "",
        f"  Δ CoT-faithfulness: [bold]{drift.delta:+.4f}[/bold] "
        f"({drift.delta_pct:+.1f}%)   p={drift.p_value:.4f}   "
        f"95% CI [{drift.ci_lower:+.4f}, {drift.ci_upper:+.4f}]",
    ]

    if drift.significant and drift.delta < 0:
        status = "[bold red]CoT-FAITHFULNESS REGRESSION[/bold red]"
        border = "red"
    elif drift.significant and drift.delta > 0:
        status = "[bold green]CoT-FAITHFULNESS IMPROVED[/bold green]"
        border = "green"
    else:
        status = "[bold]NO SIGNIFICANT CoT-FAITHFULNESS DRIFT[/bold]"
        border = "yellow"
    lines.append("")
    lines.append(f"  Status: {status}")

    console.print(Panel("\n".join(lines),
                        title="[bold]CoT-Dependence Faithfulness[/bold]",
                        border_style=border))

    if challenger_fr.perturb_stats:
        table = Table(title="By perturbation (challenger)", show_edge=False)
        table.add_column("perturbation")
        table.add_column("eligible", justify="right")
        table.add_column("flipped", justify="right")
        table.add_column("flip rate", justify="right")
        for name in sorted(challenger_fr.perturb_stats):
            st = challenger_fr.perturb_stats[name]
            table.add_row(name, str(st.n_eligible), str(st.n_flipped),
                          _pct(st.flip_rate))
        console.print(table)

    if challenger_fr.examples:
        ex_lines = [
            f"  case {idx} · {kind} · answer unchanged at '{ans}' despite corrupted CoT"
            for idx, kind, ans in challenger_fr.examples[:5]
        ]
        console.print(Panel("\n".join(ex_lines),
                            title="[dim]Post-hoc examples (answer unchanged when reasoning corrupted)[/dim]",
                            border_style="dim"))


def print_selftest_report(result, console: Console | None = None) -> None:
    """Print the null-calibration result from ``rift selftest``.

    ``result`` is a :class:`rift.selftest.SelfTestResult`. The headline is the
    false-regression rate — how often the gate would block a deploy comparing a
    model to itself — judged against the nominal ``alpha``.
    """
    if console is None:
        console = Console()
    r = result
    # A well-calibrated one-sided regression gate fires at ~alpha/2 under the
    # null; allow generous slack before calling it miscalibrated.
    expected_reg = r.alpha / 2.0
    if r.false_regression_rate > max(2 * expected_reg, expected_reg + 0.03):
        border = "red"
        verdict = "[bold red]GATE MISCALIBRATED ON THIS SUITE[/bold red]"
    elif r.false_positive_rate > r.alpha + 0.05:
        border = "yellow"
        verdict = "[bold yellow]Elevated two-sided false positives[/bold yellow]"
    else:
        border = "green"
        verdict = "[bold green]GATE WELL-CALIBRATED[/bold green]"

    lines = [
        f"  model:  {r.model}",
        f"  suite:  {r.suite_name}  ({r.n_cases} cases × {r.n_trials} trials)",
        f"  reps:   {r.reps} random self-vs-self splits",
        "",
        f"  {verdict}",
        "",
        f"  False-regression rate:  {r.false_regression_rate:.1%}   "
        f"(gate exit-1 vs an unchanged model; nominal ≈ {expected_reg:.1%})",
        f"  Two-sided FP rate:      {r.false_positive_rate:.1%}   "
        f"(nominal ≈ {r.alpha:.0%})",
        "",
        "  [dim]Noise band on the accuracy delta (self vs self):[/dim]",
        f"  mean |Δ|:  {r.mean_abs_delta:.4f}    "
        f"p95 |Δ|:  {r.p95_abs_delta:.4f}    max |Δ|:  {r.max_abs_delta:.4f}",
        "",
        "  [dim]Read: a real drift delta should clear the p95 band above. A"
        "\n  false-regression rate near the nominal means a red gate is"
        "\n  trustworthy on this suite; well above it means widen n or trials."
        "[/dim]",
    ]
    console.print(Panel("\n".join(lines),
                        title="[bold]Self-test — null calibration[/bold]",
                        border_style=border))


def print_replication_report(vc: dict, drift: DriftResult | None = None,
                             console: Console | None = None) -> None:
    """Print the run-to-run noise decomposition from a replicated run.

    ``vc`` is :func:`rift.comparator.variance_components`. When ``drift`` is
    given, compare the drift delta against the noise floor so the reader can
    see whether the headline change clears the band of simply re-running an
    unchanged model.
    """
    if console is None:
        console = Console()
    if vc.get("n_cases", 0) == 0 or vc.get("mean_trials", 0) < 2:
        return  # nothing to say without replication
    lines = [
        f"  Trials per case (mean):  {vc['mean_trials']:.1f}",
        f"  Run-to-run noise (SD):   {vc['mean_within_sd']:.4f}   "
        "(same model, same prompt, re-asked)",
        f"  Stable case spread (SD): {vc['between_case_var'] ** 0.5:.4f}",
        f"  ICC (signal fraction):   {vc['icc']:.3f}   "
        "(1 = reproducible, 0 = all noise)",
        f"  Noise floor on mean:     ±{vc['noise_floor']:.4f}",
    ]
    border = "blue"
    if drift is not None and vc["noise_floor"] > 0:
        ratio = abs(drift.delta) / vc["noise_floor"]
        lines += [
            "",
            f"  Drift delta:             {drift.delta:+.4f}",
            f"  Delta / noise floor:     {ratio:.1f}×",
        ]
        if ratio < 2.0:
            border = "yellow"
            lines.append(
                "  [yellow]Delta is within ~2× the run-to-run noise band — "
                "it may not\n  survive a re-run of the same models.[/yellow]"
            )
    if vc["icc"] < 0.5:
        border = "yellow"
        lines.append(
            "  [yellow]Low ICC: most variance is resampling noise, not stable\n"
            "  case differences. Add trials or cases before trusting a verdict."
            "[/yellow]"
        )
    console.print(Panel("\n".join(lines),
                        title="[bold]Replication / run-to-run noise[/bold]",
                        border_style=border))


def print_fingerprint_report(baseline: RunResult, challenger: RunResult,
                             console: Console | None = None) -> bool:
    """Surface the server-reported model fingerprints behind a comparison.

    Returns ``True`` when an integrity concern was flagged. Two checks:

    * **Alias collision.** Both sides resolved to the *same* single
      fingerprint despite different requested models — you may be comparing a
      model to itself (alias overlap, or a not-yet-distinct rollout), so any
      "no drift" verdict is trivially true and any "drift" is noise.
    * **Mid-run rollout.** Either side saw more than one fingerprint, meaning
      the served snapshot changed during that run.

    Silent (returns ``False``, prints nothing) when both sides report exactly
    one distinct fingerprint and they differ — the clean, comparable case —
    or when the provider exposes no fingerprint at all.
    """
    if console is None:
        console = Console()
    b_fps = sorted({c.provider_fingerprint for c in baseline.cases
                    if c.provider_fingerprint})
    c_fps = sorted({c.provider_fingerprint for c in challenger.cases
                    if c.provider_fingerprint})
    if not b_fps and not c_fps:
        return False  # provider exposes nothing; nothing to assert

    rollout = len(b_fps) > 1 or len(c_fps) > 1
    collision = (
        b_fps and c_fps and b_fps == c_fps and len(b_fps) == 1
        and baseline.model != challenger.model
    )
    lines = [
        f"  baseline   ({baseline.model}): {', '.join(b_fps) or '—'}",
        f"  challenger ({challenger.model}): {', '.join(c_fps) or '—'}",
    ]
    if collision:
        lines += [
            "",
            "  [bold]Both models resolved to the SAME served fingerprint.[/bold]",
            "  Different aliases, identical backend — this comparison cannot",
            "  show real drift. Check your model identifiers.",
        ]
    if rollout:
        lines += [
            "",
            "  [bold]A model returned multiple fingerprints[/bold] — the served",
            "  snapshot rolled over mid-run. Re-run once it settles.",
        ]
    if not collision and not rollout:
        return False
    console.print(Panel("\n".join(lines),
                        title="[bold yellow]⚠ Model fingerprint integrity[/bold yellow]",
                        border_style="yellow"))
    return True


def print_power_report(power: dict, alpha: float = 0.05) -> None:
    """Print a post-hoc power analysis."""
    console = Console()
    lines = [
        f"  Observed effect:        {power['observed_effect']:+.4f} "
        f"({power['observed_effect_kind']})",
        f"  Observed power:         {power['observed_power']:.1%}   "
        f"(at α={alpha})",
        f"  Min detectable effect:  {power['min_detectable_effect']:.4f}   "
        f"(at 80% power, α={alpha})",
    ]
    if power.get("n_for_target") is not None:
        lines.append(
            f"  N for target effect:    {power['n_for_target']} cases"
        )
    border = "blue"
    if power["observed_power"] < 0.5:
        border = "yellow"
    console.print(Panel("\n".join(lines),
                        title="[bold]Power Analysis[/bold]",
                        border_style=border))


def generate_markdown_report(drift: DriftResult, baseline: RunResult, challenger: RunResult) -> str:
    """Generate a markdown drift report suitable for blog posts."""
    if drift.significant and drift.delta < 0:
        status_emoji = "🔴"
        status_text = "Regression Detected"
    elif drift.significant and drift.delta > 0:
        status_emoji = "🟢"
        status_text = "Improvement Detected"
    else:
        status_emoji = "🔵"
        status_text = "No Significant Drift"

    lines = [
        f"# Rift Drift Report: {drift.baseline_model} vs {drift.challenger_model}",
        "",
        f"**Suite:** {drift.suite_name} ({drift.n_cases} cases)  ",
        f"**Status:** {status_emoji} {status_text}  ",
        f"**Test:** {drift.test_used}  ",
        "**Generated by [Rift](https://github.com/shahcolate/rift)**",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Baseline mean | {drift.baseline_mean:.4f} |",
        f"| Challenger mean | {drift.challenger_mean:.4f} |",
        f"| Delta | {drift.delta:+.4f} ({drift.delta_pct:+.1f}%) |",
        f"| p-value | {drift.p_value:.6f} |",
        f"| 95% CI | [{drift.ci_lower:+.4f}, {drift.ci_upper:+.4f}] |",
        f"| Effect size | {_fmt_effect(drift)} |",
        f"| Regressed cases | {len(drift.regressed_cases)} / {drift.n_cases} |",
        f"| Improved cases | {len(drift.improved_cases)} / {drift.n_cases} |",
    ]

    if drift.baseline_cost_usd or drift.challenger_cost_usd:
        lines += [
            "",
            "## Cost",
            "",
            "| Metric | Baseline | Challenger |",
            "|--------|----------|------------|",
            f"| Total spend | {_fmt_cost(drift.baseline_cost_usd)} | {_fmt_cost(drift.challenger_cost_usd)} |",
            f"| $/correct   | {_fmt_cost(drift.baseline_cost_per_correct)} | {_fmt_cost(drift.challenger_cost_per_correct)} |",
            f"| Δ $/correct | — | {drift.cost_normalized_delta_usd:+.4f} |",
        ]
        if getattr(drift, "cost_delta_ci_defined", False):
            lines.append(
                f"| 95% CI on Δ $/correct | — | "
                f"[{drift.cost_delta_ci_lower:+.4f}, "
                f"{drift.cost_delta_ci_upper:+.4f}] |"
            )

    if drift.subgroups:
        from .comparator import benjamini_hochberg
        keys = sorted(drift.subgroups.keys())
        p_values = [drift.subgroups[k].p_value for k in keys]
        q_values, _ = benjamini_hochberg(p_values)
        lines += [
            "",
            "## By Subgroup",
            "",
            "| Subgroup | n | Baseline | Challenger | Δ | Effect | p | q (BH) | 95% CI | Δ $/correct |",
            "|----------|---|----------|------------|---|--------|---|--------|--------|-------------|",
        ]
        for tag, q in zip(keys, q_values):
            d = drift.subgroups[tag]
            lines.append(
                f"| {tag} | {d.n_cases} | {d.baseline_mean:.3f} | "
                f"{d.challenger_mean:.3f} | {d.delta:+.3f} | "
                f"{_fmt_effect(d)} | {d.p_value:.4f} | {q:.4f} | "
                f"[{d.ci_lower:+.3f}, {d.ci_upper:+.3f}] | "
                f"{d.cost_normalized_delta_usd:+.4f} |"
            )

    if drift.regressed_cases:
        lines += [
            "",
            "## Regressed Cases",
            "",
            "| Case | Baseline | Challenger | Delta | Input |",
            "|------|----------|------------|-------|-------|",
        ]
        for idx in drift.regressed_cases[:20]:
            b = baseline.cases[idx]
            c = challenger.cases[idx]
            d = c.score - b.score
            inp = b.input_text[:60].replace("\n", " ").replace("|", "\\|")
            lines.append(f"| {idx} | {b.score:.4f} | {c.score:.4f} | {d:+.4f} | {inp} |")
        lines.append("")

    return "\n".join(lines)
