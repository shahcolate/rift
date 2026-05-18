"""Output formatting for drift reports."""

from __future__ import annotations

import math

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .comparator import DriftResult
from .runner import RunResult


_EFFECT_KIND_LABELS = {
    "cohens_h": "Cohen's h",
    "hedges_g": "Hedges' g",
    "smd": "SMD",
    "none": "n/a",
}


def _fmt_effect(drift) -> str:
    """Format an effect-size cell: value, kind, and magnitude bucket."""
    if drift.effect_size_kind == "none":
        return "n/a"
    label = _EFFECT_KIND_LABELS.get(drift.effect_size_kind, drift.effect_size_kind)
    return f"{drift.effect_size:+.3f} ({label}, {drift.effect_size_magnitude})"


def _fmt_cost(x: float) -> str:
    if x == float("inf") or math.isinf(x):
        return "∞"
    if x >= 1:
        return f"${x:,.2f}"
    return f"${x:.4f}"


def print_drift_report(drift: DriftResult, baseline: RunResult, challenger: RunResult) -> None:
    """Print a formatted drift report to the terminal."""
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

    if drift.baseline_cost_usd or drift.challenger_cost_usd:
        lines += [
            "",
            "  [dim]Cost (USD)[/dim]",
            f"  Baseline total:     {_fmt_cost(drift.baseline_cost_usd)}",
            f"  Challenger total:   {_fmt_cost(drift.challenger_cost_usd)}",
            f"  Baseline $/correct: {_fmt_cost(drift.baseline_cost_per_correct)}",
            f"  Challenger $/corr:  {_fmt_cost(drift.challenger_cost_per_correct)}",
        ]
        if drift.cost_normalized_delta_usd:
            arrow = "▲" if drift.cost_normalized_delta_usd > 0 else "▼"
            lines.append(
                f"  Δ $/correct:        {arrow} {_fmt_cost(abs(drift.cost_normalized_delta_usd))}"
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


def print_subgroup_table(subgroups: dict[str, DriftResult], title: str,
                          alpha: float = 0.05) -> None:
    """Render a subgroup comparison table (e.g. by distractor level).

    Adds a Benjamini–Hochberg adjusted q-value column so a reader of
    a many-subgroup table doesn't have to mentally bonferroni-correct
    on the fly. Significance highlighting uses the q-value, not the
    raw p-value.
    """
    from .comparator import benjamini_hochberg
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


def print_matrix(results: dict[tuple[str, str], DriftResult]) -> None:
    """Render an NxN model-vs-model matrix of drifts."""
    console = Console()
    models = sorted({m for pair in results for m in pair})
    table = Table(title="Model Drift Matrix  (cells: Δ mean  /  p  /  Δ$-per-correct)")
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
            cost = d.cost_normalized_delta_usd
            color = "red" if delta < 0 and d.significant else (
                "green" if delta > 0 and d.significant else "white"
            )
            cell = f"[{color}]{delta:+.3f}[/{color}]\np={p:.3f}"
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
