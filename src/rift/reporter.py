"""Output formatting for drift reports."""

from __future__ import annotations

import math

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .comparator import DriftResult
from .runner import RunResult


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

    sr_arrow = (
        "▲" if drift.success_rate_delta > 0
        else ("▼" if drift.success_rate_delta < 0 else "=")
    )
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
        "",
        f"  Success rate (≥{drift.success_threshold:g}):",
        f"    baseline:    {drift.baseline_success_rate:.1%}",
        f"    challenger:  {drift.challenger_success_rate:.1%}",
        f"    Δ:           {sr_arrow} {drift.success_rate_delta:+.1%}",
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


def print_subgroup_table(subgroups: dict[str, DriftResult], title: str) -> None:
    """Render a subgroup comparison table (e.g. by distractor level).

    Shows both mean-score delta (what aggregate drift looks like) and
    success-rate delta (what fraction of cases each side completes).
    For long-workload subgroups the success-rate column is usually
    the more actionable signal.
    """
    console = Console()
    table = Table(title=title, show_lines=False)
    table.add_column("Subgroup", style="bold")
    table.add_column("n")
    table.add_column("Baseline")
    table.add_column("Challenger")
    table.add_column("Δ")
    table.add_column("Success (B→C)")
    table.add_column("Δ success")
    table.add_column("p-value")
    table.add_column("95% CI")
    table.add_column("$/correct Δ")

    # Stable sort by subgroup key for readability.
    for tag in sorted(subgroups.keys()):
        d = subgroups[tag]
        arrow = "▼" if d.delta < 0 else ("▲" if d.delta > 0 else "=")
        color = "red" if d.delta < 0 and d.significant else (
            "green" if d.delta > 0 and d.significant else "white"
        )
        sr_arrow = (
            "▲" if d.success_rate_delta > 0
            else ("▼" if d.success_rate_delta < 0 else "=")
        )
        sr_color = (
            "green" if d.success_rate_delta > 0
            else ("red" if d.success_rate_delta < 0 else "white")
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
            f"{d.baseline_success_rate:.0%} → {d.challenger_success_rate:.0%}",
            f"[{sr_color}]{sr_arrow} {d.success_rate_delta:+.1%}[/{sr_color}]",
            f"{d.p_value:.4f}",
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
        f"**Generated by [Rift](https://github.com/shahcolate/rift)**",
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
        f"| Baseline success rate (≥{drift.success_threshold:g}) | "
        f"{drift.baseline_success_rate:.1%} |",
        f"| Challenger success rate (≥{drift.success_threshold:g}) | "
        f"{drift.challenger_success_rate:.1%} |",
        f"| Δ success rate | {drift.success_rate_delta:+.1%} |",
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
        lines += [
            "",
            "## By Subgroup",
            "",
            "| Subgroup | n | Baseline | Challenger | Δ | Success (B→C) | "
            "Δ success | p | 95% CI | Δ $/correct |",
            "|----------|---|----------|------------|---|---------------|"
            "-----------|---|--------|-------------|",
        ]
        for tag in sorted(drift.subgroups.keys()):
            d = drift.subgroups[tag]
            lines.append(
                f"| {tag} | {d.n_cases} | {d.baseline_mean:.3f} | "
                f"{d.challenger_mean:.3f} | {d.delta:+.3f} | "
                f"{d.baseline_success_rate:.0%} → {d.challenger_success_rate:.0%} | "
                f"{d.success_rate_delta:+.1%} | "
                f"{d.p_value:.4f} | "
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
