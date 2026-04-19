"""Rift CLI: Detect behavioral regressions between LLM model versions."""

from __future__ import annotations

import asyncio
import itertools
import sys
from pathlib import Path

import click
from rich.console import Console

from .comparator import compare_runs, compare_by_subgroup
from .config import load_suite, resolve_model
from .context_rot import expand_suite
from .reporter import (
    generate_markdown_report,
    print_drift_report,
    print_matrix,
    print_subgroup_table,
)
from .runner import RunResult, run_suite

console = Console()


@click.group()
@click.version_option(version="0.2.0", prog_name="rift")
def main():
    """Rift: You upgraded your model. What broke?"""
    pass


def _maybe_expand(suite_config, context_rot: bool):
    return expand_suite(suite_config) if context_rot else suite_config


@main.command()
@click.option("--baseline", required=True, help="Baseline model identifier")
@click.option("--challenger", required=True, help="Challenger model identifier")
@click.option("--suite", required=True, help="Eval suite name or path to YAML file")
@click.option("--concurrency", default=5, help="Max concurrent API calls")
@click.option("--alpha", default=0.05, help="Significance threshold")
@click.option("--output", "-o", default=None, help="Save comparison results to JSON")
@click.option("--report", "-r", default=None, help="Save markdown report to file")
@click.option("--cache-dir", default=None, help="Cache directory for completions")
@click.option("--context-rot", is_flag=True, default=False,
              help="Expand suite with distractor-context variants per case.")
@click.option("--enterprise-multiplier", default=1.0, type=float,
              help="Apply a contracted-price multiplier to list pricing (e.g. 0.65).")
@click.option("--subgroup", default=None,
              help="Tag prefix to split cases by in the report (e.g. 'distractor:').")
def compare(baseline, challenger, suite, concurrency, alpha, output, report,
            cache_dir, context_rot, enterprise_multiplier, subgroup):
    """Compare two models on an eval suite."""
    suite_config = _maybe_expand(load_suite(suite), context_rot)
    baseline_config = resolve_model(baseline)
    challenger_config = resolve_model(challenger)

    console.print(
        f"\n[bold]Rift[/bold] comparing [cyan]{baseline}[/cyan] "
        f"vs [cyan]{challenger}[/cyan]"
    )
    console.print(
        f"Suite: [yellow]{suite_config.name}[/yellow] ({len(suite_config.cases)} cases)\n"
    )

    baseline_result = asyncio.run(
        run_suite(suite_config, baseline_config, concurrency=concurrency,
                  cache_dir=cache_dir, enterprise_multiplier=enterprise_multiplier)
    )
    challenger_result = asyncio.run(
        run_suite(suite_config, challenger_config, concurrency=concurrency,
                  cache_dir=cache_dir, enterprise_multiplier=enterprise_multiplier)
    )

    drift = compare_runs(
        baseline_scores=baseline_result.scores,
        challenger_scores=challenger_result.scores,
        baseline_model=baseline,
        challenger_model=challenger,
        suite_name=suite_config.name,
        alpha=alpha,
        baseline_costs=[c.cost_usd for c in baseline_result.cases],
        challenger_costs=[c.cost_usd for c in challenger_result.cases],
    )

    if subgroup:
        tags = [c.tags for c in baseline_result.cases]
        drift.subgroups = compare_by_subgroup(
            baseline_scores=baseline_result.scores,
            challenger_scores=challenger_result.scores,
            tags_per_case=tags,
            subgroup_prefix=subgroup,
            baseline_model=baseline,
            challenger_model=challenger,
            suite_name=suite_config.name,
            alpha=alpha,
            baseline_costs=[c.cost_usd for c in baseline_result.cases],
            challenger_costs=[c.cost_usd for c in challenger_result.cases],
        )

    print_drift_report(drift, baseline_result, challenger_result)
    if drift.subgroups:
        print_subgroup_table(drift.subgroups, title=f"By {subgroup}")

    if output:
        import json
        from dataclasses import asdict

        results = {
            "drift": asdict(drift),
            "baseline": baseline_result.to_dict(),
            "challenger": challenger_result.to_dict(),
        }
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        console.print(f"\nResults saved to [green]{output}[/green]")

    if report:
        md = generate_markdown_report(drift, baseline_result, challenger_result)
        Path(report).parent.mkdir(parents=True, exist_ok=True)
        with open(report, "w") as f:
            f.write(md)
        console.print(f"Report saved to [green]{report}[/green]")

    if drift.significant and drift.delta < 0:
        sys.exit(1)


@main.command()
@click.option("--model", required=True, help="Model identifier")
@click.option("--suite", required=True, help="Eval suite name or path")
@click.option("--concurrency", default=5, help="Max concurrent API calls")
@click.option("--output", "-o", required=True, help="Save run results to JSON")
@click.option("--cache-dir", default=None, help="Cache directory")
@click.option("--context-rot", is_flag=True, default=False,
              help="Expand suite with distractor-context variants per case.")
@click.option("--enterprise-multiplier", default=1.0, type=float)
def run(model, suite, concurrency, output, cache_dir, context_rot,
        enterprise_multiplier):
    """Run a single model against an eval suite and save results."""
    suite_config = _maybe_expand(load_suite(suite), context_rot)
    model_config = resolve_model(model)

    console.print(f"\n[bold]Rift[/bold] running [cyan]{model}[/cyan]")
    console.print(
        f"Suite: [yellow]{suite_config.name}[/yellow] ({len(suite_config.cases)} cases)\n"
    )

    result = asyncio.run(
        run_suite(suite_config, model_config, concurrency=concurrency,
                  cache_dir=cache_dir,
                  enterprise_multiplier=enterprise_multiplier)
    )

    result.save(output)
    console.print(f"\nMean score: [bold]{result.mean_score:.4f}[/bold]")
    console.print(f"Spend: [bold]${result.total_cost_usd:.4f}[/bold]  "
                  f"$/correct: [bold]${result.cost_per_correct():.4f}[/bold]")
    console.print(f"Results saved to [green]{output}[/green]")


@main.command()
@click.argument("baseline_path")
@click.argument("challenger_path")
@click.option("--alpha", default=0.05, help="Significance threshold")
@click.option("--report", "-r", default=None, help="Save markdown report")
@click.option("--subgroup", default=None,
              help="Tag prefix to split cases by in the report.")
def diff(baseline_path, challenger_path, alpha, report, subgroup):
    """Compare two saved run results."""
    baseline = RunResult.load(baseline_path)
    challenger = RunResult.load(challenger_path)

    drift = compare_runs(
        baseline_scores=baseline.scores,
        challenger_scores=challenger.scores,
        baseline_model=baseline.model,
        challenger_model=challenger.model,
        suite_name=baseline.suite_name,
        alpha=alpha,
        baseline_costs=[c.cost_usd for c in baseline.cases],
        challenger_costs=[c.cost_usd for c in challenger.cases],
    )

    if subgroup:
        drift.subgroups = compare_by_subgroup(
            baseline_scores=baseline.scores,
            challenger_scores=challenger.scores,
            tags_per_case=[c.tags for c in baseline.cases],
            subgroup_prefix=subgroup,
            baseline_model=baseline.model,
            challenger_model=challenger.model,
            suite_name=baseline.suite_name,
            alpha=alpha,
            baseline_costs=[c.cost_usd for c in baseline.cases],
            challenger_costs=[c.cost_usd for c in challenger.cases],
        )

    print_drift_report(drift, baseline, challenger)
    if drift.subgroups:
        print_subgroup_table(drift.subgroups, title=f"By {subgroup}")

    if report:
        md = generate_markdown_report(drift, baseline, challenger)
        with open(report, "w") as f:
            f.write(md)
        console.print(f"Report saved to [green]{report}[/green]")

    if drift.significant and drift.delta < 0:
        sys.exit(1)


@main.command()
@click.option("--models", required=True,
              help="Comma-separated list of model identifiers.")
@click.option("--suite", required=True, help="Eval suite name or path")
@click.option("--concurrency", default=5)
@click.option("--cache-dir", default=None)
@click.option("--context-rot", is_flag=True, default=False)
@click.option("--enterprise-multiplier", default=1.0, type=float)
@click.option("--output-dir", default=None,
              help="Directory to save per-model run JSONs.")
def matrix(models, suite, concurrency, cache_dir, context_rot,
           enterprise_multiplier, output_dir):
    """Run every model in ``--models`` and print an NxN drift matrix.

    Useful for: "how do Opus 4.7, Sonnet 4.6, and GPT-4o disagree on
    this suite?" — every pairwise comparison, one table.
    """
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    if len(model_list) < 2:
        raise click.UsageError("--models needs at least two identifiers.")

    suite_config = _maybe_expand(load_suite(suite), context_rot)

    runs: dict[str, RunResult] = {}
    for m in model_list:
        cfg = resolve_model(m)
        console.print(f"[bold]Running[/bold] [cyan]{m}[/cyan]")
        result = asyncio.run(
            run_suite(suite_config, cfg, concurrency=concurrency,
                      cache_dir=cache_dir,
                      enterprise_multiplier=enterprise_multiplier)
        )
        runs[m] = result
        if output_dir:
            out = Path(output_dir) / f"{m.replace('/', '_')}.json"
            result.save(out)

    comparisons: dict[tuple[str, str], object] = {}
    for base, chal in itertools.product(model_list, repeat=2):
        if base == chal:
            continue
        b = runs[base]
        c = runs[chal]
        comparisons[(base, chal)] = compare_runs(
            baseline_scores=b.scores,
            challenger_scores=c.scores,
            baseline_model=base,
            challenger_model=chal,
            suite_name=suite_config.name,
            baseline_costs=[x.cost_usd for x in b.cases],
            challenger_costs=[x.cost_usd for x in c.cases],
        )
    print_matrix(comparisons)  # type: ignore[arg-type]

    # Per-model summary row.
    from rich.table import Table
    tbl = Table(title="Per-model summary")
    tbl.add_column("Model", style="bold")
    tbl.add_column("Mean")
    tbl.add_column("n correct")
    tbl.add_column("Spend")
    tbl.add_column("$/correct")
    for m, r in runs.items():
        n_correct = sum(1 for c in r.cases if c.score >= 0.999)
        tbl.add_row(
            m,
            f"{r.mean_score:.4f}",
            f"{n_correct}/{len(r.cases)}",
            f"${r.total_cost_usd:.4f}",
            f"${r.cost_per_correct():.4f}" if n_correct else "∞",
        )
    console.print(tbl)


if __name__ == "__main__":
    main()
