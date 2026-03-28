"""Rift CLI: Detect behavioral regressions between LLM model versions."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console

from .config import load_suite, resolve_model
from .runner import run_suite, RunResult
from .comparator import compare_runs
from .reporter import print_drift_report, generate_markdown_report

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="rift")
def main():
    """Rift: You upgraded your model. What broke?"""
    pass


@main.command()
@click.option("--baseline", required=True, help="Baseline model identifier")
@click.option("--challenger", required=True, help="Challenger model identifier")
@click.option("--suite", required=True, help="Eval suite name or path to YAML file")
@click.option("--concurrency", default=5, help="Max concurrent API calls")
@click.option("--alpha", default=0.05, help="Significance threshold")
@click.option("--output", "-o", default=None, help="Save comparison results to JSON")
@click.option("--report", "-r", default=None, help="Save markdown report to file")
@click.option("--cache-dir", default=None, help="Cache directory for completions")
def compare(baseline, challenger, suite, concurrency, alpha, output, report, cache_dir):
    """Compare two models on an eval suite."""
    suite_config = load_suite(suite)
    baseline_config = resolve_model(baseline)
    challenger_config = resolve_model(challenger)

    console.print(f"\n[bold]Rift[/bold] comparing [cyan]{baseline}[/cyan] vs [cyan]{challenger}[/cyan]")
    console.print(f"Suite: [yellow]{suite_config.name}[/yellow] ({len(suite_config.cases)} cases)\n")

    # Run both models
    baseline_result = asyncio.run(
        run_suite(suite_config, baseline_config, concurrency=concurrency, cache_dir=cache_dir)
    )
    challenger_result = asyncio.run(
        run_suite(suite_config, challenger_config, concurrency=concurrency, cache_dir=cache_dir)
    )

    # Compare
    drift = compare_runs(
        baseline_scores=baseline_result.scores,
        challenger_scores=challenger_result.scores,
        baseline_model=baseline,
        challenger_model=challenger,
        suite_name=suite_config.name,
        alpha=alpha,
    )

    # Print report
    print_drift_report(drift, baseline_result, challenger_result)

    # Save outputs
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

    # Exit code for CI/CD
    if drift.significant and drift.delta < 0:
        sys.exit(1)


@main.command()
@click.option("--model", required=True, help="Model identifier")
@click.option("--suite", required=True, help="Eval suite name or path")
@click.option("--concurrency", default=5, help="Max concurrent API calls")
@click.option("--output", "-o", required=True, help="Save run results to JSON")
@click.option("--cache-dir", default=None, help="Cache directory")
def run(model, suite, concurrency, output, cache_dir):
    """Run a single model against an eval suite and save results."""
    suite_config = load_suite(suite)
    model_config = resolve_model(model)

    console.print(f"\n[bold]Rift[/bold] running [cyan]{model}[/cyan]")
    console.print(f"Suite: [yellow]{suite_config.name}[/yellow] ({len(suite_config.cases)} cases)\n")

    result = asyncio.run(
        run_suite(suite_config, model_config, concurrency=concurrency, cache_dir=cache_dir)
    )

    result.save(output)
    console.print(f"\nMean score: [bold]{result.mean_score:.4f}[/bold]")
    console.print(f"Results saved to [green]{output}[/green]")


@main.command()
@click.argument("baseline_path")
@click.argument("challenger_path")
@click.option("--alpha", default=0.05, help="Significance threshold")
@click.option("--report", "-r", default=None, help="Save markdown report")
def diff(baseline_path, challenger_path, alpha, report):
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
    )

    print_drift_report(drift, baseline, challenger)

    if report:
        md = generate_markdown_report(drift, baseline, challenger)
        with open(report, "w") as f:
            f.write(md)
        console.print(f"Report saved to [green]{report}[/green]")

    if drift.significant and drift.delta < 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
