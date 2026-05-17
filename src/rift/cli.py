"""Rift CLI: Detect behavioral regressions between LLM model versions."""

from __future__ import annotations

import asyncio
import itertools
import sys
from pathlib import Path

import click
from rich.console import Console

from .calibration import compare_calibration
from .comparator import compare_runs, compare_by_subgroup, power_analysis
from .config import load_suite, resolve_model
from .context_rot import expand_suite
from .discovery import discover as discover_loop, to_suite_yaml
from .refusal import compare_refusal
from .reporter import (
    generate_markdown_report,
    print_calibration_report,
    print_drift_report,
    print_matrix,
    print_power_report,
    print_refusal_report,
    print_subgroup_table,
    print_sycophancy_report,
)
from .runner import RunResult, run_suite
from .sycophancy import build_pushback_suite, compute_sycophancy

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
@click.option("--refusal/--no-refusal", default=True,
              help="Also report refusal / over-refusal drift between the two runs.")
@click.option("--calibration/--no-calibration", default=False,
              help="Parse 'Confidence: X' from outputs and report Brier/ECE drift.")
@click.option("--power/--no-power", default=True,
              help="Include post-hoc power and minimum-detectable-effect analysis.")
@click.option("--judge-model", default=None,
              help="Judge model for llm_judge scoring. Overrides the suite's "
                   "`judge_model` field and $RIFT_JUDGE_MODEL.")
def compare(baseline, challenger, suite, concurrency, alpha, output, report,
            cache_dir, context_rot, enterprise_multiplier, subgroup,
            refusal, calibration, power, judge_model):
    """Compare two models on an eval suite."""
    suite_config = _maybe_expand(load_suite(suite), context_rot)
    if judge_model:
        # CLI override beats suite-level field beats env var.
        suite_config.judge_model = judge_model
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
        print_subgroup_table(drift.subgroups, title=f"By {subgroup}", alpha=alpha)

    refusal_analysis = None
    if refusal:
        refusal_analysis = compare_refusal(baseline_result, challenger_result)
        print_refusal_report(refusal_analysis)

    calibration_analysis = None
    if calibration:
        calibration_analysis = compare_calibration(baseline_result, challenger_result)
        print_calibration_report(calibration_analysis)

    power_result = None
    if power:
        power_result = power_analysis(
            baseline_result.scores, challenger_result.scores, alpha=alpha,
        )
        print_power_report(power_result, alpha=alpha)

    if output:
        import json
        from dataclasses import asdict

        results = {
            "drift": asdict(drift),
            "baseline": baseline_result.to_dict(),
            "challenger": challenger_result.to_dict(),
        }
        if refusal_analysis is not None:
            results["refusal"] = asdict(refusal_analysis)
        if calibration_analysis is not None:
            results["calibration"] = asdict(calibration_analysis)
        if power_result is not None:
            results["power"] = power_result
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
@click.option("--judge-model", default=None,
              help="Judge model for llm_judge scoring. Overrides the suite's "
                   "`judge_model` field and $RIFT_JUDGE_MODEL.")
def run(model, suite, concurrency, output, cache_dir, context_rot,
        enterprise_multiplier, judge_model):
    """Run a single model against an eval suite and save results."""
    suite_config = _maybe_expand(load_suite(suite), context_rot)
    if judge_model:
        suite_config.judge_model = judge_model
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
        print_subgroup_table(drift.subgroups, title=f"By {subgroup}", alpha=alpha)

    # Refusal + power are cheap and informative; run by default on diff
    # since the user has already chosen to compare two saved runs.
    print_refusal_report(compare_refusal(baseline, challenger))
    print_power_report(
        power_analysis(baseline.scores, challenger.scores, alpha=alpha),
        alpha=alpha,
    )

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


@main.command()
@click.argument("baseline_path")
@click.argument("challenger_path")
def refusal(baseline_path, challenger_path):
    """Refusal / over-refusal drift between two saved runs.

    No new API calls — operates on the already-collected outputs.
    """
    baseline = RunResult.load(baseline_path)
    challenger = RunResult.load(challenger_path)
    analysis = compare_refusal(baseline, challenger)
    print_refusal_report(analysis)


@main.command()
@click.argument("baseline_path")
@click.argument("challenger_path")
def calibration(baseline_path, challenger_path):
    """Calibration drift (Brier / ECE / overconfidence).

    Expects models to emit a confidence number (e.g. ``Confidence:
    0.85`` or ``I am 85% sure``) in their output. Cases without a
    parseable confidence are reported and excluded from the metrics.
    """
    baseline = RunResult.load(baseline_path)
    challenger = RunResult.load(challenger_path)
    comp = compare_calibration(baseline, challenger)
    print_calibration_report(comp)


@main.command()
@click.option("--model", required=True, help="Model identifier")
@click.option("--suite", required=True, help="Eval suite to probe")
@click.option("--concurrency", default=5)
@click.option("--cache-dir", default=None)
@click.option("--enterprise-multiplier", default=1.0, type=float)
def sycophancy(model, suite, concurrency, cache_dir, enterprise_multiplier):
    """Probe a model for sycophancy: does it fold under pushback?

    Runs the suite twice — once normally, then a follow-up suite
    generated from the model's own answers with adversarial
    pushback. Reports the flip rate among originally-correct cases.
    """
    suite_config = load_suite(suite)
    model_config = resolve_model(model)

    console.print(f"\n[bold]Rift[/bold] sycophancy probe on [cyan]{model}[/cyan]")
    console.print(
        f"Suite: [yellow]{suite_config.name}[/yellow] "
        f"({len(suite_config.cases)} cases)\n"
    )

    original = asyncio.run(
        run_suite(suite_config, model_config, concurrency=concurrency,
                  cache_dir=cache_dir,
                  enterprise_multiplier=enterprise_multiplier)
    )
    pushback_suite = build_pushback_suite(suite_config, original)
    pushback = asyncio.run(
        run_suite(pushback_suite, model_config, concurrency=concurrency,
                  cache_dir=cache_dir,
                  enterprise_multiplier=enterprise_multiplier)
    )
    analysis = compute_sycophancy(original, pushback)
    print_sycophancy_report(analysis)


@main.command()
@click.option("--baseline", required=True, help="Baseline model identifier")
@click.option("--challenger", required=True, help="Challenger model identifier")
@click.option("--seed-suite", "seed_suite", required=True,
              help="Seed suite name or path to YAML.")
@click.option("--proposer-model", required=True,
              help="Model used to propose candidate cases. Best practice: "
                   "a strong frontier model in a different family from both "
                   "the baseline and the challenger.")
@click.option("--target-power", default=0.9, type=float,
              help="Target power for the paired test on the discovered suite.")
@click.option("--target-effect", default=0.05, type=float,
              help="Target effect size (paired delta) at which to evaluate power.")
@click.option("--max-cases", default=50, type=int,
              help="Stop after this many accepted cases (bounds total spend).")
@click.option("--batch-size", default=16, type=int,
              help="Candidates per proposer batch.")
@click.option("--alpha", default=0.05, type=float,
              help="Significance threshold used in the power calculation.")
@click.option("--concurrency", default=5, type=int,
              help="Max concurrent verification API calls.")
@click.option("--cache-dir", default=None,
              help="Cache directory for completions (default .rift/cache).")
@click.option("--output", "-o", required=True,
              help="Path to write the discovered suite YAML.")
def discover(baseline, challenger, seed_suite, proposer_model,
             target_power, target_effect, max_cases, batch_size,
             alpha, concurrency, cache_dir, output):
    """Discover cases that maximize the paired test's power for a model pair.

    Uses a proposer model to generate candidate prompts, runs both
    baseline and challenger on each, and keeps the cases that
    contribute most to McNemar's test on the discovered suite.

    The output is a Rift-compatible suite YAML — feed it straight
    into ``rift compare``.
    """
    import yaml

    seed = load_suite(seed_suite)
    base_cfg = resolve_model(baseline)
    chal_cfg = resolve_model(challenger)

    console.print(
        f"\n[bold]Rift[/bold] discovering cases targeting "
        f"[cyan]{baseline}[/cyan] vs [cyan]{challenger}[/cyan]"
    )
    console.print(
        f"Seed: [yellow]{seed.name}[/yellow]  "
        f"Proposer: [yellow]{proposer_model}[/yellow]  "
        f"Target: power≥{target_power} at Δ={target_effect}, α={alpha}\n"
    )

    result = asyncio.run(discover_loop(
        baseline=base_cfg,
        challenger=chal_cfg,
        seed_suite=seed,
        proposer_model=proposer_model,
        target_power=target_power,
        target_effect=target_effect,
        max_cases=max_cases,
        batch_size=batch_size,
        alpha=alpha,
        concurrency=concurrency,
        cache_dir=cache_dir,
    ))

    suite_dict = to_suite_yaml(result)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        yaml.safe_dump(suite_dict, f, sort_keys=False, width=120)

    console.print(
        f"\n[bold]Discovered {result.n_kept} cases[/bold] "
        f"(from {result.n_proposed} proposed, "
        f"{result.n_after_dedup} after dedup, "
        f"{result.n_after_validity} after validity)."
    )
    console.print(
        f"  Discordant rate: {result.discordant_rate:.1%}"
    )
    console.print(
        f"  Achieved power:  {result.achieved_power:.2f}   "
        f"(target {result.target_power} at Δ={result.target_effect})"
    )
    console.print(
        f"  Spend: proposer ${result.proposer_spend_usd:.4f}, "
        f"verification ${result.verification_spend_usd:.4f}, "
        f"[bold]total ${result.proposer_spend_usd + result.verification_spend_usd:.4f}[/bold]"
    )
    console.print(
        "\n[dim]Note: cases were selected on divergence; "
        "achieved_power is the sensitivity of THIS suite, not an "
        "unbiased population estimate.[/dim]"
    )
    console.print(f"\nSuite saved to [green]{output}[/green]")
    console.print(
        f"Next step: [bold]rift compare --baseline {baseline} "
        f"--challenger {challenger} --suite {output}[/bold]"
    )


if __name__ == "__main__":
    main()
