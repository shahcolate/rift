"""Rift CLI: Detect behavioral regressions between LLM model versions."""

from __future__ import annotations

import asyncio
import itertools
import sys
from pathlib import Path

import click
from rich.console import Console

from . import __version__
from .calibration import compare_calibration
from .comparator import (
    compare_runs,
    compare_by_subgroup,
    power_analysis,
    variance_components,
)
from .config import load_suite, resolve_model
from .context_rot import expand_suite
from .demo import (
    SCENARIOS,
    export_demo_html,
    export_demo_markdown,
    export_demo_svg,
    load_scenario,
    run_demo,
)
from .discovery import discover as discover_loop, to_suite_yaml
from .keys import ensure_provider_keys, load_env, run_setup
from .refusal import compare_refusal
from .reporter import (
    generate_markdown_report,
    print_calibration_report,
    print_cot_faithfulness_report,
    print_drift_report,
    print_faithfulness_report,
    print_fingerprint_report,
    print_judge_validation_report,
    print_matrix,
    print_observation_report,
    print_preregistration_report,
    print_replication_report,
    print_selftest_report,
    print_power_report,
    print_refusal_report,
    print_subgroup_table,
    print_sycophancy_report,
)
from .runner import RunResult, run_suite
from .selftest import self_test
from .sycophancy import build_pushback_suite, compute_sycophancy
from .faithfulness import (
    build_control_suite,
    build_cot_perturbation_suite,
    build_faithfulness_suite,
    build_wrong_answer_suite,
    compute_cot_faithfulness,
    compute_faithfulness,
    parse_hint_targets,
)
from .scoring import get_scorer
from .scoring.faithfulness_judge import FaithfulnessJudge

console = Console()

# Exit-code contract (the CI integration surface):
#   0 — ran cleanly; no significant regression
#   1 — significant regression detected (the gate)
#   2 — operational error: bad arguments, missing/malformed files,
#       unknown model, missing API key, all cases errored
# Operational failures must never exit 1: a CI job that treats 1 as
# "regression" would misclassify infrastructure problems as model drift.


from ._errors import OperationalError  # noqa: E402 — the contract's base class


def _load_run(path: str) -> RunResult:
    """Load a saved run, mapping file problems to clean exit-2 messages."""
    try:
        return RunResult.load(path)
    except FileNotFoundError:
        raise OperationalError(f"Run file not found: {path}") from None
    except (ValueError, KeyError, TypeError) as e:
        # json.JSONDecodeError subclasses ValueError.
        raise OperationalError(
            f"Not a Rift run file: {path} ({e})"
        ) from None


def _reject_all_errored(**runs: RunResult) -> None:
    """Exit 2 when every case on a side errored.

    An all-errored run means the provider was unreachable or every call
    failed — infrastructure, not drift. Scoring it (all zeros) could even
    exit 0 and green-light a CI gate.
    """
    for side, result in runs.items():
        if result.cases and all(c.error for c in result.cases):
            first = next(c.error for c in result.cases if c.error)
            raise OperationalError(
                f"Every {side} case errored (first: {first}). This is an "
                "infrastructure failure, not drift — no verdict computed."
            )


@click.group()
@click.version_option(version=__version__, prog_name="rift")
def main():
    """Rift: You upgraded your model. What broke?"""
    # Load saved API keys (~/.rift/.env, then ./.env) before any command
    # runs. Real env vars are never overridden.
    load_env()


@main.command()
def setup():
    """Save your model-provider API keys so live commands just work.

    Walks through Anthropic / OpenAI / Google, stores whatever you paste
    in ~/.rift/.env (readable only by you), and loads it automatically on
    every future run. You only need keys for the providers you actually
    compare against; the demo runs without any.
    """
    run_setup(console)


def _maybe_expand(suite_config, context_rot: bool):
    return expand_suite(suite_config) if context_rot else suite_config


@main.command()
@click.option("--baseline", required=True, help="Baseline model identifier")
@click.option("--challenger", required=True, help="Challenger model identifier")
@click.option("--suite", required=True, help="Eval suite name or path to YAML file")
@click.option("--concurrency", default=5, show_default=True,
              help="Max concurrent API calls")
@click.option("--alpha", default=0.05, show_default=True,
              type=click.FloatRange(0, 1, min_open=True, max_open=True),
              help="Significance threshold")
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
@click.option("--trials", default=1, type=int,
              help="Replicates per case (default 1). >1 re-samples each case to "
                   "measure run-to-run generation noise and report whether the "
                   "drift delta clears that noise band. NOTE: with >1, per-case "
                   "scores become trial means (continuous), so the paired test "
                   "switches from McNemar to the paired t-test and $/correct "
                   "counts only all-trials-correct cases.")
@click.option("--preregister", default=None,
              help="Path to a pre-registration YAML pinning the primary "
                   "endpoint. The headline + exit code bind to it; all other "
                   "numbers become exploratory.")
@click.option("--judge-model", default=None,
              help="Judge model for llm_judge scoring. Overrides the suite's "
                   "`judge_model` field and $RIFT_JUDGE_MODEL.")
@click.option("--strip-io", is_flag=True, default=False,
              help="When writing --output, omit per-case input_text and "
                   "output fields. Use for proprietary suites whose prompts "
                   "or completions should not leave your machine.")
@click.option("--metrics-out", default=None,
              help="Write flat drift metrics to PATH for dashboards "
                   "(node_exporter / time-series stores). See --metrics-format.")
@click.option("--metrics-format", type=click.Choice(["json", "prometheus"]),
              default="json", show_default=True,
              help="Format for --metrics-out.")
def compare(baseline, challenger, suite, concurrency, alpha, output, report,
            cache_dir, context_rot, enterprise_multiplier, subgroup,
            refusal, calibration, power, trials, preregister, judge_model,
            strip_io, metrics_out, metrics_format):
    """Compare two models on an eval suite.

    \b
    Exit codes (the CI contract):
      0  no significant regression
      1  significant regression detected
      2  operational error (bad model/suite/key, all cases errored)
    """
    if trials < 1:
        raise click.UsageError("--trials must be >= 1.")
    prereg = None
    if preregister:
        from .preregistration import load_preregistration
        prereg = load_preregistration(preregister)
        # A pre-registered alpha governs the primary endpoint.
        alpha = prereg.alpha
    suite_config = _maybe_expand(load_suite(suite), context_rot)
    if judge_model:
        # CLI override beats suite-level field beats env var.
        suite_config.judge_model = judge_model
    baseline_config = resolve_model(baseline)
    challenger_config = resolve_model(challenger)
    ensure_provider_keys(
        [baseline_config.provider, challenger_config.provider], console
    )

    console.print(
        f"\n[bold]Rift[/bold] comparing [cyan]{baseline}[/cyan] "
        f"vs [cyan]{challenger}[/cyan]"
    )
    console.print(
        f"Suite: [yellow]{suite_config.name}[/yellow] ({len(suite_config.cases)} cases)\n"
    )

    baseline_result = asyncio.run(
        run_suite(suite_config, baseline_config, concurrency=concurrency,
                  cache_dir=cache_dir, enterprise_multiplier=enterprise_multiplier,
                  trials=trials)
    )
    challenger_result = asyncio.run(
        run_suite(suite_config, challenger_config, concurrency=concurrency,
                  cache_dir=cache_dir, enterprise_multiplier=enterprise_multiplier,
                  trials=trials)
    )
    _reject_all_errored(baseline=baseline_result, challenger=challenger_result)

    drift = compare_runs(
        baseline_scores=baseline_result.scores,
        challenger_scores=challenger_result.scores,
        baseline_model=baseline_result.model,
        challenger_model=challenger_result.model,
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
            baseline_model=baseline_result.model,
            challenger_model=challenger_result.model,
            suite_name=suite_config.name,
            alpha=alpha,
            baseline_costs=[c.cost_usd for c in baseline_result.cases],
            challenger_costs=[c.cost_usd for c in challenger_result.cases],
        )

    print_drift_report(drift, baseline_result, challenger_result)
    print_fingerprint_report(baseline_result, challenger_result)

    prereg_outcome = None
    if prereg is not None:
        from .preregistration import evaluate as _eval_prereg
        prereg_outcome = _eval_prereg(
            prereg, drift, drift.n_cases,
            baseline_model=baseline, challenger_model=challenger,
        )
        print_preregistration_report(prereg_outcome)

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

    replication = None
    if trials > 1:
        # Pool both sides' per-case trial scores: the noise floor is a property
        # of the measurement, estimated from all available replicates.
        pooled = (
            [c.trial_scores for c in baseline_result.cases if c.trial_scores]
            + [c.trial_scores for c in challenger_result.cases if c.trial_scores]
        )
        replication = variance_components(pooled)
        print_replication_report(replication, drift)

    if output:
        import json
        from dataclasses import asdict

        results = {
            "drift": asdict(drift),
            "baseline": baseline_result.to_dict(strip_io=strip_io),
            "challenger": challenger_result.to_dict(strip_io=strip_io),
        }
        if strip_io:
            results["_strip_io"] = True
        if refusal_analysis is not None:
            results["refusal"] = asdict(refusal_analysis)
        if calibration_analysis is not None:
            results["calibration"] = asdict(calibration_analysis)
        if power_result is not None:
            results["power"] = power_result
        if replication is not None:
            results["replication"] = replication
        if prereg_outcome is not None:
            results["preregistration"] = asdict(prereg_outcome)
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

    if metrics_out:
        from .observability import comparison_metrics, write_metrics

        write_metrics(comparison_metrics(drift), metrics_out, metrics_format)
        console.print(
            f"Metrics ([cyan]{metrics_format}[/cyan]) saved to "
            f"[green]{metrics_out}[/green]"
        )

    # When pre-registered, the gate binds to the declared primary endpoint —
    # not whichever axis happened to look significant. Otherwise fall back to
    # the default accuracy-regression gate.
    if prereg_outcome is not None:
        if prereg_outcome.adverse_confirmed and prereg_outcome.direction != "improvement":
            sys.exit(1)
    elif drift.significant and drift.delta < 0:
        sys.exit(1)


@main.command()
@click.option("--model", required=True, help="Model identifier")
@click.option("--suite", required=True, help="Eval suite name or path")
@click.option("--concurrency", default=5, help="Max concurrent API calls")
@click.option("--output", "-o", required=True, help="Save run results to JSON")
@click.option("--cache-dir", default=None, help="Cache directory")
@click.option("--context-rot", is_flag=True, default=False,
              help="Expand suite with distractor-context variants per case.")
@click.option("--enterprise-multiplier", default=1.0, type=float,
              show_default=True,
              help="Apply a contracted-price multiplier to list pricing.")
@click.option("--trials", default=1, type=int,
              help="Replicates per case (default 1). >1 re-samples each case so "
                   "the saved run carries per-trial scores for noise analysis.")
@click.option("--judge-model", default=None,
              help="Judge model for llm_judge scoring. Overrides the suite's "
                   "`judge_model` field and $RIFT_JUDGE_MODEL.")
@click.option("--strip-io", is_flag=True, default=False,
              help="Omit per-case input_text and output from the saved "
                   "JSON. Use for proprietary suites.")
@click.option("--metrics-out", default=None,
              help="Write flat run metrics to PATH for dashboards. "
                   "See --metrics-format.")
@click.option("--metrics-format", type=click.Choice(["json", "prometheus"]),
              default="json", show_default=True,
              help="Format for --metrics-out.")
def run(model, suite, concurrency, output, cache_dir, context_rot,
        enterprise_multiplier, trials, judge_model, strip_io,
        metrics_out, metrics_format):
    """Run a single model against an eval suite and save results."""
    if trials < 1:
        raise click.UsageError("--trials must be >= 1.")
    suite_config = _maybe_expand(load_suite(suite), context_rot)
    if judge_model:
        suite_config.judge_model = judge_model
    model_config = resolve_model(model)
    ensure_provider_keys([model_config.provider], console)

    console.print(f"\n[bold]Rift[/bold] running [cyan]{model}[/cyan]")
    console.print(
        f"Suite: [yellow]{suite_config.name}[/yellow] ({len(suite_config.cases)} cases)\n"
    )

    result = asyncio.run(
        run_suite(suite_config, model_config, concurrency=concurrency,
                  cache_dir=cache_dir,
                  enterprise_multiplier=enterprise_multiplier, trials=trials)
    )

    result.save(output, strip_io=strip_io)
    if trials > 1:
        vc = variance_components(
            [c.trial_scores for c in result.cases if c.trial_scores]
        )
        print_replication_report(vc)
    console.print(f"\nMean score: [bold]{result.mean_score:.4f}[/bold]")
    cpc = result.cost_per_correct()
    cpc_str = f"${cpc:.4f}" if cpc != float("inf") else "n/a (0 correct)"
    console.print(f"Spend: [bold]${result.total_cost_usd:.4f}[/bold]  "
                  f"$/correct: [bold]{cpc_str}[/bold]")
    console.print(f"Results saved to [green]{output}[/green]")
    if metrics_out:
        from .observability import run_metrics, write_metrics

        write_metrics(run_metrics(result), metrics_out, metrics_format)
        console.print(
            f"Metrics ([cyan]{metrics_format}[/cyan]) saved to "
            f"[green]{metrics_out}[/green]"
        )


@main.command()
@click.argument("baseline_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("challenger_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--alpha", default=0.05, show_default=True,
              type=click.FloatRange(0, 1, min_open=True, max_open=True),
              help="Significance threshold")
@click.option("--report", "-r", default=None, help="Save markdown report")
@click.option("--subgroup", default=None,
              help="Tag prefix to split cases by in the report.")
def diff(baseline_path, challenger_path, alpha, report, subgroup):
    """Compare two saved run results.

    \b
    Exit codes: 0 = no significant regression, 1 = significant
    regression, 2 = operational error (bad file, bad arguments).
    """
    baseline = _load_run(baseline_path)
    challenger = _load_run(challenger_path)
    _reject_all_errored(baseline=baseline, challenger=challenger)

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
    print_fingerprint_report(baseline, challenger)
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
@click.option("--concurrency", default=5, show_default=True,
              help="Max concurrent API calls")
@click.option("--cache-dir", default=None,
              help="Cache directory for completions")
@click.option("--context-rot", is_flag=True, default=False,
              help="Expand suite with distractor-context variants per case.")
@click.option("--enterprise-multiplier", default=1.0, type=float,
              show_default=True,
              help="Apply a contracted-price multiplier to list pricing.")
@click.option("--output-dir", default=None,
              help="Directory to save per-model run JSONs.")
@click.option("--strip-io", is_flag=True, default=False,
              help="When writing per-model JSONs, omit input_text and "
                   "output fields. Use for proprietary suites.")
def matrix(models, suite, concurrency, cache_dir, context_rot,
           enterprise_multiplier, output_dir, strip_io):
    """Run every model in --models and print an NxN drift matrix.

    Useful for: "how do Opus 4.7, Sonnet 4.6, and GPT-4o disagree on
    this suite?" — every pairwise comparison, one table.
    """
    model_list = [m.strip() for m in models.split(",") if m.strip()]
    if len(model_list) < 2:
        raise click.UsageError("--models needs at least two identifiers.")

    suite_config = _maybe_expand(load_suite(suite), context_rot)
    ensure_provider_keys([resolve_model(m).provider for m in model_list], console)

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
            result.save(out, strip_io=strip_io)

    comparisons: dict[tuple[str, str], object] = {}
    for base, chal in itertools.product(model_list, repeat=2):
        if base == chal:
            continue
        b = runs[base]
        c = runs[chal]
        comparisons[(base, chal)] = compare_runs(
            baseline_scores=b.scores,
            challenger_scores=c.scores,
            baseline_model=b.model,
            challenger_model=c.model,
            suite_name=suite_config.name,
            baseline_costs=[x.cost_usd for x in b.cases],
            challenger_costs=[x.cost_usd for x in c.cases],
        )
    # Surface per-model API errors loudly before printing the matrix.
    # Without this, a model that errored on every case (e.g. missing
    # API key, all requests refused) shows up as "0.0000 mean" with no
    # indication of why — and any drift cell against it is meaningless.
    error_rows = [
        (m, r.metadata.get("n_errors", 0), len(r.cases))
        for m, r in runs.items()
        if r.metadata.get("n_errors", 0) > 0
    ]
    if error_rows:
        from rich.panel import Panel
        lines = [
            "  One or more models had API errors during this run. Their",
            "  scores are biased downward (errored cases are counted as 0)",
            "  and any drift cell against them is unreliable.",
            "",
        ]
        for m, n_err, n_total in error_rows:
            lines.append(f"    {m}: {n_err}/{n_total} cases errored")
        console.print(Panel("\n".join(lines),
                            title="[bold yellow]⚠ Run integrity warning[/bold yellow]",
                            border_style="yellow"))

    print_matrix(comparisons)  # type: ignore[arg-type]

    # Per-model summary row.
    from rich.table import Table
    tbl = Table(title="Per-model summary")
    tbl.add_column("Model", style="bold")
    tbl.add_column("Mean")
    tbl.add_column("n correct")
    tbl.add_column("Errors")
    tbl.add_column("Spend")
    tbl.add_column("$/correct")
    for m, r in runs.items():
        n_correct = sum(1 for c in r.cases if c.score >= 0.999)
        n_err = r.metadata.get("n_errors", 0)
        err_cell = f"[red]{n_err}[/red]" if n_err else "0"
        tbl.add_row(
            m,
            f"{r.mean_score:.4f}",
            f"{n_correct}/{len(r.cases)}",
            err_cell,
            f"${r.total_cost_usd:.4f}",
            f"${r.cost_per_correct():.4f}" if n_correct else "∞",
        )
    console.print(tbl)


@main.command()
@click.argument("baseline_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("challenger_path", type=click.Path(exists=True, dir_okay=False))
def refusal(baseline_path, challenger_path):
    """Refusal / over-refusal drift between two saved runs.

    No new API calls — operates on the already-collected outputs.
    """
    baseline = _load_run(baseline_path)
    challenger = _load_run(challenger_path)
    analysis = compare_refusal(baseline, challenger)
    print_refusal_report(analysis)


@main.command()
@click.argument("baseline_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("challenger_path", type=click.Path(exists=True, dir_okay=False))
def calibration(baseline_path, challenger_path):
    """Calibration drift (Brier / ECE / overconfidence).

    Expects models to emit a confidence number (e.g. 'Confidence: 0.85'
    or 'I am 85% sure') in their output. Cases without a parseable
    confidence are reported and excluded from the metrics.
    """
    baseline = _load_run(baseline_path)
    challenger = _load_run(challenger_path)
    comp = compare_calibration(baseline, challenger)
    print_calibration_report(comp)


@main.command()
@click.option("--model", required=True, help="Model identifier")
@click.option("--suite", required=True, help="Eval suite name or path")
@click.option("--trials", default=5, type=int,
              help="Replicates per case (default 5, min 2). More trials give a "
                   "tighter estimate of the null false-positive rate.")
@click.option("--reps", default=500, type=int,
              help="Random self-vs-self splits used to estimate the rate.")
@click.option("--alpha", default=0.05, show_default=True,
              type=click.FloatRange(0, 1, min_open=True, max_open=True),
              help="Significance threshold.")
@click.option("--concurrency", default=5, help="Max concurrent API calls")
@click.option("--cache-dir", default=None, help="Cache directory for completions")
@click.option("--output", "-o", default=None, help="Save the run + result to JSON")
def selftest(model, suite, trials, reps, alpha, concurrency, cache_dir, output):
    """Calibrate the drift gate: how often does it fire on an UNCHANGED model?

    Runs one model against a suite with replication, then repeatedly splits its
    own trials into two arms and feeds them through the same statistical test
    'compare' uses. Reports the empirical false-positive rate — most
    importantly the false-*regression* rate, i.e. how often the CI gate would
    block a deploy comparing a model to itself. A rate near the nominal alpha
    means a red gate is trustworthy on this suite; well above it means you need
    more cases or trials before gating on this suite.
    """
    if trials < 2:
        raise click.UsageError("--trials must be >= 2 for self-test.")
    suite_config = load_suite(suite)
    model_config = resolve_model(model)
    ensure_provider_keys([model_config.provider], console)

    console.print(f"\n[bold]Rift[/bold] self-test (null calibration) on "
                  f"[cyan]{model}[/cyan]")
    console.print(
        f"Suite: [yellow]{suite_config.name}[/yellow] "
        f"({len(suite_config.cases)} cases × {trials} trials)\n"
    )

    result = asyncio.run(
        run_suite(suite_config, model_config, concurrency=concurrency,
                  cache_dir=cache_dir, trials=trials)
    )
    print_fingerprint_report(result, result)
    trial_scores = [c.trial_scores for c in result.cases if len(c.trial_scores) >= 2]
    if not trial_scores:
        raise click.ClickException(
            "No case produced >=2 successful trials; cannot calibrate. "
            "Check for API errors above."
        )
    st = self_test(trial_scores, model_config.model, suite_config.name,
                   alpha=alpha, reps=reps)
    print_selftest_report(st)

    if output:
        import json
        from dataclasses import asdict
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump({"selftest": asdict(st),
                       "run": result.to_dict()}, f, indent=2, default=str)
        console.print(f"\nResults saved to [green]{output}[/green]")


@main.command()
@click.option("--model", required=True, help="Model identifier")
@click.option("--suite", required=True, help="Eval suite to probe")
@click.option("--concurrency", default=5, show_default=True,
              help="Max concurrent API calls")
@click.option("--cache-dir", default=None,
              help="Cache directory for completions")
@click.option("--enterprise-multiplier", default=1.0, type=float,
              show_default=True,
              help="Apply a contracted-price multiplier to list pricing.")
def sycophancy(model, suite, concurrency, cache_dir, enterprise_multiplier):
    """Probe a model for sycophancy: does it fold under pushback?

    Runs the suite twice — once normally, then a follow-up suite
    generated from the model's own answers with adversarial
    pushback. Reports the flip rate among originally-correct cases.
    """
    suite_config = load_suite(suite)
    model_config = resolve_model(model)
    ensure_provider_keys([model_config.provider], console)

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
    # An all-errored original run would probe pushback against error text
    # and report a meaningless flip rate — that's an outage, exit 2.
    _reject_all_errored(original=original)
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
@click.option("--suite", required=True, help="Eval suite name or path to YAML file")
@click.option("--judge-model", default=None,
              help="Model that judges whether reasoning acknowledged the cue. "
                   "Best practice: a third model family. Defaults to "
                   "$RIFT_JUDGE_MODEL or the built-in default.")
@click.option("--proposer-model", default=None,
              help="Model that invents a plausible-wrong target per case. "
                   "Defaults to the challenger.")
@click.option("--mode", type=click.Choice(["hint", "cot", "both"]), default="hint",
              help="hint: biasing-cue articulation (default). cot: chain-of-"
                   "thought dependence (does corrupting the CoT change the "
                   "answer?). both: run each and report separately.")
@click.option("--cues", default=None,
              help="hint mode: comma-separated subset of cues to apply "
                   "(suggested,authority,consensus). Default: all.")
@click.option("--cot-perturbations", "cot_perturbations", default=None,
              help="cot mode: comma-separated subset of perturbations "
                   "(early,mistake). Default: all.")
@click.option("--concurrency", default=5, help="Max concurrent API calls")
@click.option("--alpha", default=0.05, show_default=True,
              type=click.FloatRange(0, 1, min_open=True, max_open=True),
              help="Significance threshold")
@click.option("--cache-dir", default=None, help="Cache directory for completions")
@click.option("--output", "-o", default=None, help="Save results to JSON")
def faithfulness(baseline, challenger, suite, judge_model, proposer_model,
                 mode, cues, cot_perturbations, concurrency, alpha, cache_dir,
                 output):
    """Measure reasoning-faithfulness drift between two models.

    Two modes (run one or --mode both):

    \b
    hint  Plants a biasing cue ("a professor says the answer is X") pointing
          at a plausible-wrong answer, then checks how often each model is
          silently swayed WITHOUT its reasoning admitting the cue.
    cot   Captures each model's chain-of-thought, then re-asks under a
          truncated / corrupted version of it. A faithful model's answer
          changes when its reasoning is corrupted; a post-hoc one's does not.

    Reports the drift in faithfulness with significance + CI. Exit 1 on a
    significant regression in any mode that ran.
    """
    base_suite = load_suite(suite)
    if base_suite.scoring not in ("exact_match", "fuzzy_match"):
        raise click.ClickException(
            f"rift faithfulness supports exact_match or fuzzy_match suites; "
            f"'{base_suite.name}' uses scoring='{base_suite.scoring}'."
        )
    base_cfg = resolve_model(baseline)
    chal_cfg = resolve_model(challenger)
    proposer_cfg = resolve_model(proposer_model) if proposer_model else chal_cfg
    cue_list = [c.strip() for c in cues.split(",") if c.strip()] if cues else None
    cot_list = (
        [c.strip() for c in cot_perturbations.split(",") if c.strip()]
        if cot_perturbations else None
    )

    # Suite-level prompt overrides thread into the judge and the suite builders.
    prompts_block = base_suite.prompts
    judge = FaithfulnessJudge(
        judge_model=judge_model, cache_dir=cache_dir,
        prompt_template=prompts_block.get("faithfulness_judge"),
    )
    judge_cfg = resolve_model(judge.judge_model)
    # hint mode needs the proposer + judge; cot mode needs neither. Preflight
    # only the providers the chosen mode(s) will actually call.
    needed = [base_cfg.provider, chal_cfg.provider]
    if mode in ("hint", "both"):
        needed += [proposer_cfg.provider, judge_cfg.provider]
    ensure_provider_keys(needed, console)

    console.print(
        f"\n[bold]Rift[/bold] faithfulness ([magenta]{mode}[/magenta]): "
        f"[cyan]{baseline}[/cyan] vs [cyan]{challenger}[/cyan]"
    )
    console.print(
        f"Suite: [yellow]{base_suite.name}[/yellow] "
        f"({len(base_suite.cases)} cases)\n"
    )

    scorer = get_scorer(base_suite.scoring)
    regressed = False
    out_payload: dict = {}

    if mode in ("hint", "both"):
        regressed |= _run_hint_mode(
            base_suite, base_cfg, chal_cfg, proposer_cfg, judge, scorer,
            cue_list, concurrency, alpha, cache_dir,
            out_payload, prompts_block,
        )
    if mode in ("cot", "both"):
        regressed |= _run_cot_mode(
            base_suite, base_cfg, chal_cfg, scorer,
            cot_list, concurrency, alpha, cache_dir, out_payload, prompts_block,
        )

    asyncio.run(judge.close())

    if output:
        import json
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(out_payload, f, indent=2, default=str)
        console.print(f"\nResults saved to [green]{output}[/green]")

    if regressed:
        sys.exit(1)


def _run_hint_mode(base_suite, base_cfg, chal_cfg, proposer_cfg, judge, scorer,
                   cue_list, concurrency, alpha, cache_dir,
                   out_payload, prompts_block) -> bool:
    """Hint-articulation mode. Returns True if a significant regression was found."""
    from dataclasses import asdict

    from .prompts import resolve_cues

    console.print(f"[dim]hint mode · judge: {judge.judge_model}[/dim]")
    wrong_suite = build_wrong_answer_suite(
        base_suite,
        wrong_answer_prompt=prompts_block.get("faithfulness_wrong_answer"),
    )
    wrong_run = asyncio.run(
        run_suite(wrong_suite, proposer_cfg, concurrency=concurrency,
                  cache_dir=cache_dir)
    )
    hint_targets = parse_hint_targets(wrong_run, base_suite=base_suite)
    n_dropped = len(base_suite.cases) - len(hint_targets)
    if n_dropped:
        console.print(
            f"  [yellow]{n_dropped} case(s) excluded: proposer produced the "
            "correct answer (or nothing) — no usable wrong-answer cue.[/yellow]"
        )

    # Resolve cue templates once and reuse for both the suite build and the
    # judge-side cue reconstruction in compute_faithfulness, so an overridden
    # or newly-added cue is judged against the exact text the model saw.
    cue_templates = resolve_cues(base_suite.cues)
    derived = build_faithfulness_suite(
        base_suite, hint_targets, cues=cue_list,
        cue_templates=cue_templates,
        format_instruction=prompts_block.get("faithfulness_format_instruction"),
    )
    base_run = asyncio.run(
        run_suite(derived, base_cfg, concurrency=concurrency, cache_dir=cache_dir)
    )
    chal_run = asyncio.run(
        run_suite(derived, chal_cfg, concurrency=concurrency, cache_dir=cache_dir)
    )
    # An outage must exit 2, not slip through as "no shared control-correct
    # cases" → no regression → exit 0.
    _reject_all_errored(baseline=base_run, challenger=chal_run)

    def _ack(question, cue_text, reasoning, answer, target) -> bool:
        return asyncio.run(
            judge.acknowledged(question, cue_text, reasoning, answer, target)
        )

    base_fr = compute_faithfulness(base_run, scorer, _ack, hint_targets,
                                   cue_templates=cue_templates)
    chal_fr = compute_faithfulness(chal_run, scorer, _ack, hint_targets,
                                   cue_templates=cue_templates)

    shared = sorted(set(base_fr.per_case) & set(chal_fr.per_case))
    if not shared:
        console.print(
            "\n[yellow]hint: no cases answered correctly by both models in "
            "control,[/yellow] so faithfulness drift is undefined."
        )
        return False
    drift = compare_runs(
        baseline_scores=[base_fr.per_case[i] for i in shared],
        challenger_scores=[chal_fr.per_case[i] for i in shared],
        baseline_model=base_cfg.model,
        challenger_model=chal_cfg.model,
        suite_name=f"{base_suite.name} (faithfulness/hint)",
        alpha=alpha,
    )
    print_faithfulness_report(base_fr, chal_fr, drift, alpha=alpha)
    out_payload["hint"] = {
        "drift": asdict(drift),
        "baseline": asdict(base_fr),
        "challenger": asdict(chal_fr),
        "hint_targets": hint_targets,
    }
    return drift.significant and drift.delta < 0


def _run_cot_mode(base_suite, base_cfg, chal_cfg, scorer,
                  cot_list, concurrency, alpha, cache_dir, out_payload,
                  prompts_block) -> bool:
    """CoT-dependence mode. Returns True if a significant regression was found."""
    from dataclasses import asdict

    fmt = prompts_block.get("faithfulness_format_instruction")
    early_tpl = prompts_block.get("faithfulness_cot_early")
    mistake_tpl = prompts_block.get("faithfulness_cot_mistake")

    console.print("[dim]cot mode · no judge/proposer needed[/dim]")
    # 1. Control run per model captures each model's own chain-of-thought.
    control = build_control_suite(base_suite, format_instruction=fmt)
    base_ctrl = asyncio.run(
        run_suite(control, base_cfg, concurrency=concurrency, cache_dir=cache_dir)
    )
    chal_ctrl = asyncio.run(
        run_suite(control, chal_cfg, concurrency=concurrency, cache_dir=cache_dir)
    )
    # An outage must exit 2, not slip through as an empty intersection.
    _reject_all_errored(baseline=base_ctrl, challenger=chal_ctrl)

    # 2. Per-model perturbation suites, built from that model's own reasoning.
    base_pert_suite, base_answers = build_cot_perturbation_suite(
        base_suite, base_ctrl, scorer, perturbations=cot_list,
        early_template=early_tpl, mistake_template=mistake_tpl,
    )
    chal_pert_suite, chal_answers = build_cot_perturbation_suite(
        base_suite, chal_ctrl, scorer, perturbations=cot_list,
        early_template=early_tpl, mistake_template=mistake_tpl,
    )
    base_pert = asyncio.run(
        run_suite(base_pert_suite, base_cfg, concurrency=concurrency, cache_dir=cache_dir)
    )
    chal_pert = asyncio.run(
        run_suite(chal_pert_suite, chal_cfg, concurrency=concurrency, cache_dir=cache_dir)
    )

    base_fr = compute_cot_faithfulness(base_pert, base_answers)
    chal_fr = compute_cot_faithfulness(chal_pert, chal_answers)

    shared = sorted(set(base_fr.per_case) & set(chal_fr.per_case))
    if not shared:
        console.print(
            "\n[yellow]cot: no cases with usable reasoning answered correctly "
            "by both models in control,[/yellow] so CoT-faithfulness drift is "
            "undefined."
        )
        return False
    drift = compare_runs(
        baseline_scores=[base_fr.per_case[i] for i in shared],
        challenger_scores=[chal_fr.per_case[i] for i in shared],
        baseline_model=base_cfg.model,
        challenger_model=chal_cfg.model,
        suite_name=f"{base_suite.name} (faithfulness/cot)",
        alpha=alpha,
    )
    print_cot_faithfulness_report(base_fr, chal_fr, drift, alpha=alpha)
    out_payload["cot"] = {
        "drift": asdict(drift),
        "baseline": asdict(base_fr),
        "challenger": asdict(chal_fr),
    }
    return drift.significant and drift.delta < 0


@main.command(name="validate-judge")
@click.option("--judge-model", default=None,
              help="Articulation judge to validate. Defaults to "
                   "$RIFT_JUDGE_MODEL or the built-in default.")
@click.option("--cache-dir", default=None, help="Cache directory for completions")
@click.option("--output", "-o", default=None, help="Save the result to JSON")
def validate_judge_cmd(judge_model, cache_dir, output):
    """Validate the faithfulness articulation judge against human gold labels.

    Runs the chosen judge over a committed, hand-labelled gold set and reports
    Cohen's kappa (chance-corrected agreement) plus accuracy and the confusion
    matrix. Publish the kappa alongside any faithfulness number so the metric
    rests on a validated classifier, not faith. Uses the judge's normal cache,
    so a re-run is free.
    """
    from dataclasses import asdict

    from .judge_validation import GOLD_ARTICULATION, validate_judge

    judge = FaithfulnessJudge(judge_model=judge_model, cache_dir=cache_dir)
    judge_cfg = resolve_model(judge.judge_model)
    ensure_provider_keys([judge_cfg.provider], console)

    console.print(
        f"\n[bold]Rift[/bold] validating articulation judge "
        f"[cyan]{judge.judge_model}[/cyan] against "
        f"{len(GOLD_ARTICULATION)} gold cases\n"
    )

    async def _ack(question, cue, reasoning, answer, target):
        return await judge.acknowledged(question, cue, reasoning, answer, target)

    result = asyncio.run(validate_judge(_ack, judge.judge_model))
    asyncio.run(judge.close())
    print_judge_validation_report(result)

    if output:
        import json
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)
        console.print(f"\nResults saved to [green]{output}[/green]")


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
@click.option("--min-info", default=0.0, type=float,
              help="Minimum per-case info contribution "
                   "(|baseline_score − challenger_score|) for acceptance. "
                   "Leave at 0.0 for binary scorers; set 0.2-0.3 for "
                   "fuzzy_match / llm_judge seed suites to filter near-ties.")
@click.option("--min-cases-early-stop", default=20, type=int,
              help="Minimum accepted cases before early-stop on achieved-power "
                   "may fire. Guards against stopping at a tiny sample whose "
                   "power estimate is variance-flattered.")
@click.option("--output", "-o", required=True,
              help="Path to write the discovered suite YAML.")
def discover(baseline, challenger, seed_suite, proposer_model,
             target_power, target_effect, max_cases, batch_size,
             alpha, concurrency, cache_dir, min_info,
             min_cases_early_stop, output):
    """Discover cases that maximize the paired test's power for a model pair.

    Uses a proposer model to generate candidate prompts, runs both
    baseline and challenger on each, and keeps the cases that
    contribute most to McNemar's test on the discovered suite.

    The output is a Rift-compatible suite YAML — feed it straight
    into 'rift compare'.
    """
    import yaml

    seed = load_suite(seed_suite)
    base_cfg = resolve_model(baseline)
    chal_cfg = resolve_model(challenger)
    ensure_provider_keys(
        [base_cfg.provider, chal_cfg.provider, resolve_model(proposer_model).provider],
        console,
    )

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
        min_info=min_info,
        min_cases_before_early_stop=min_cases_early_stop,
    ))

    suite_dict = to_suite_yaml(result)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        yaml.safe_dump(suite_dict, f, sort_keys=False, width=120)

    stop_reason = (
        "early-stopped on achieved-power"
        if result.early_stopped else "reached max_cases / proposer exhausted"
    )
    console.print(
        f"\n[bold]Discovered {result.n_kept} cases[/bold] "
        f"(from {result.n_proposed} proposed, "
        f"{result.n_after_dedup} after dedup, "
        f"{result.n_both_zero} both-zero rejects)."
    )
    console.print(
        f"  Discordant rate (of verified): {result.discordant_rate:.1%}"
    )
    console.print(
        f"  Achieved power:  {result.achieved_power:.2f}   "
        f"(target {result.target_power} at Δ={result.target_effect}, "
        f"{stop_reason})"
    )
    console.print(
        f"  Spend: proposer ${result.proposer_spend_usd:.4f}, "
        f"verification ${result.verification_spend_usd:.4f}, "
        f"[bold]total ${result.proposer_spend_usd + result.verification_spend_usd:.4f}[/bold]"
    )
    from rich.panel import Panel
    console.print()
    console.print(Panel(
        "  [bold]Cases here were selected on divergence between [/bold]"
        f"[cyan]{baseline}[/cyan][bold] and [/bold][cyan]{challenger}[/cyan][bold].[/bold]\n"
        "\n"
        f"  The reported [bold]achieved power = {result.achieved_power:.2f}[/bold]"
        " is the sensitivity of THIS\n"
        "  suite to the specific divergences we found — NOT an unbiased\n"
        "  estimate of how often the two models disagree on arbitrary user\n"
        "  prompts. If you cite this number for procurement or roadmap\n"
        "  decisions, qualify it as 'power conditional on the discovered\n"
        "  adversarial suite'. Re-run on a random-prompt suite (e.g. one of\n"
        "  the stock suites under suites/) for an unbiased population\n"
        "  estimate.",
        title="[bold yellow]⚠ Selection-bias caveat — read before sharing this number[/bold yellow]",
        border_style="yellow",
    ))
    console.print(f"\nSuite saved to [green]{output}[/green]")
    console.print(
        f"Next step: [bold]rift compare --baseline {baseline} "
        f"--challenger {challenger} --suite {output}[/bold]"
    )


@main.command(name="report")
@click.argument("comparison_json", type=click.Path(exists=True, dir_okay=False))
@click.option("--format", "fmt", default="terminal", show_default=True,
              type=click.Choice(["terminal", "markdown", "brief", "brief-md"]),
              help="terminal re-renders the drift report; markdown writes the "
                   "full technical report; brief writes a one-page HTML "
                   "'model upgrade brief' for a non-engineering audience; "
                   "brief-md is the same brief as markdown.")
@click.option("--output", "-o", default=None, type=click.Path(),
              help="Where to write (required for every format but terminal).")
def report(comparison_json, fmt, output):
    """Render a saved comparison (from `rift compare --output`) as a report.

    Keyless and offline: everything is rebuilt from the saved JSON, so a
    comparison can be re-rendered — or turned into an executive brief —
    long after the run, without touching any API.

    \b
    Examples:
      rift report cmp.json                          # re-render in the terminal
      rift report cmp.json --format markdown -o drift_report.md
      rift report cmp.json --format brief -o brief.html
    """
    from .brief import (
        export_brief_html,
        export_brief_markdown,
        load_comparison,
    )

    drift, baseline_result, challenger_result, _extras = (
        load_comparison(comparison_json)
    )

    if fmt == "terminal":
        print_drift_report(drift, baseline_result, challenger_result)
        print_fingerprint_report(baseline_result, challenger_result)
        if drift.subgroups:
            # Label CIs at the level the comparison actually ran at, not a
            # default: a preregistered alpha=0.01 payload carries 99% CIs.
            print_subgroup_table(
                drift.subgroups, title="By subgroup",
                alpha=round(1 - getattr(drift, "ci_level", 0.95), 4),
            )
        return

    if not output:
        raise click.UsageError(f"--format {fmt} needs --output PATH.")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    if fmt == "markdown":
        md = generate_markdown_report(drift, baseline_result, challenger_result)
        with open(output, "w") as f:
            f.write(md)
    elif fmt == "brief":
        export_brief_html(drift, baseline_result, challenger_result, output)
    else:  # brief-md
        export_brief_markdown(drift, baseline_result, challenger_result, output)
    console.print(f"Report saved to [green]{output}[/green]")


@main.command(name="import")
@click.argument("source", type=click.Path(exists=True, dir_okay=False))
@click.option("--from", "source_format", required=True,
              type=click.Choice(["promptfoo", "inspect", "lm-eval", "openai-evals"]),
              help="Format of SOURCE: a promptfoo config YAML, an Inspect AI "
                   "dataset (JSONL/JSON), an lm-eval task YAML, or an OpenAI "
                   "evals samples JSONL.")
@click.option("--output", "-o", required=True, type=click.Path(),
              help="Where to write the Rift suite YAML.")
@click.option("--name", default=None,
              help="Suite name (default: derived from the source filename).")
@click.option("--scoring", default=None,
              type=click.Choice(["exact_match", "fuzzy_match", "semantic",
                                 "llm_judge"]),
              help="Override/choose the scoring method where the source "
                   "doesn't carry one (inspect, openai-evals) or to replace "
                   "the source's assertions (promptfoo, lm-eval).")
@click.option("--dataset", default=None,
              type=click.Path(exists=True, dir_okay=False),
              help="lm-eval only: the documents file (JSONL/JSON) the task "
                   "templates over. Required when the task's dataset_path "
                   "isn't a local file.")
@click.option("--split-by-assert", is_flag=True, default=False,
              help="promptfoo only: when tests mix assertion types, emit one "
                   "suite per scoring method instead of erroring.")
def import_cmd(source, source_format, output, name, scoring, dataset,
               split_by_assert):
    """Import an eval suite from another harness.

    Converts promptfoo / Inspect AI / lm-eval / OpenAI-evals files into
    Rift suite YAML, so existing evals get Rift's paired statistics,
    cost tracking, and drift gate without being re-authored. Conversion
    is conservative: anything that can't be represented faithfully is
    dropped WITH a warning, and every caveat is recorded in the emitted
    suite's description. Keyless — nothing is executed or sent anywhere.

    \b
    Examples:
      rift import --from promptfoo promptfooconfig.yaml -o suites/mine.yaml
      rift import --from inspect samples.jsonl -o suites/mine.yaml --scoring exact_match
      rift import --from lm-eval task.yaml --dataset docs.jsonl -o suites/mine.yaml
      rift import --from openai-evals samples.jsonl -o suites/mine.yaml
    """
    import yaml as _yaml

    from .adapters import convert

    results = convert(
        source_format, source, name=name, scoring=scoring, dataset=dataset,
        split_by_assert=split_by_assert,
    )

    out_base = Path(output)
    written: list[Path] = []
    all_warnings: list[str] = []
    for imported in results:
        out_path = out_base
        if imported.variant:
            out_path = out_base.with_name(
                f"{out_base.stem}_{imported.variant}{out_base.suffix or '.yaml'}"
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            _yaml.safe_dump(imported.suite, f, sort_keys=False, width=100,
                            allow_unicode=True)
        # Round-trip through the real loader so a bad conversion fails here,
        # loudly, not at compare time.
        load_suite(str(out_path))
        written.append(out_path)
        all_warnings.extend(imported.warnings)

    n_cases = sum(len(r.suite["cases"]) for r in results)
    console.print(
        f"\n[bold]Imported {n_cases} cases[/bold] from "
        f"[cyan]{source_format}[/cyan] into "
        f"{', '.join(f'[green]{p}[/green]' for p in written)}"
    )
    if all_warnings:
        # Dedupe repeated per-case warnings for terminal display; the full
        # list is already embedded in the suite description.
        unique = list(dict.fromkeys(all_warnings))
        console.print(
            f"\n[yellow]{len(unique)} import caveat(s)[/yellow] "
            "(also recorded in the suite description):"
        )
        for w in unique[:12]:
            console.print(f"  [yellow]•[/yellow] {w}")
        if len(unique) > 12:
            console.print(f"  [yellow]… and {len(unique) - 12} more[/yellow]")
    first = written[0]
    console.print(
        f"\nNext step: [bold]rift compare --baseline <old> --challenger <new> "
        f"--suite {first}[/bold]"
    )


@main.command()
@click.option("--scenario", default="opus-46-vs-47",
              type=click.Choice(sorted(SCENARIOS)),
              help="Which prepared scenario to walk through.")
@click.option("--auto/--paced", default=True,
              help="Auto-advance between acts (default). --paced waits for "
                   "Enter between acts for a live presenter.")
@click.option("--beat-multiplier", default=1.0, type=float,
              help="Multiplier on per-act pause length (--auto only). "
                   "Set to 0.0 to skip pauses entirely.")
@click.option("--export-html", default=None, type=click.Path(),
              help="Also write a self-contained HTML executive memo.")
@click.option("--export-md", default=None, type=click.Path(),
              help="Also write a markdown executive memo.")
@click.option("--export-svg", default=None, type=click.Path(),
              help="Also write a static SVG screenshot of the terminal demo "
                   "(self-contained, GitHub-embeddable).")
@click.option("--no-clear", is_flag=True, default=False,
              help="Don't clear the screen on start. Use this when piping "
                   "output, recording, or running in CI.")
def demo(scenario, auto, beat_multiplier, export_html, export_md,
         export_svg, no_clear):
    """Run a guided, narrated demo of a real drift finding (offline).

    A four-act walkthrough with no API keys required. Replays the
    committed Opus 4.6 → 4.7 benchmark: accuracy ticks up, but
    cost-per-correct rises ~40% due to a silent tokenizer change.

    Use --export-html to produce a single-file executive memo PMs can
    forward; --export-svg to capture a screenshot for README embedding.
    """
    with console.status("[bold cyan]Preparing demo…", spinner="dots"):
        script, base_run, chal_run, drift = load_scenario(scenario)
    run_demo(script, auto=auto, beat_multiplier=beat_multiplier,
             console=console, no_clear=no_clear)

    if export_html:
        export_demo_html(script, export_html, base_run, chal_run, drift)
        console.print(f"  HTML memo:   [green]{export_html}[/green]")
    if export_md:
        export_demo_markdown(script, export_md)
        console.print(f"  Markdown:    [green]{export_md}[/green]")
    if export_svg:
        export_demo_svg(script, export_svg)
        console.print(f"  SVG capture: [green]{export_svg}[/green]")


@main.command()
@click.option("--panel", "panel_path", default="observatory/panel.yaml",
              show_default=True,
              help="Panel YAML defining endpoints, suites, and the cost cap.")
@click.option("--data-dir", required=True,
              help="Observatory data directory (append-only; typically a "
                   "checkout of the observatory-data branch).")
@click.option("--date", default=None,
              help="Observation date YYYY-MM-DD (default: today UTC; replay "
                   "mode defaults to each run's completed_at date).")
@click.option("--max-cost", default=None, type=float,
              help="Hard USD cap for this pass. Overrides the panel's "
                   "max_cost_usd. Remaining stages are skipped at the cap; "
                   "partial observations are still recorded.")
@click.option("--endpoint", "endpoints", multiple=True,
              help="Observe only these endpoint ids (repeatable). "
                   "Default: every endpoint in the panel.")
@click.option("--from-runs", "from_runs", multiple=True,
              type=click.Path(exists=True),
              help="Replay mode: build observations from saved RunResult "
                   "JSONs instead of live calls (repeatable, keyless). A run "
                   "whose suite ends in '__pushback' pairs as the sycophancy "
                   "follow-up of the same model's base run.")
@click.option("--alpha", default=None, type=float,
              help="Significance threshold for drift events (default: the "
                   "panel's alpha, or 0.05 in replay mode).")
@click.option("--concurrency", default=5, help="Max concurrent API calls")
@click.option("--cache-dir", default=None, help="Cache directory for completions")
def observe(panel_path, data_dir, date, max_cost, endpoints, from_runs,
            alpha, concurrency, cache_dir):
    """Record one observatory observation and update the drift feed.

    Runs the observatory panel (a fixed set of suites + behavioral probes)
    against every configured endpoint, appends the results to the
    append-only data directory, and compares each new observation against
    the previous one with the same paired statistics 'compare' uses —
    pooled through Benjamini–Hochberg across the whole panel. Server
    fingerprint changes are tracked independently of scores, so a silent
    model swap behind a stable alias shows up even when accuracy holds.

    Always exits 0 when the pass completes (drift is a finding, not a
    failure); a non-zero exit means infrastructure failure.
    """
    from .observatory import (
        append_events,
        append_records,
        detect_drift,
        load_panel,
        replay_panel,
        run_panel,
    )

    epoch_baselines: dict[str, str] = {}
    budget = None
    if from_runs:
        records = replay_panel(list(from_runs), date=date)
        alpha = 0.05 if alpha is None else alpha
    else:
        if not Path(panel_path).is_file():
            # The default is a repo-relative path; running from anywhere
            # else hits this immediately — say so instead of tracebacking.
            raise OperationalError(
                f"Panel file not found: {panel_path} — pass --panel "
                "path/to/panel.yaml (see observatory/panel.yaml in the "
                "Rift repo for the format)."
            )
        panel = load_panel(panel_path)
        alpha = panel.alpha if alpha is None else alpha
        epoch_baselines = {
            ep.id: ep.epoch_baseline for ep in panel.endpoints
            if ep.epoch_baseline
        }
        selected = [ep for ep in panel.endpoints
                    if not endpoints or ep.id in endpoints]
        if not selected:
            raise click.UsageError(
                f"--endpoint matched nothing; panel defines "
                f"{[ep.id for ep in panel.endpoints]}."
            )
        ensure_provider_keys(
            [resolve_model(ep.model).provider for ep in selected], console
        )
        console.print(
            f"\n[bold]Rift[/bold] observatory pass: "
            f"{len(selected)} endpoint(s) × {len(panel.suites)} suite(s)"
        )
        records, budget = asyncio.run(run_panel(
            panel, data_dir, date=date, cache_dir=cache_dir,
            concurrency=concurrency, max_cost_usd=max_cost,
            endpoints=[ep.id for ep in selected],
        ))

    if not records:
        console.print("[yellow]No observations were produced "
                      "(budget cap hit before the first stage, or no input "
                      "runs).[/yellow]")
        return

    entries = append_records(records, data_dir)
    events = []
    for obs_date in sorted({r.date for r in records}):
        date_events = detect_drift(data_dir, obs_date, alpha=alpha,
                                   epoch_baselines=epoch_baselines)
        append_events(date_events, data_dir)
        events.extend(date_events)
    print_observation_report(entries, events, budget=budget)
    console.print(f"\nObservations appended to [green]{data_dir}[/green]")


@main.command()
@click.option("--model", "models", multiple=True,
              help="Model to price (alias or id; repeatable). Every model is "
                   "estimated against every --suite, like `rift matrix`.")
@click.option("--suite", "suites", multiple=True,
              help="Suite name or path (repeatable).")
@click.option("--panel", "panel_path", default=None,
              help="Estimate one full observatory pass from a panel YAML "
                   "instead of --model/--suite (includes the sycophancy "
                   "pushback stage).")
@click.option("--trials", default=1, show_default=True,
              help="Replicates per case (matches compare/run --trials).")
@click.option("--output-tokens-per-case", default=None, type=int,
              help="Output-token allowance per call. Default: the budget "
                   "guard's 300. Thinking models on reasoning-heavy suites "
                   "run 2-3x that.")
@click.option("--calibrate-from", "calibrate", multiple=True,
              type=click.Path(exists=True, dir_okay=False),
              help="Saved run/comparison JSON whose measured token counts "
                   "replace the heuristic for its suite (repeatable).")
def estimate(models, suites, panel_path, trials, output_tokens_per_case,
             calibrate):
    """Estimate what a run would cost — keyless, before spending anything.

    Prices every model × suite cell at standard-mode list price using the
    same heuristic the observatory budget guard applies (prompt chars/4 in,
    a flat per-case allowance out), so the number here is the one the
    guard would check against max_cost_usd. It is an order-of-magnitude
    figure: Fable 5/5.1 and Opus 5 bill always-on thinking as output and
    can run 2-3x over on hard suites; Batch is -50%; fast mode is a premium.

    \b
    Examples:
      rift estimate --model fable-5-1 --model opus-5 --suite reasoning --suite hard_reasoning
      rift estimate --panel observatory/panel.yaml
      rift estimate --model fable-5-1 --suite hard_reasoning --calibrate-from benchmarks/fable5_vs_opus47/hard_reasoning.json
    """
    from rich.table import Table

    from .estimate import (
        calibration_from_run,
        estimate_grid,
        estimate_panel,
    )
    from .observatory import EST_OUTPUT_TOKENS_PER_CASE, load_panel

    if panel_path and (models or suites):
        raise click.UsageError("--panel replaces --model/--suite; pass one or the other.")
    if not panel_path and not (models and suites):
        raise click.UsageError("Pass --model and --suite (repeatable), or --panel.")

    per_case = output_tokens_per_case or EST_OUTPUT_TOKENS_PER_CASE
    calibrations: dict[str, dict[str, dict]] = {}
    for path in calibrate:
        name, rows = calibration_from_run(path)
        calibrations.setdefault(name, {}).update(rows)

    if panel_path:
        panel = load_panel(panel_path)
        est = estimate_panel(panel, output_per_case=per_case,
                             calibrations=calibrations)
        title = f"Estimated cost of one observatory pass ({panel_path})"
        cap = panel.max_cost_usd
    else:
        est = estimate_grid(list(models), list(suites), trials=trials,
                            output_per_case=per_case, calibrations=calibrations)
        title = "Estimated cost (standard-mode list price)"
        cap = None

    tbl = Table(title=title)
    for col, just in (("Model", "left"), ("Suite", "left"), ("Calls", "right"),
                      ("In tok", "right"), ("Out tok", "right"),
                      ("Est. USD", "right"), ("Note", "left")):
        tbl.add_column(col, justify=just, overflow="fold")  # type: ignore[arg-type]
    for st in est.stages:
        tbl.add_row(st.model, st.suite, str(st.calls), f"{st.input_tokens:,}",
                    f"{st.output_tokens:,}", f"${st.cost_usd:.3f}",
                    f"[yellow]{st.note}[/yellow]" if st.note else "")
    tbl.add_row("[bold]Total[/bold]", "", "", "", "",
                f"[bold]${est.total_usd:.2f}[/bold]", "")
    console.print(tbl)

    if cap is not None:
        verdict = ("[green]within[/green]" if est.total_usd <= cap
                   else "[bold red]OVER[/bold red]")
        console.print(f"  Panel cap max_cost_usd = ${cap:.2f} → estimate is "
                      f"{verdict} the cap (guard skips stages past it).")
    if est.unpriced_models:
        console.print(
            f"  [yellow]Unpriced model(s) {est.unpriced_models} estimated at the "
            "catalog maximum — add them to rift/pricing.py for a real "
            "number.[/yellow]"
        )
    console.print(
        f"  [dim]Heuristic: chars/4 input, {per_case} output tokens/call"
        + (f", calibrated for {sorted(calibrations)}" if calibrations else "")
        + ". Thinking models (Fable, Opus 5) can run 2-3x over on hard suites;"
        " Batch API is -50%.[/dim]"
    )


@main.command(name="observatory-site")
@click.option("--data-dir", required=True,
              type=click.Path(exists=True, file_okay=False),
              help="Observatory data directory to render.")
@click.option("--out", "out_dir", default="_site", show_default=True,
              help="Output directory for the static site.")
def observatory_site_cmd(data_dir, out_dir):
    """Render the observatory data directory into a static dashboard.

    Produces a self-contained site (no JavaScript, no external assets):
    a front page with the drift feed and per-endpoint summaries, one
    timeline page per endpoint with fingerprint-change markers, and the
    raw JSONL data for machine consumption. Deploy anywhere static —
    GitHub Pages, an artifact viewer, or file://.
    """
    from .observatory_site import render_site

    written = render_site(data_dir, out_dir)
    n = len(written)
    console.print(
        f"Observatory site rendered: [bold]{n}[/bold] "
        f"file{'s' if n != 1 else ''} under [green]{out_dir}[/green]"
    )
    console.print(f"Open [cyan]{Path(out_dir) / 'index.html'}[/cyan]")


@main.group()
def lm():
    """Train and poke RiftLM — Rift's own tiny built-in LLM.

    A character-level GPT implemented in pure numpy and trained from
    scratch on synthetic string tasks in a few minutes on a laptop CPU.
    One `rift lm train` produces a baseline/challenger checkpoint pair
    (the training data mix shifts partway through, dropping the `rev`
    task), so you can run the whole drift pipeline against a model you
    built yourself — no API key, no network:

    \b
      rift lm train
      rift compare --baseline riftlm:models/riftlm-a.npz \\
                   --challenger riftlm:models/riftlm-b.npz \\
                   --suite riftlm --subgroup task:
    """


@lm.command(name="train")
@click.option("--out", "out_dir", default="models", show_default=True,
              help="Directory for the riftlm-a/riftlm-b checkpoints.")
@click.option("--steps", default=3000, show_default=True,
              help="Total optimizer steps.")
@click.option("--switch", default=0.6, show_default=True,
              type=click.FloatRange(0, 1, min_open=True, max_open=True),
              help="Fraction of steps after which the task mix shifts "
                   "(rev dropped) and checkpoint A is saved.")
@click.option("--batch-size", default=64, show_default=True)
@click.option("--lr", default=1e-3, show_default=True, type=float,
              help="Peak Adam learning rate (5% warmup, cosine decay).")
@click.option("--seed", default=0, show_default=True,
              help="Seeds both the weight init and the data stream.")
def lm_train(out_dir, steps, switch, batch_size, lr, seed):
    """Train RiftLM from scratch (pure numpy, CPU, a few minutes)."""
    if not 0.0 < switch < 1.0:
        raise click.UsageError("--switch must be between 0 and 1 (exclusive).")
    from .lm.train import train_riftlm

    with console.status("[bold]training RiftLM...", spinner="dots"):
        result = train_riftlm(
            out_dir=out_dir, steps=steps, switch=switch,
            batch_size=batch_size, lr=lr, seed=seed,
            log=lambda msg: console.print(f"[dim]{msg}[/dim]"),
        )
    console.print(
        f"\n[green]Done.[/green] Baseline [cyan]{result.checkpoint_a}[/cyan] "
        f"(all 4 tasks) vs challenger [cyan]{result.checkpoint_b}[/cyan] "
        f"(continued training, rev dropped)."
    )
    console.print(
        "\nNow catch the regression you just manufactured:\n"
        f"  [bold]rift compare --baseline riftlm:{result.checkpoint_a} "
        f"--challenger riftlm:{result.checkpoint_b} "
        f"--suite riftlm --subgroup task:[/bold]"
    )


@lm.command(name="sample")
@click.option("--checkpoint", "-c", required=True,
              help="Path to a RiftLM .npz checkpoint.")
@click.option("--prompt", "-p", required=True,
              help="Prompt text, e.g. 'rev abcde = '.")
@click.option("--max-new", default=32, show_default=True,
              help="Max characters to generate (stops at newline).")
def lm_sample(checkpoint, prompt, max_new):
    """Greedy-decode one prompt against a checkpoint.

    Goes through RiftLMProvider — the same inference path 'rift
    compare' scores — so what you see here is exactly what a run
    would grade (and a missing/corrupt checkpoint gets the same clean
    one-line error).
    """
    from .providers.riftlm import RiftLMProvider

    provider = RiftLMProvider(model=f"riftlm:{checkpoint}")
    completion = asyncio.run(provider.complete(prompt, max_tokens=max_new))
    console.print(f"[dim]{prompt}[/dim][bold]{completion.output_text}[/bold]")


@lm.command(name="suite")
@click.option("--out", "out_path", default="suites/riftlm.yaml",
              show_default=True, help="Where to write the suite YAML.")
@click.option("--per-task", default=30, show_default=True,
              help="Held-out cases per task.")
@click.option("--seed", default=1234, show_default=True,
              help="RNG seed for drawing eval cases (held-out split).")
def lm_suite(out_path, per_task, seed):
    """Regenerate the held-out RiftLM eval suite (suites/riftlm.yaml).

    Cases are drawn exclusively from the evaluation split — lines the
    training sampler refuses by content hash — so the committed suite is
    held out by construction. Deterministic for a given seed; the
    committed default is seed 1234.
    """
    import yaml as _yaml

    from .lm.data import gen_eval_cases

    cases = gen_eval_cases(per_task, seed=seed)
    doc = {
        "name": "riftlm",
        "description": (
            "Held-out synthetic string tasks (cpy/rev/srt/max) for RiftLM, "
            f"generated by `rift lm suite --per-task {per_task} --seed {seed}`. "
            "Do not hand-edit; regenerate instead."
        ),
        "scoring": "exact_match",
        "cases": cases,
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        _yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)
    console.print(
        f"Wrote [green]{out}[/green] ({len(cases)} cases, "
        f"{per_task} per task)."
    )
    console.print(
        f"Use it by path: [bold]--suite {out}[/bold]  (the bare name "
        f"'riftlm' resolves to the copy bundled with the install, which "
        f"won't reflect a regenerated file)."
    )


if __name__ == "__main__":
    main()
