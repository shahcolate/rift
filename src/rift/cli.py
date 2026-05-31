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
from .comparator import compare_runs, compare_by_subgroup, power_analysis
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
    print_matrix,
    print_power_report,
    print_refusal_report,
    print_subgroup_table,
    print_sycophancy_report,
)
from .runner import RunResult, run_suite
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
@click.option("--strip-io", is_flag=True, default=False,
              help="When writing --output, omit per-case input_text and "
                   "output fields. Use for proprietary suites whose prompts "
                   "or completions should not leave your machine.")
def compare(baseline, challenger, suite, concurrency, alpha, output, report,
            cache_dir, context_rot, enterprise_multiplier, subgroup,
            refusal, calibration, power, judge_model, strip_io):
    """Compare two models on an eval suite."""
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
                  cache_dir=cache_dir, enterprise_multiplier=enterprise_multiplier)
    )
    challenger_result = asyncio.run(
        run_suite(suite_config, challenger_config, concurrency=concurrency,
                  cache_dir=cache_dir, enterprise_multiplier=enterprise_multiplier)
    )

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
@click.option("--strip-io", is_flag=True, default=False,
              help="Omit per-case input_text and output from the saved "
                   "JSON. Use for proprietary suites.")
def run(model, suite, concurrency, output, cache_dir, context_rot,
        enterprise_multiplier, judge_model, strip_io):
    """Run a single model against an eval suite and save results."""
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
                  enterprise_multiplier=enterprise_multiplier)
    )

    result.save(output, strip_io=strip_io)
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
@click.option("--strip-io", is_flag=True, default=False,
              help="When writing per-model JSONs, omit input_text and "
                   "output fields. Use for proprietary suites.")
def matrix(models, suite, concurrency, cache_dir, context_rot,
           enterprise_multiplier, output_dir, strip_io):
    """Run every model in ``--models`` and print an NxN drift matrix.

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
@click.option("--alpha", default=0.05, help="Significance threshold")
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
    hint_targets = parse_hint_targets(wrong_run)

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
    into ``rift compare``.
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
        "  the stock suites under ``suites/``) for an unbiased population\n"
        "  estimate.",
        title="[bold yellow]⚠ Selection-bias caveat — read before sharing this number[/bold yellow]",
        border_style="yellow",
    ))
    console.print(f"\nSuite saved to [green]{output}[/green]")
    console.print(
        f"Next step: [bold]rift compare --baseline {baseline} "
        f"--challenger {challenger} --suite {output}[/bold]"
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


if __name__ == "__main__":
    main()
