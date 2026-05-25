"""Reproducible context-rot benchmark driver.

Runs the `context_rot_reasoning` suite against one or more frontier
models at four distractor regimes (0k, 2k, 8k, 32k tokens) and
emits a drift-by-regime report.

Two modes:

* **live** — hits the real provider APIs. Requires ANTHROPIC_API_KEY
  and/or OPENAI_API_KEY. Results are cached in `.rift/cache/` so a
  re-run is free. Use this when you want the actual numbers.

* **record** — reads a pre-recorded outcomes YAML at
  ``benchmarks/context_rot_outcomes.yaml`` (see format in that file)
  and injects them into Rift's completion cache keyed so the normal
  ``run_suite`` machinery serves from cache 100%. Use this to
  reproduce a published report exactly, or to demonstrate the
  pipeline without API access.

The choice matters for credibility: always disclose which mode
produced a given report, and keep recorded outcomes under version
control so reviewers can audit them.

Usage:

    python benchmarks/run_context_rot.py --mode record \\
        --models opus-4-7,sonnet-4-6,gpt-4o \\
        --output benchmarks/context_rot_opus47.md
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import yaml

from rift.comparator import compare_by_subgroup, compare_runs
from rift.config import load_suite, resolve_model
from rift.context_rot import expand_suite
from rift.demo import prime_cache_from_recording
from rift.reporter import generate_markdown_report
from rift.runner import run_suite


ROOT = Path(__file__).parent.parent
RECORDED = Path(__file__).parent / "context_rot_outcomes.yaml"


async def _run_all(models: list[str], cache_dir: Path, mode: str,
                   enterprise_multiplier: float, concurrency: int):
    base = load_suite("context_rot_reasoning")
    suite = expand_suite(base)

    outcomes: dict = {}
    if mode == "record":
        if not RECORDED.exists():
            sys.exit(f"Recorded outcomes not found at {RECORDED}")
        outcomes = yaml.safe_load(RECORDED.read_text()) or {}

    runs = {}
    for m in models:
        cfg = resolve_model(m)
        if mode == "record":
            prime_cache_from_recording(suite, cfg.model, outcomes, cache_dir)
        runs[m] = await run_suite(
            suite, cfg, concurrency=concurrency, cache_dir=str(cache_dir),
            enterprise_multiplier=enterprise_multiplier,
        )
    return suite, runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("live", "record"), default="record")
    ap.add_argument("--models", default="opus-4-7,opus-4-6,sonnet-4-6,gpt-4o")
    ap.add_argument("--baseline", default="opus-4-6",
                    help="Model to treat as baseline in pairwise drift reports.")
    ap.add_argument("--enterprise-multiplier", type=float, default=1.0)
    ap.add_argument("--concurrency", type=int, default=2,
                    help="Per-model parallel requests. Lower if you hit 429s.")
    ap.add_argument("--cache-dir", default=str(ROOT / ".rift" / "cache"))
    ap.add_argument("--output", default=str(ROOT / "benchmarks" / "context_rot_opus47.md"))
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    # Keep recorded (synthetic) outcomes out of the live cache so a
    # subsequent live run can never be silently served prior synthesis.
    cache_dir = Path(args.cache_dir)
    if args.mode == "record":
        cache_dir = cache_dir.parent / (cache_dir.name + "_recorded")
    suite, runs = asyncio.run(
        _run_all(models, cache_dir, args.mode,
                 args.enterprise_multiplier, args.concurrency)
    )

    baseline = args.baseline
    if baseline not in runs:
        sys.exit(f"baseline {baseline!r} not in --models list")

    sections: list[str] = []
    sections.append(f"# Context-Rot Benchmark — Opus 4.7 and Friends\n")
    sections.append(
        f"_Suite: `{suite.name}` · {len(suite.cases)} cases "
        f"({len(suite.cases)//4} base × 4 distractor regimes). "
        f"Mode: `{args.mode}`._\n"
    )
    if args.mode == "record":
        sections.append(
            "> ⚠️ **Synthetic replay — not a live API run.** This report is "
            "regenerated from `benchmarks/context_rot_outcomes.yaml`, a "
            "deterministic synthesis whose token-inflation parameter is "
            "calibrated to reproduce the headline numbers of a 2026-04-21 "
            "live capture. For the authoritative live numbers, see "
            "[`opus47_live.md`](opus47_live.md). For methodology, see "
            "[`generate_synthetic_outcomes.py`](generate_synthetic_outcomes.py).\n"
        )

    # Per-model headline table.
    error_models = [m for m in models if runs[m].metadata.get("n_errors", 0) > 0]
    if error_models:
        sections.append(
            "> ⚠️ **Run integrity warning.** One or more models had API "
            "errors during this run; their scores are biased downward "
            "because errored cases are counted as 0. Do not interpret "
            "drift on the affected models without re-running.\n"
        )

    sections.append("## Headline: score and cost by model\n")
    sections.append(
        "| Model | Mean | Correct | Errors | Spend | $/correct |\n"
        "|-------|------|---------|--------|-------|-----------|"
    )
    for m in models:
        r = runs[m]
        n_correct = sum(1 for c in r.cases if c.score >= 0.999)
        n_err = r.metadata.get("n_errors", 0)
        cpc = r.cost_per_correct()
        flag = " ⚠️" if n_err else ""
        sections.append(
            f"| `{m}` | {r.mean_score:.3f} | {n_correct}/{len(r.cases)} | "
            f"{n_err}{flag} | ${r.total_cost_usd:.4f} | "
            f"{'∞' if cpc == float('inf') else f'${cpc:.4f}'} |"
        )
    sections.append("")

    sections.append("## Tokens: input vs output by model\n")
    sections.append(
        "| Model | Input tokens | Output tokens | Input/case | Output/case | In:Out |\n"
        "|-------|--------------|---------------|------------|-------------|--------|"
    )
    for m in models:
        r = runs[m]
        n = len(r.cases) or 1
        ti, to = r.total_input_tokens, r.total_output_tokens
        ratio = f"{ti/to:.1f}:1" if to else "—"
        sections.append(
            f"| `{m}` | {ti:,} | {to:,} | {ti/n:,.0f} | {to/n:,.0f} | {ratio} |"
        )
    sections.append("")

    sections.append(f"## Token inflation vs `{baseline}`\n")
    sections.append(
        "Does output inflate along with input, or is the inflation "
        "input-only? A uniform ratio on both axes points at a "
        "tokenizer-vocabulary change; divergent ratios point at a "
        "response-length shift.\n"
    )
    sections.append(
        "| Challenger | Input ratio | Output ratio | Total ratio |\n"
        "|------------|-------------|--------------|-------------|"
    )
    base_r = runs[baseline]
    bi, bo = base_r.total_input_tokens, base_r.total_output_tokens
    for m in models:
        if m == baseline:
            continue
        r = runs[m]
        in_ratio = r.total_input_tokens / bi if bi else float("nan")
        out_ratio = r.total_output_tokens / bo if bo else float("nan")
        tot_ratio = ((r.total_input_tokens + r.total_output_tokens)
                     / (bi + bo)) if (bi + bo) else float("nan")
        sections.append(
            f"| `{m}` | {in_ratio:.3f}× | {out_ratio:.3f}× | {tot_ratio:.3f}× |"
        )
    sections.append("")

    # Pairwise drift vs baseline, subgroup-split by distractor level.
    base_run = runs[baseline]
    for m in models:
        if m == baseline:
            continue
        chal_run = runs[m]
        drift = compare_runs(
            baseline_scores=base_run.scores,
            challenger_scores=chal_run.scores,
            baseline_model=baseline,
            challenger_model=m,
            suite_name=suite.name,
            baseline_costs=[c.cost_usd for c in base_run.cases],
            challenger_costs=[c.cost_usd for c in chal_run.cases],
        )
        drift.subgroups = compare_by_subgroup(
            baseline_scores=base_run.scores,
            challenger_scores=chal_run.scores,
            tags_per_case=[c.tags for c in base_run.cases],
            subgroup_prefix="distractor:",
            baseline_model=baseline,
            challenger_model=m,
            suite_name=suite.name,
            baseline_costs=[c.cost_usd for c in base_run.cases],
            challenger_costs=[c.cost_usd for c in chal_run.cases],
        )
        sections.append(f"## {baseline} vs {m}\n")
        sections.append(generate_markdown_report(drift, base_run, chal_run))
        sections.append("")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(sections))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
