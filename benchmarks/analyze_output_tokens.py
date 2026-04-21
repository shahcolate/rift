"""Output-token decomposition driver.

Reads two saved :class:`rift.runner.RunResult` JSONs (baseline and
challenger on the same suite), re-tokenizes each model's outputs
through *both* models' tokenizers via Anthropic's ``count_tokens``
endpoint, and prints the tokenizer-vs-verbosity split plus a cost
attribution.

Why this script exists. Our context-rot report showed Opus 4.7
consuming ~1.7× the output tokens of Opus 4.6 on identical prompts.
That number is ambiguous: is 4.7 writing more, or does the 4.7
tokenizer chop the same text into more pieces? The two have very
different implications — verbosity is a prompt-engineering fix,
tokenizer inflation is a pricing-tier conversation — so we need to
split them rather than pick one story.

The pure math lives in :mod:`rift.output_tokens`. This file is only
orchestration: read JSONs, call the API, cache responses, tabulate.

Modes
-----

* ``--mode live``   — hits the real ``count_tokens`` endpoint. Cheap
  (``count_tokens`` is free) but requires ``ANTHROPIC_API_KEY``.
  Responses are cached under ``.rift/tokenizer_cache/`` keyed by
  ``(tokenizer_model, sha256(text)[:16])`` so re-runs don't re-hit
  the API.
* ``--mode cached`` — read-only; fails if a required cache entry is
  missing. Use this to reproduce a published analysis exactly.

Usage
-----

    python benchmarks/analyze_output_tokens.py \\
        --baseline  runs/opus46_context_rot.json \\
        --challenger runs/opus47_context_rot.json \\
        --output benchmarks/output_token_decomposition.md
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

import httpx

from rift.output_tokens import OutputRow, cost_attribution, decompose
from rift.runner import RunResult


ROOT = Path(__file__).parent.parent
DEFAULT_CACHE = ROOT / ".rift" / "tokenizer_cache"


def _cache_key(tokenizer_model: str, text: str) -> str:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{tokenizer_model}_{h}.json"


async def _count_tokens(
    client: httpx.AsyncClient,
    tokenizer_model: str,
    text: str,
    cache_dir: Path,
    mode: str,
) -> int:
    """Return the token count of ``text`` under ``tokenizer_model``.

    ``count_tokens`` is a free endpoint on the Messages API; we still
    cache because the *paired* analysis crosses two models' tokenizers
    over every case, and re-runs during iteration would otherwise
    burn request quota for no signal.
    """
    cached = cache_dir / _cache_key(tokenizer_model, text)
    if cached.exists():
        return int(json.loads(cached.read_text())["input_tokens"])
    if mode == "cached":
        raise RuntimeError(
            f"cache miss for {tokenizer_model} on text[:40]={text[:40]!r} "
            f"and --mode cached; re-run with --mode live."
        )
    # count_tokens takes the same Messages-API shape as /v1/messages.
    resp = await client.post(
        "/v1/messages/count_tokens",
        json={
            "model": tokenizer_model,
            "messages": [{"role": "user", "content": text}],
        },
    )
    resp.raise_for_status()
    payload = resp.json()
    n = int(payload.get("input_tokens", 0))
    tmp = cached.with_suffix(".tmp")
    tmp.write_text(json.dumps({"input_tokens": n}))
    tmp.replace(cached)
    return n


async def _build_rows(
    baseline: RunResult,
    challenger: RunResult,
    baseline_tokenizer: str,
    challenger_tokenizer: str,
    cache_dir: Path,
    mode: str,
    concurrency: int,
) -> list[OutputRow]:
    """Re-tokenize each case's outputs under both models' tokenizers."""
    assert len(baseline.cases) == len(challenger.cases), (
        "paired analysis requires baseline and challenger to have the "
        "same case count from the same suite"
    )

    if mode == "live":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            sys.exit("ANTHROPIC_API_KEY required for --mode live")
        client = httpx.AsyncClient(
            base_url="https://api.anthropic.com",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=60.0,
        )
    else:
        # Offline mode: client is never called, but we still need an
        # object so the helper can be awaited uniformly.
        client = httpx.AsyncClient(base_url="http://unused")

    sem = asyncio.Semaphore(concurrency)

    async def retok(tok_model: str, text: str) -> int:
        if not text:
            return 0
        async with sem:
            return await _count_tokens(client, tok_model, text, cache_dir, mode)

    try:
        rows: list[OutputRow] = []
        for i, (bc, cc) in enumerate(zip(baseline.cases, challenger.cases)):
            b_out = bc.output or ""
            c_out = cc.output or ""
            # Four re-tokenizations per case: each output under each
            # tokenizer. The paired structure is what lets us factor
            # out "tokenizer alone" vs. "text length alone" downstream.
            counts = await asyncio.gather(
                retok(baseline_tokenizer, b_out),
                retok(challenger_tokenizer, b_out),
                retok(baseline_tokenizer, c_out),
                retok(challenger_tokenizer, c_out),
            )
            rows.append(OutputRow(
                case_index=i,
                baseline_chars=len(b_out),
                challenger_chars=len(c_out),
                baseline_actual_tokens=int(bc.output_tokens),
                challenger_actual_tokens=int(cc.output_tokens),
                baseline_output_under_baseline_tokenizer=counts[0],
                baseline_output_under_challenger_tokenizer=counts[1],
                challenger_output_under_baseline_tokenizer=counts[2],
                challenger_output_under_challenger_tokenizer=counts[3],
            ))
        return rows
    finally:
        await client.aclose()


def _render_markdown(
    baseline: RunResult,
    challenger: RunResult,
    rows: list[OutputRow],
    baseline_tokenizer: str,
    challenger_tokenizer: str,
    enterprise_multiplier: float,
) -> str:
    """Produce a self-contained markdown write-up of the decomposition."""
    decomp = decompose(rows)
    attr = cost_attribution(
        rows,
        baseline_model=baseline.model,
        challenger_model=challenger.model,
        enterprise_multiplier=enterprise_multiplier,
    )

    # Under a clean multiplicative model observed ≈ tokenizer × verbosity.
    # The residual flags cases where the assumption breaks (pathological
    # tokenizer behavior on specific input families).
    lines = [
        f"# Output-Token Decomposition: `{baseline.model}` → `{challenger.model}`",
        "",
        f"_Suite: `{baseline.suite_name}` · {decomp.n} usable paired cases._",
        "",
        "## What this is",
        "",
        "Splits the observed output-token ratio into two parts:",
        "",
        "- **Tokenizer effect** — same string, different tokenizer. "
        "Pure pricing artifact; the model isn't doing anything different.",
        "- **Verbosity effect** — challenger writes longer outputs. "
        "A real behavioral change measurable in characters too.",
        "",
        "## Decomposition",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Observed output-token ratio (challenger / baseline) | **{decomp.observed_ratio:.3f}×** |",
        f"| Tokenizer ratio on baseline's outputs | {decomp.tokenizer_ratio_on_baseline:.3f}× |",
        f"| Tokenizer ratio on challenger's outputs | {decomp.tokenizer_ratio_on_challenger:.3f}× |",
        f"| **Tokenizer ratio (averaged)** | **{decomp.tokenizer_ratio:.3f}×** |",
        f"| **Verbosity ratio (characters)** | **{decomp.verbosity_ratio:.3f}×** |",
        f"| Multiplicative residual (observed − tokenizer × verbosity) | {decomp.multiplicative_residual:+.4f} |",
        "",
        f"Tokenizers compared: `{baseline_tokenizer}` vs `{challenger_tokenizer}`.",
        "",
        "## Cost attribution",
        "",
        "Walks the counterfactual **baseline → tokenizer-swap → "
        "add-verbosity → price-change → challenger**, so the three "
        "components sum exactly to the observed output-cost delta.",
        "",
        "| Component | USD |",
        "|-----------|-----|",
        f"| Baseline output cost | ${attr.baseline_output_cost_usd:,.4f} |",
        f"| Challenger output cost | ${attr.challenger_output_cost_usd:,.4f} |",
        f"| **Observed delta** | **${attr.delta_usd:+,.4f}** |",
        f"| → from tokenizer change | ${attr.tokenizer_component_usd:+,.4f} |",
        f"| → from verbosity change | ${attr.verbosity_component_usd:+,.4f} |",
        f"| → from $/Mtok price change | ${attr.price_change_component_usd:+,.4f} |",
        "",
    ]
    if enterprise_multiplier != 1.0:
        lines.append(f"_Enterprise multiplier applied: ×{enterprise_multiplier}._")
        lines.append("")

    # How to read the headline, in one sentence, before the reader has
    # to decide if they care about the math.
    if decomp.tokenizer_ratio > 1.02 and decomp.verbosity_ratio > 1.02:
        verdict = (
            "Both effects are real: the challenger's tokenizer packs "
            "fewer characters per token *and* the model writes more "
            "text. Cost impact is the product."
        )
    elif decomp.tokenizer_ratio > 1.05:
        verdict = (
            "The token increase is mostly a tokenizer artifact: the "
            "model isn't writing meaningfully more, its tokenizer is "
            "just splitting the same text into more pieces."
        )
    elif decomp.verbosity_ratio > 1.05:
        verdict = (
            "The token increase is real verbosity: the model is "
            "actually writing longer outputs (visible in character "
            "counts, not just token counts)."
        )
    else:
        verdict = "Neither effect is material; outputs are effectively equivalent."
    lines += ["## Headline", "", verdict, ""]

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--baseline", required=True, help="Path to baseline RunResult JSON.")
    ap.add_argument("--challenger", required=True, help="Path to challenger RunResult JSON.")
    ap.add_argument(
        "--baseline-tokenizer", default=None,
        help="Model ID to use for the baseline tokenizer (defaults to "
             "the baseline run's model).",
    )
    ap.add_argument(
        "--challenger-tokenizer", default=None,
        help="Model ID to use for the challenger tokenizer (defaults "
             "to the challenger run's model).",
    )
    ap.add_argument("--mode", choices=("live", "cached"), default="live")
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--enterprise-multiplier", type=float, default=1.0)
    ap.add_argument("--output", default=None,
                    help="Write markdown report here. Prints to stdout if omitted.")
    args = ap.parse_args()

    baseline = RunResult.load(args.baseline)
    challenger = RunResult.load(args.challenger)

    b_tok = args.baseline_tokenizer or baseline.model
    c_tok = args.challenger_tokenizer or challenger.model

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows = asyncio.run(_build_rows(
        baseline=baseline,
        challenger=challenger,
        baseline_tokenizer=b_tok,
        challenger_tokenizer=c_tok,
        cache_dir=cache_dir,
        mode=args.mode,
        concurrency=args.concurrency,
    ))

    md = _render_markdown(
        baseline, challenger, rows, b_tok, c_tok, args.enterprise_multiplier
    )
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(md)
        print(f"Wrote {args.output}")
    else:
        print(md)


if __name__ == "__main__":
    main()
