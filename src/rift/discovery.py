"""Power-stratified auto-adversarial case discovery.

For a given baseline / challenger pair, propose candidate prompts
and keep the ones that contribute most to the *paired statistical
test's power* — not just the ones where the two models happen to
disagree. The output is a Rift-compatible ``SuiteConfig`` ready to
feed into ``rift compare``.

Why power, not just disagreement
--------------------------------
McNemar's exact test depends only on the discordant pairs — cases
where exactly one of ``(baseline, challenger)`` is correct.
Concordant pairs contribute zero information. Discovering useful
cases is therefore equivalent to discovering discordant pairs, and
the suite's statistical power follows directly from the discordant
count and skew. Ranking candidates by their predicted contribution
to the paired test (rather than by raw divergence) is what makes
this a methodological contribution rather than a "find prompts
where they differ" loop.

Selection bias caveat
---------------------
Cases here are *selected on divergence*. The ``achieved_power``
metric reported by a discovery run therefore measures the
sensitivity of the discovered suite — **not** an unbiased estimate
of how often the two models will disagree on arbitrary user
prompts. The CLI report, the YAML metadata, and this docstring all
state this explicitly so a reader cannot mistake the number for a
population estimate.

v0 scope (intentional)
----------------------
This module ships the minimum-credible version:

* Single-batch proposer (no iterative refinement loop)
* Binary scorers only (continuous Fisher-info ranking is v1)
* Jaccard 5-gram dedup (embedding dedup is v1)
* Deterministic validity gate (LLM critic is v1)
* No adaptive early-stopping on achieved-power

The descopes are deliberate so the first cut ships as one PR and
the iterative version follows once this proves out.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Callable

from .comparator import power_analysis
from .config import EvalCase, ModelConfig, SuiteConfig, resolve_model
from .pricing import cost_of
from .providers import BaseProvider
from .providers.anthropic import AnthropicProvider
from .providers.openai import OpenAIProvider
from .runner import run_suite


# Knobs surfaced as keyword arguments to :func:`discover` rather than
# left at the call site — they are operational, not methodological.
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_CASES = 50
DEFAULT_TARGET_POWER = 0.9
DEFAULT_TARGET_EFFECT = 0.05
DEFAULT_ALPHA = 0.05

# Maximum proposer-output characters we'll attempt to parse. Guards
# against a runaway proposer that fills its context.
_MAX_PROPOSER_OUTPUT = 200_000

# Bail out after this many consecutive batches yielded zero accepted
# cases. Protects against a proposer that keeps emitting the same
# candidates (every one of which gets dedup'd) — without this guard,
# discover() would loop forever consuming proposer spend.
_MAX_CONSECUTIVE_STALE_BATCHES = 3


# ---------------------------------------------------------------------------
# Proposer prompt
# ---------------------------------------------------------------------------

PROPOSER_PROMPT_TEMPLATE = """\
You are helping construct an eval suite that probes for behavioral \
differences between two language model versions.

Seed suite name: {suite_name}
Seed suite description: {suite_description}
Scoring method: {scoring}

Sample cases from the seed suite (for format / domain guidance):
{seed_examples}

Your task: propose {batch_size} new cases in the same domain and \
format. Aim for cases that are:
- answerable (a competent reasoner can determine the correct answer)
- at the harder end of the seed suite's difficulty (so two model \
versions might plausibly disagree)
- diverse (cover different sub-skills within the domain)
- self-contained (no external context needed)

Do NOT include cases that test for jailbreak / unsafe-content \
refusal — this is a capability eval, not a safety eval.

Return ONLY a JSON array, no prose before or after. Each element \
must be an object with fields: "input" (string, the full prompt), \
"expected" (the ground-truth answer in the same format the seed \
suite uses), and "rationale" (one short sentence explaining why \
this case is plausibly hard).

Example shape:
[{{"input": "...", "expected": "...", "rationale": "..."}}, ...]
"""


def _build_proposer_prompt(
    suite: SuiteConfig,
    batch_size: int,
) -> str:
    """Render the proposer prompt for one batch.

    Includes up to 5 seed examples — enough for format learning,
    few enough that the proposer doesn't pattern-match too tightly
    on the specific phrasings.
    """
    seed = suite.cases[:5]
    examples_blob = "\n".join(
        f"--- example {i + 1} ---\n"
        f"input: {ex.input.strip()}\n"
        f"expected: {ex.expected!r}"
        for i, ex in enumerate(seed)
    )
    return PROPOSER_PROMPT_TEMPLATE.format(
        suite_name=suite.name,
        suite_description=suite.description or "(no description)",
        scoring=suite.scoring,
        seed_examples=examples_blob,
        batch_size=batch_size,
    )


# ---------------------------------------------------------------------------
# Parsing + validity gate
# ---------------------------------------------------------------------------

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def parse_proposer_response(text: str) -> list[dict]:
    """Extract a list of candidate dicts from a proposer response.

    Tolerates triple-backtick ``json`` fences and a small amount of
    surrounding prose. Each returned dict has at minimum an
    ``input`` and an ``expected`` field; everything else is
    preserved as-is for the validity gate to inspect. Returns
    ``[]`` on any parse failure rather than raising so a bad batch
    is just skipped.
    """
    if not text:
        return []
    if len(text) > _MAX_PROPOSER_OUTPUT:
        text = text[:_MAX_PROPOSER_OUTPUT]
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        m = _JSON_ARRAY_RE.search(text)
        if not m:
            return []
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    if not isinstance(obj, list):
        return []
    out: list[dict] = []
    for item in obj:
        if not isinstance(item, dict):
            continue
        if "input" not in item or "expected" not in item:
            continue
        if not isinstance(item["input"], str) or not item["input"].strip():
            continue
        out.append(item)
    return out


def _jaccard_5gram(a: str, b: str) -> float:
    """Character-5-gram Jaccard similarity. Cheap and surprisingly OK.

    Used to drop near-duplicate proposals against the existing
    suite (seed + already-accepted). v1 swaps this for embedding
    similarity; for v0 the false-positive rate is acceptable
    because the consequence of a missed dedup is a slightly
    redundant suite, not a wrong drift conclusion.

    Inputs shorter than the 5-gram window have no 5-grams to
    intersect, so fall back to case-insensitive exact equality —
    otherwise dedup silently no-ops on every short prompt.
    """
    def grams(s: str) -> set[str]:
        s = s.lower()
        return {s[i:i + 5] for i in range(max(0, len(s) - 4))}

    if len(a) < 5 or len(b) < 5:
        return 1.0 if a.strip().lower() == b.strip().lower() else 0.0

    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 0.0
    inter = len(ga & gb)
    union = len(ga | gb)
    return inter / union


# A pair is considered a duplicate above this Jaccard. Calibrated
# loosely — 0.8+ usually means the candidate is a rephrase of an
# existing prompt rather than a meaningfully new case.
DEDUP_JACCARD_THRESHOLD = 0.8


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class DiscoveryResult:
    """Output of one ``discover()`` run."""

    cases: list[EvalCase]
    achieved_power: float
    target_power: float
    target_effect: float
    alpha: float
    proposer_model: str
    baseline_model: str
    challenger_model: str
    seed_suite_name: str
    n_proposed: int
    n_parsed: int
    n_after_dedup: int
    n_after_validity: int
    n_kept: int
    discordant_rate: float
    proposer_spend_usd: float
    verification_spend_usd: float
    # Per-case info contribution; same order as ``cases``.
    case_info: list[float] = field(default_factory=list)
    # One-line rationale per kept case, sourced from the proposer.
    rationales: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Proposer plumbing
# ---------------------------------------------------------------------------


ProviderFactory = Callable[[str], BaseProvider]


def _default_provider_factory(model_id: str) -> BaseProvider:
    """Build a provider for a proposer model identifier.

    Same shape as ``LLMJudgeScorer._default_provider_factory``; kept
    a separate function so tests can patch one without affecting the
    other.
    """
    cfg = resolve_model(model_id)
    if cfg.provider == "anthropic":
        return AnthropicProvider(model=cfg.model, **cfg.params)
    if cfg.provider == "openai":
        return OpenAIProvider(model=cfg.model, **cfg.params)
    raise ValueError(
        f"discover() does not support proposer provider "
        f"'{cfg.provider}' (model={model_id})"
    )


# ---------------------------------------------------------------------------
# Discovery loop
# ---------------------------------------------------------------------------


async def discover(
    baseline: ModelConfig,
    challenger: ModelConfig,
    seed_suite: SuiteConfig,
    proposer_model: str,
    target_power: float = DEFAULT_TARGET_POWER,
    target_effect: float = DEFAULT_TARGET_EFFECT,
    max_cases: int = DEFAULT_MAX_CASES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    alpha: float = DEFAULT_ALPHA,
    concurrency: int = 5,
    cache_dir: str | None = None,
    provider_factory: ProviderFactory | None = None,
    proposer_params: dict | None = None,
) -> DiscoveryResult:
    """Run one discovery pass and return discovered cases + metadata.

    The loop runs until either ``max_cases`` accepted cases is
    reached or the proposer runs out of usable proposals. v0 does
    not early-stop on ``achieved_power``; that knob is reserved for
    v1 when adaptive ranking lands.
    """
    proposer_params = {"temperature": 0.7, **(proposer_params or {})}
    factory = provider_factory or _default_provider_factory
    proposer_provider = factory(proposer_model)

    # Cumulative accumulators. ``accepted_*`` is the running set of
    # cases the loop has committed to keeping.
    accepted_cases: list[EvalCase] = []
    accepted_info: list[float] = []
    accepted_rationales: list[str] = []
    baseline_scores: list[float] = []
    challenger_scores: list[float] = []

    n_proposed = 0
    n_parsed_total = 0
    n_after_dedup_total = 0
    n_after_validity_total = 0
    proposer_spend = 0.0
    verification_spend = 0.0

    # Pool of all texts we've ever seen so dedup is global, not just
    # within a batch. Starts with the seed suite's inputs so the
    # proposer can't just rephrase a seed case.
    seen_inputs: list[str] = [c.input for c in seed_suite.cases]

    # Bail counter. Reset on every batch that produces at least one
    # accepted case; trip when we've burned _MAX_CONSECUTIVE_STALE_BATCHES
    # batches without accepting anything new.
    stale_batches = 0

    try:
        while len(accepted_cases) < max_cases:
            # ---- 1. Propose ----
            prompt = _build_proposer_prompt(seed_suite, batch_size)
            completion = await proposer_provider.complete(
                prompt, **proposer_params
            )
            proposer_spend += cost_of(
                proposer_model,
                completion.input_tokens,
                completion.output_tokens,
            )
            n_proposed += batch_size

            candidates = parse_proposer_response(completion.output_text)
            n_parsed_total += len(candidates)
            if not candidates:
                # Proposer returned junk — bail to avoid an infinite loop.
                break

            # ---- 2. Dedup ----
            fresh: list[dict] = []
            for cand in candidates:
                inp = cand["input"]
                if any(_jaccard_5gram(inp, prior) >= DEDUP_JACCARD_THRESHOLD
                       for prior in seen_inputs):
                    continue
                fresh.append(cand)
                seen_inputs.append(inp)
            n_after_dedup_total += len(fresh)
            if not fresh:
                stale_batches += 1
                if stale_batches >= _MAX_CONSECUTIVE_STALE_BATCHES:
                    break
                continue

            # ---- 3. Verify with baseline + challenger ----
            candidate_suite = SuiteConfig(
                name=f"{seed_suite.name}__candidates",
                description=f"(internal) discovery batch for "
                            f"{seed_suite.name}",
                scoring=seed_suite.scoring,
                model_params=dict(seed_suite.model_params),
                judge_model=seed_suite.judge_model,
                cases=[
                    EvalCase(input=c["input"], expected=c["expected"],
                             tags=["discovered"])
                    for c in fresh
                ],
            )
            b_run, c_run = await asyncio.gather(
                run_suite(candidate_suite, baseline, concurrency=concurrency,
                          cache_dir=cache_dir),
                run_suite(candidate_suite, challenger, concurrency=concurrency,
                          cache_dir=cache_dir),
            )
            verification_spend += b_run.total_cost_usd + c_run.total_cost_usd

            # ---- 4. Validity gate + ranking ----
            batch_accepted = 0
            for i, cand in enumerate(fresh):
                b_score = b_run.cases[i].score
                c_score = c_run.cases[i].score
                # Validity: both-zero usually means the case is
                # under-specified or the proposer's ``expected`` is
                # wrong. Drop it — we can't tell which side is right.
                if b_score == 0.0 and c_score == 0.0:
                    continue
                # Info contribution. For binary scorers this is
                # equivalent to a discordance indicator; for
                # continuous scorers we use the absolute paired
                # difference (v1 will switch to centered squared
                # difference once continuous ranking is enabled).
                info = (
                    1.0 if (b_score != c_score) else 0.0
                )
                if info <= 0.0:
                    # Concordant pair — zero McNemar information.
                    continue

                n_after_validity_total += 1
                accepted_cases.append(EvalCase(
                    input=cand["input"],
                    expected=cand["expected"],
                    tags=["discovered", f"info:{info:.2f}"],
                ))
                accepted_info.append(info)
                accepted_rationales.append(
                    str(cand.get("rationale", "")).strip()
                )
                baseline_scores.append(b_score)
                challenger_scores.append(c_score)
                batch_accepted += 1
                if len(accepted_cases) >= max_cases:
                    break

            if batch_accepted == 0:
                stale_batches += 1
                if stale_batches >= _MAX_CONSECUTIVE_STALE_BATCHES:
                    break
            else:
                stale_batches = 0
    finally:
        await proposer_provider.close()

    # ---- 5. Achieved power on accepted set ----
    if len(accepted_cases) >= 2:
        power = power_analysis(
            baseline_scores, challenger_scores,
            alpha=alpha, power=target_power,
            target_effect=target_effect,
        )
        achieved = power["observed_power"]
    else:
        achieved = 0.0

    discordant_rate = (
        sum(1 for s in accepted_info if s > 0) / max(1, n_after_validity_total)
    )

    return DiscoveryResult(
        cases=accepted_cases,
        achieved_power=round(achieved, 4),
        target_power=target_power,
        target_effect=target_effect,
        alpha=alpha,
        proposer_model=proposer_model,
        baseline_model=baseline.model,
        challenger_model=challenger.model,
        seed_suite_name=seed_suite.name,
        n_proposed=n_proposed,
        n_parsed=n_parsed_total,
        n_after_dedup=n_after_dedup_total,
        n_after_validity=n_after_validity_total,
        n_kept=len(accepted_cases),
        discordant_rate=round(discordant_rate, 4),
        proposer_spend_usd=round(proposer_spend, 6),
        verification_spend_usd=round(verification_spend, 6),
        case_info=accepted_info,
        rationales=accepted_rationales,
    )


# ---------------------------------------------------------------------------
# Suite emission
# ---------------------------------------------------------------------------


_BIAS_CAVEAT = (
    "IMPORTANT: cases were selected on divergence between the named "
    "baseline/challenger pair. The achieved_power figure is the "
    "sensitivity of THIS suite, not an unbiased estimate of "
    "population drift between these two models on arbitrary prompts."
)


def to_suite_yaml(result: DiscoveryResult) -> dict:
    """Render a DiscoveryResult as a ``SuiteConfig``-shaped dict.

    The full provenance lives in ``description`` so it travels with
    the suite even when the suite is copied / committed elsewhere.
    Includes the selection-bias caveat verbatim so a reader cannot
    misuse the suite's achieved-power number.
    """
    desc_lines = [
        f"Discovered by rift discover targeting drift between "
        f"{result.baseline_model} and {result.challenger_model} on seed "
        f"suite '{result.seed_suite_name}'.",
        f"Proposer: {result.proposer_model}.",
        f"Target power: {result.target_power} at "
        f"Δ={result.target_effect}, α={result.alpha}.",
        f"Achieved power: {result.achieved_power}.",
        f"Discordant rate: {result.discordant_rate}.",
        f"n_proposed={result.n_proposed}, "
        f"n_parsed={result.n_parsed}, "
        f"n_after_dedup={result.n_after_dedup}, "
        f"n_after_validity={result.n_after_validity}, "
        f"n_kept={result.n_kept}.",
        f"Spend: proposer ${result.proposer_spend_usd:.4f}, "
        f"verification ${result.verification_spend_usd:.4f}.",
        "",
        _BIAS_CAVEAT,
    ]
    return {
        "name": (
            f"{result.seed_suite_name}__discovered_"
            f"{_sanitize(result.baseline_model)}_vs_"
            f"{_sanitize(result.challenger_model)}"
        ),
        "description": "\n".join(desc_lines),
        "scoring": "exact_match",
        "cases": [
            {
                "input": c.input,
                "expected": c.expected,
                "tags": c.tags,
            }
            for c in result.cases
        ],
    }


def _sanitize(model: str) -> str:
    """Reduce a model id to a filename-safe slug."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", model)
