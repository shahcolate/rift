"""Output-token decomposition: tokenizer effect vs. verbosity effect.

A raw output-token ratio between two models conflates two different
things and they need different responses:

* **Tokenizer effect** — the same string tokenizes to more tokens on
  the challenger than on the baseline. Pure API-pricing artifact; no
  change in what the model actually says. Detectable by re-tokenizing
  each model's output through the other model's tokenizer.
* **Verbosity effect** — the challenger writes longer outputs. A real
  behavioral change that shows up as tokens but would also show up
  as characters, words, or user-perceived length. Detectable via
  char counts.

This module provides the pure-math primitives that take per-case
token counts (four counts per case, spelled out in :class:`OutputRow`)
and return the decomposition plus cost attribution. The counts
themselves come from Anthropic's ``count_tokens`` endpoint in the
benchmark script — that code is orchestration and lives in
``benchmarks/analyze_output_tokens.py`` so this module stays
offline-testable.

Decomposition assumes approximate multiplicativity:

    observed_ratio ≈ tokenizer_ratio × verbosity_ratio

which holds when the tokenizer behaves uniformly across text lengths
(the baseline finding on the input side). On pathological inputs
(JSON-heavy, multilingual) it can drift; the attribution is a
reasonable first pass, not a dollar-exact accounting.
"""

from __future__ import annotations

from dataclasses import dataclass

from .pricing import lookup as lookup_price


@dataclass(frozen=True)
class OutputRow:
    """Per-case output-side token and character counts.

    All fields are filled by the benchmark script before this module
    sees them. ``actual_tokens_*`` are the counts the API returned
    when the model actually generated the text (what users pay).
    The ``retok_*`` fields come from a separate ``count_tokens`` call
    on each output text under each model's tokenizer — they give us
    the pure-tokenizer counterfactual.
    """

    case_index: int
    baseline_chars: int
    challenger_chars: int
    # Observed token counts from the live run (what the bill was).
    baseline_actual_tokens: int
    challenger_actual_tokens: int
    # Re-tokenized counts: each output text under each tokenizer.
    # Used to isolate the tokenizer effect from verbosity.
    baseline_output_under_baseline_tokenizer: int
    baseline_output_under_challenger_tokenizer: int
    challenger_output_under_baseline_tokenizer: int
    challenger_output_under_challenger_tokenizer: int


@dataclass(frozen=True)
class Decomposition:
    """Ratios that answer ‘how much is tokenizer, how much is verbosity’.

    Each ratio is ``challenger / baseline``, so a value >1 means the
    challenger side is more expensive/longer on that dimension.
    """

    n: int
    observed_ratio: float
    tokenizer_ratio_on_baseline: float
    tokenizer_ratio_on_challenger: float
    tokenizer_ratio: float              # average of the two
    verbosity_ratio: float
    # Sanity: observed should be close to tokenizer × verbosity.
    multiplicative_residual: float


@dataclass(frozen=True)
class CostAttribution:
    """Dollar decomposition of the output-cost delta.

    Attributes the observed output-cost delta to tokenizer vs.
    verbosity under the assumption that switching tokenizer is what
    happens first (the cost change from baseline→tokenizer-swapped,
    holding verbosity fixed) and verbosity change happens second.
    Order matters for the split, not the total: the two components
    sum to the observed delta.
    """

    baseline_output_cost_usd: float
    challenger_output_cost_usd: float
    delta_usd: float
    tokenizer_component_usd: float
    verbosity_component_usd: float
    price_change_component_usd: float   # nonzero only when $/Mtok differs


def _safe_ratio(num: float, den: float) -> float:
    return num / den if den else 0.0


def decompose(rows: list[OutputRow]) -> Decomposition:
    """Compute observed, tokenizer, and verbosity ratios from paired rows.

    Rows with empty output on either side are dropped — a tokenizer
    ratio is undefined on an empty string. If every row is dropped,
    all ratios come back as zero rather than raising.
    """
    usable = [
        r for r in rows
        if r.baseline_chars > 0 and r.challenger_chars > 0
        and r.baseline_output_under_baseline_tokenizer > 0
        and r.challenger_output_under_baseline_tokenizer > 0
    ]
    n = len(usable)
    if n == 0:
        return Decomposition(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    sum_b_actual = sum(r.baseline_actual_tokens for r in usable)
    sum_c_actual = sum(r.challenger_actual_tokens for r in usable)
    observed = _safe_ratio(sum_c_actual, sum_b_actual)

    # Tokenizer effect = same text, different tokenizer. Compute on
    # both sides of the paired output distribution and average so the
    # result isn't driven by whichever model happened to write more.
    tok_on_base = _safe_ratio(
        sum(r.baseline_output_under_challenger_tokenizer for r in usable),
        sum(r.baseline_output_under_baseline_tokenizer for r in usable),
    )
    tok_on_chal = _safe_ratio(
        sum(r.challenger_output_under_challenger_tokenizer for r in usable),
        sum(r.challenger_output_under_baseline_tokenizer for r in usable),
    )
    tokenizer = (tok_on_base + tok_on_chal) / 2

    verbosity = _safe_ratio(
        sum(r.challenger_chars for r in usable),
        sum(r.baseline_chars for r in usable),
    )

    residual = observed - (tokenizer * verbosity)
    return Decomposition(
        n=n,
        observed_ratio=round(observed, 4),
        tokenizer_ratio_on_baseline=round(tok_on_base, 4),
        tokenizer_ratio_on_challenger=round(tok_on_chal, 4),
        tokenizer_ratio=round(tokenizer, 4),
        verbosity_ratio=round(verbosity, 4),
        multiplicative_residual=round(residual, 4),
    )


def cost_attribution(
    rows: list[OutputRow],
    baseline_model: str,
    challenger_model: str,
    enterprise_multiplier: float = 1.0,
) -> CostAttribution:
    """Split the observed output-cost delta into tokenizer + verbosity.

    Walks the counterfactual baseline → tokenizer-swap → add-verbosity
    → price-change → challenger so the three components sum exactly to
    the observed delta. ``price_change_component_usd`` is zero when
    list ``$/Mtok`` is identical across the two models (the common
    case within a family — Opus 4.6 and 4.7 both list at $75/Mtok
    output).
    """
    b_price = lookup_price(baseline_model)
    c_price = lookup_price(challenger_model)
    if b_price is None or c_price is None:
        return CostAttribution(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    b_out_rate = b_price.output_per_mtok * enterprise_multiplier / 1_000_000
    c_out_rate = c_price.output_per_mtok * enterprise_multiplier / 1_000_000

    b_tokens = sum(r.baseline_actual_tokens for r in rows)
    c_tokens = sum(r.challenger_actual_tokens for r in rows)

    baseline_cost = b_tokens * b_out_rate
    challenger_cost = c_tokens * c_out_rate
    delta = challenger_cost - baseline_cost

    # Counterfactual step 1: re-tokenize baseline's outputs under the
    # challenger's tokenizer, still at the baseline price. Uses the
    # summed re-tokenized counts on the baseline's own outputs so the
    # ‘same text, different tokenizer’ arithmetic is exact, not
    # extrapolated from the ratio.
    tokenizer_swapped_tokens = sum(
        r.baseline_output_under_challenger_tokenizer for r in rows
    )
    after_tokenizer_cost = tokenizer_swapped_tokens * b_out_rate
    tokenizer_component = after_tokenizer_cost - baseline_cost

    # Counterfactual step 2: add verbosity — move from baseline's
    # outputs (re-tokenized) to challenger's actual outputs, still at
    # baseline price. Captures the dollar hit of ‘the model is writing
    # more’ in isolation from the tokenizer change.
    after_verbosity_cost = c_tokens * b_out_rate
    verbosity_component = after_verbosity_cost - after_tokenizer_cost

    # Counterfactual step 3: price change — only matters when
    # $/Mtok differs between the two models (e.g. cross-family
    # comparisons). Within family it's zero.
    price_component = c_tokens * (c_out_rate - b_out_rate)

    return CostAttribution(
        baseline_output_cost_usd=round(baseline_cost, 6),
        challenger_output_cost_usd=round(challenger_cost, 6),
        delta_usd=round(delta, 6),
        tokenizer_component_usd=round(tokenizer_component, 6),
        verbosity_component_usd=round(verbosity_component, 6),
        price_change_component_usd=round(price_component, 6),
    )
