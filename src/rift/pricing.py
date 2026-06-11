"""Per-model token pricing and cost computation.

Rift treats cost as a first-class drift signal. With Anthropic's shift to
token-based Enterprise pricing and the release of Claude Opus 4.7, the
question "did we regress?" is inseparable from "at what cost?". A model
that matches its predecessor's quality at 40% of the spend is an
improvement even when the raw score is flat, and a model that gains 2
points of accuracy at 3x the spend may not be.

The catalog below reflects public list pricing (per 1M tokens, USD).
Enterprise contracts typically negotiate a flat per-token rate with
committed volume; we model this via an optional ``enterprise_multiplier``
applied uniformly to both input and output prices.

All numbers are published rates as of 2026-05 (Opus 4.8 launch, which
also cut the Opus 4.5-generation list price to $5/$25). Update
``PRICING`` when rates change — do not hardcode elsewhere.

IMPORTANT: this catalog is **standard-mode list price only** — one cell
of the provider's serving-configuration matrix. The same model also
ships at other prices: Anthropic's Batch API is −50% on both sides,
fast mode is a premium (Opus 4.8 fast = $10/$50, i.e. exactly Fable 5's
standard rate; Opus 4.6/4.7 fast = $30/$150), cache reads bill at 0.1×
input, and US-only inference_geo adds 1.1×. Any published cost
comparison built on this catalog must say so, and should situate its
headline multiple against the configurations a reader could actually
buy (see benchmarks/fable5_vs_opus47/analysis.md, "The price in
context", for the worked example: Fable's "2× premium" vs live Opus 4.7
is 0.5× vs Opus-4.7-fast and ~1× vs live Opus when Fable is batched).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenPrice:
    """Dollar cost per 1M input / output tokens on list pricing."""

    input_per_mtok: float
    output_per_mtok: float
    tier: str = "standard"  # "standard" | "enterprise"

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        """Dollar cost for a single completion."""
        return (
            input_tokens * self.input_per_mtok / 1_000_000
            + output_tokens * self.output_per_mtok / 1_000_000
        )


PRICING: dict[str, TokenPrice] = {
    # Anthropic — Claude 5 family (Mythos-class tier, above Opus).
    # NOTE: Fable 5 uses a new tokenizer that yields ~30% more tokens for
    # the same content vs Opus-tier models, so per-token price alone
    # understates the effective cost delta — compare $/correct, not rates.
    "claude-fable-5":           TokenPrice(10.00, 50.00),

    # Anthropic — Claude 4 family (list price, per 1M tokens)
    # The Opus 4.5 generation (4.5/4.6/4.7/4.8) lists at $5 / $25 — a
    # 3x cut from the original Opus 4 / 4.1 rate of $15 / $75, which the
    # deprecated dated build below retains.
    "claude-opus-4-8":          TokenPrice( 5.00, 25.00),
    "claude-opus-4-7":          TokenPrice( 5.00, 25.00),
    "claude-opus-4-6":          TokenPrice( 5.00, 25.00),
    "claude-opus-4-20250514":   TokenPrice(15.00, 75.00),  # Opus 4 (deprecated)
    "claude-sonnet-4-6":        TokenPrice( 3.00, 15.00),
    "claude-sonnet-4-20250514": TokenPrice( 3.00, 15.00),
    "claude-haiku-4-5-20251001": TokenPrice(1.00,  5.00),

    # Anthropic — legacy
    "claude-3-5-sonnet-20241022": TokenPrice(3.00, 15.00),
    "claude-3-5-haiku-20241022":  TokenPrice(0.80,  4.00),

    # OpenAI — frontier
    "gpt-5.5":        TokenPrice( 5.00, 20.00),
    "gpt-4o":         TokenPrice( 2.50, 10.00),
    "gpt-4-turbo":    TokenPrice(10.00, 30.00),
    "o1":             TokenPrice(15.00, 60.00),
    "o3":             TokenPrice(10.00, 40.00),

    # Google Gemini — list price as published by Google AI (per 1M tokens).
    # Gemini 3.5 Flash (May 2026): $1.50 input / $9.00 output. Thinking
    # tokens are billed as output — GoogleProvider sums them into
    # ``output_tokens`` so this catalog entry covers both.
    "gemini-3.5-flash":         TokenPrice( 1.50,  9.00),
    "gemini-3-flash-preview":   TokenPrice( 0.50,  3.00),
}


def lookup(model: str) -> TokenPrice | None:
    """Resolve a model string to its pricing entry.

    Falls back to family-prefix matching so dated variants
    (e.g. ``claude-opus-4-7-20260315``) inherit their family's price.
    """
    if model in PRICING:
        return PRICING[model]
    # family-prefix fallback, longest match wins
    prefix_match = None
    for key in PRICING:
        if model.startswith(key) and (prefix_match is None or len(key) > len(prefix_match)):
            prefix_match = key
    return PRICING[prefix_match] if prefix_match else None


def cost_of(model: str, input_tokens: int, output_tokens: int,
            enterprise_multiplier: float = 1.0) -> float:
    """Compute USD cost of a completion.

    ``enterprise_multiplier`` models contracted rates: 0.7 = 30% discount.
    Returns 0.0 if the model is unknown; callers should check with ``lookup``
    when a strict guarantee is required.
    """
    price = lookup(model)
    if price is None:
        return 0.0
    return price.cost(input_tokens, output_tokens) * enterprise_multiplier
