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

All numbers are published rates as of 2026-04. Update ``PRICING`` when
rates change — do not hardcode elsewhere.
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
    # Anthropic — Claude 4 family (list price, per 1M tokens)
    "claude-opus-4-7":          TokenPrice(15.00, 75.00),
    "claude-opus-4-6":          TokenPrice(15.00, 75.00),
    "claude-opus-4-20250514":   TokenPrice(15.00, 75.00),
    "claude-sonnet-4-6":        TokenPrice( 3.00, 15.00),
    "claude-sonnet-4-20250514": TokenPrice( 3.00, 15.00),
    "claude-haiku-4-5-20251001": TokenPrice(1.00,  5.00),

    # Anthropic — legacy
    "claude-3-5-sonnet-20241022": TokenPrice(3.00, 15.00),
    "claude-3-5-haiku-20241022":  TokenPrice(0.80,  4.00),

    # OpenAI — frontier
    "gpt-4o":         TokenPrice(2.50, 10.00),
    "gpt-4-turbo":    TokenPrice(10.00, 30.00),
    "o1":             TokenPrice(15.00, 60.00),
    "o3":             TokenPrice(10.00, 40.00),
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
