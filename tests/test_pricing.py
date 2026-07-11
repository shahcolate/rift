"""Tests for token-price lookup and cost computation."""

import math

from rift.pricing import PRICING, cost_of, lookup


class TestLookup:
    def test_exact_match(self):
        p = lookup("claude-opus-4-7")
        assert p is not None
        assert p.input_per_mtok == 5.00

    def test_family_prefix_fallback(self):
        # A future dated variant should inherit the family's price.
        p = lookup("claude-opus-4-7-20260801")
        assert p is not None
        assert p.input_per_mtok == PRICING["claude-opus-4-7"].input_per_mtok

    def test_opus_4_8_priced(self):
        # 4.8 must have an explicit entry: the family-prefix fallback
        # does NOT cover it (it doesn't start with 4-7/4-6/4-20250514),
        # so a missing entry would silently price every case at $0.
        # The Opus 4.5 generation lists at $5 / $25.
        p = lookup("claude-opus-4-8")
        assert p is not None
        assert p.input_per_mtok == 5.00
        assert p.output_per_mtok == 25.00

    def test_gpt_5_6_tiers_priced(self):
        # Each named tier is a distinct product: the family-prefix
        # fallback does NOT bridge "-sol"/"-terra"/"-luna" (non-dated
        # suffixes), so a missing entry would silently price a tier at
        # $0. Launch list price, July 2026.
        for model, inp, out in [
            ("gpt-5.6-sol", 5.00, 30.00),
            ("gpt-5.6-terra", 2.50, 15.00),
            ("gpt-5.6-luna", 1.00, 6.00),
        ]:
            p = lookup(model)
            assert p is not None, model
            assert p.input_per_mtok == inp
            assert p.output_per_mtok == out

    def test_gpt_5_6_bare_priced_as_sol(self):
        # Bare "gpt-5.6" is OpenAI's alias for Sol. MODEL_ALIASES pins it,
        # but saved run files may carry the bare id.
        p = lookup("gpt-5.6")
        assert p is not None
        assert (p.input_per_mtok, p.output_per_mtok) == (5.00, 30.00)

    def test_gpt_5_6_dated_variant_inherits_tier_price(self):
        # A dated snapshot inherits its tier's price via the longest
        # prefix match — the tier entry, not the bare-family entry.
        p = lookup("gpt-5.6-luna-2026-07-09")
        assert p is not None
        assert p.output_per_mtok == PRICING["gpt-5.6-luna"].output_per_mtok

    def test_gpt_5_6_named_submodel_not_inherited(self):
        # An unknown *named* submodel must not inherit the family price
        # (same guard that keeps gpt-4o-mini off gpt-4o rates).
        assert lookup("gpt-5.6-nano") is None

    def test_unknown_model_returns_none(self):
        assert lookup("fake-model-9000") is None


class TestCostOf:
    def test_basic_cost(self):
        # 1M input + 1M output at opus-4-7 list price = 5 + 25 = $30
        c = cost_of("claude-opus-4-7", 1_000_000, 1_000_000)
        assert math.isclose(c, 30.0)

    def test_small_cost(self):
        # 500 input + 200 output: 500*5/1e6 + 200*25/1e6 = 0.0025+0.005=.0075
        c = cost_of("claude-opus-4-7", 500, 200)
        assert math.isclose(c, 0.0075)

    def test_enterprise_multiplier(self):
        full = cost_of("claude-opus-4-7", 1_000_000, 1_000_000)
        discounted = cost_of(
            "claude-opus-4-7", 1_000_000, 1_000_000, enterprise_multiplier=0.65
        )
        assert math.isclose(discounted, full * 0.65)

    def test_unknown_model_zero(self):
        assert cost_of("fake-model-9000", 1000, 1000) == 0.0
