"""Tests for token-price lookup and cost computation."""

import math

from rift.pricing import PRICING, cost_of, lookup


class TestLookup:
    def test_exact_match(self):
        p = lookup("claude-opus-4-7")
        assert p is not None
        assert p.input_per_mtok == 15.00

    def test_family_prefix_fallback(self):
        # A future dated variant should inherit the family's price.
        p = lookup("claude-opus-4-7-20260801")
        assert p is not None
        assert p.input_per_mtok == PRICING["claude-opus-4-7"].input_per_mtok

    def test_unknown_model_returns_none(self):
        assert lookup("fake-model-9000") is None


class TestCostOf:
    def test_basic_cost(self):
        # 1M input + 1M output at opus-4-7 list price = 15 + 75 = $90
        c = cost_of("claude-opus-4-7", 1_000_000, 1_000_000)
        assert math.isclose(c, 90.0)

    def test_small_cost(self):
        # 500 input + 200 output: 500*15/1e6 + 200*75/1e6 = 0.0075+0.015=.0225
        c = cost_of("claude-opus-4-7", 500, 200)
        assert math.isclose(c, 0.0225)

    def test_enterprise_multiplier(self):
        full = cost_of("claude-opus-4-7", 1_000_000, 1_000_000)
        discounted = cost_of(
            "claude-opus-4-7", 1_000_000, 1_000_000, enterprise_multiplier=0.65
        )
        assert math.isclose(discounted, full * 0.65)

    def test_unknown_model_zero(self):
        assert cost_of("fake-model-9000", 1000, 1000) == 0.0
