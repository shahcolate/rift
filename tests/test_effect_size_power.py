"""Tests for effect-size, BH correction, and power analysis additions."""

import math

import pytest

from rift.comparator import (
    benjamini_hochberg,
    compare_runs,
    power_analysis,
)


class TestEffectSize:
    def test_binary_uses_cohens_h(self):
        b = [1.0] * 10 + [0.0] * 10
        c = [1.0] * 5 + [0.0] * 15
        r = compare_runs(b, c, "A", "B", "suite")
        assert r.effect_size_kind == "cohens_h"
        # Direction: challenger has lower proportion → negative h.
        assert r.effect_size < 0

    def test_continuous_uses_hedges_g(self):
        b = [0.80, 0.82, 0.85, 0.79, 0.81, 0.84, 0.83, 0.82]
        c = [0.62, 0.65, 0.63, 0.60, 0.64, 0.66, 0.61, 0.63]
        r = compare_runs(b, c, "A", "B", "suite")
        assert r.effect_size_kind == "hedges_g"
        assert r.effect_size < 0
        # Big mean difference, low variance → large effect.
        assert r.effect_size_magnitude == "large"

    def test_no_variation_yields_zero_effect(self):
        # All identical binary → mcnemar runs, but effect is exactly 0.
        b = [1.0] * 5
        c = [1.0] * 5
        r = compare_runs(b, c, "A", "B", "suite")
        assert r.effect_size == 0.0
        assert r.effect_size_magnitude == "negligible"

    def test_short_input_yields_none(self):
        # n<2 → effect size is not defined.
        r = compare_runs([1.0], [0.0], "A", "B", "suite")
        assert r.effect_size_kind == "none"
        assert r.effect_size == 0.0

    def test_magnitude_buckets(self):
        # Very small proportion diff → negligible.
        b = [1.0] * 50 + [0.0] * 50
        c = [1.0] * 49 + [0.0] * 51
        r = compare_runs(b, c, "A", "B", "suite")
        assert r.effect_size_magnitude in {"negligible", "small"}

    def test_cohens_h_sign_convention(self):
        # Challenger BETTER than baseline → positive h.
        b = [1.0] * 4 + [0.0] * 6
        c = [1.0] * 8 + [0.0] * 2
        r = compare_runs(b, c, "A", "B", "suite")
        assert r.effect_size > 0


class TestBenjaminiHochberg:
    def test_empty(self):
        q, rej = benjamini_hochberg([])
        assert q == []
        assert rej == []

    def test_preserves_order(self):
        ps = [0.04, 0.001, 0.5, 0.02]
        q, _ = benjamini_hochberg(ps, alpha=0.05)
        # Same length, same positions.
        assert len(q) == 4

    def test_monotonic_in_p(self):
        # Larger p ⇒ at least as large q.
        ps = [0.001, 0.01, 0.02, 0.04, 0.08, 0.5]
        q, _ = benjamini_hochberg(ps)
        for i in range(len(q) - 1):
            # By construction q is monotone non-decreasing in
            # rank order, and ps is already sorted.
            assert q[i] <= q[i + 1] + 1e-9

    def test_rejection_at_alpha(self):
        ps = [0.001, 0.008, 0.02, 0.5, 0.5]
        q, rej = benjamini_hochberg(ps, alpha=0.05)
        # The two clearly-small p-values should be rejected.
        assert rej[0] is True
        assert rej[1] is True
        assert rej[3] is False
        assert rej[4] is False

    def test_clipped_to_one(self):
        q, _ = benjamini_hochberg([1.0, 1.0, 1.0])
        assert all(qi <= 1.0 for qi in q)

    def test_no_correction_when_single_test(self):
        q, _ = benjamini_hochberg([0.03])
        assert q[0] == pytest.approx(0.03)


class TestPowerAnalysis:
    def test_high_power_at_large_effect(self):
        # Binary, strong shift, n=40 — should be well-powered.
        b = [1.0] * 36 + [0.0] * 4
        c = [1.0] * 12 + [0.0] * 28
        p = power_analysis(b, c)
        assert p["observed_power"] >= 0.95
        assert p["min_detectable_effect"] > 0
        assert p["observed_effect_kind"] == "cohens_h"

    def test_low_power_at_tiny_effect(self):
        b = [0.5, 0.5, 0.6, 0.5]
        c = [0.5, 0.5, 0.55, 0.5]
        p = power_analysis(b, c)
        assert p["observed_power"] < 0.5

    def test_n_for_target_grows_with_smaller_target(self):
        b = [0.5] * 10
        c = [0.6] * 10
        p_small = power_analysis(b, c, target_effect=0.1)
        p_big = power_analysis(b, c, target_effect=0.5)
        assert p_small["n_for_target"] > p_big["n_for_target"]

    def test_short_input_safe(self):
        p = power_analysis([1.0], [0.0])
        assert math.isinf(p["min_detectable_effect"])
        assert p["observed_power"] == 0.0
