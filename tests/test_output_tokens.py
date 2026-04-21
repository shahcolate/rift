"""Tests for the output-token decomposition module."""

from __future__ import annotations

import pytest

from rift.output_tokens import (
    CostAttribution,
    Decomposition,
    OutputRow,
    cost_attribution,
    decompose,
)


def _row(
    i: int = 0,
    *,
    b_chars: int = 100,
    c_chars: int = 150,
    b_actual: int = 100,
    c_actual: int = 170,
    b_under_b: int = 100,
    b_under_c: int = 110,
    c_under_b: int = 150,
    c_under_c: int = 170,
) -> OutputRow:
    """Build an :class:`OutputRow` with sensible per-field defaults.

    Defaults describe a scenario where the challenger writes 1.5× more
    characters (verbosity) and its tokenizer packs them into 10% more
    tokens than the baseline's tokenizer (tokenizer effect). The test
    fixtures mutate only what each test needs so intent is visible at
    the call site.
    """
    return OutputRow(
        case_index=i,
        baseline_chars=b_chars,
        challenger_chars=c_chars,
        baseline_actual_tokens=b_actual,
        challenger_actual_tokens=c_actual,
        baseline_output_under_baseline_tokenizer=b_under_b,
        baseline_output_under_challenger_tokenizer=b_under_c,
        challenger_output_under_baseline_tokenizer=c_under_b,
        challenger_output_under_challenger_tokenizer=c_under_c,
    )


class TestDecompose:
    def test_empty_input_returns_zeros(self):
        d = decompose([])
        assert d.n == 0
        assert d.observed_ratio == 0.0
        assert d.tokenizer_ratio == 0.0
        assert d.verbosity_ratio == 0.0

    def test_drops_rows_with_empty_output(self):
        # Second row has challenger_chars=0 → unusable; only the first counts.
        rows = [
            _row(i=0),
            _row(i=1, c_chars=0, c_under_b=0, c_under_c=0),
        ]
        d = decompose(rows)
        assert d.n == 1

    def test_pure_tokenizer_effect_zero_verbosity(self):
        # Identical text length, but challenger tokenizer chops it into 10% more.
        rows = [_row(
            b_chars=100, c_chars=100,
            b_actual=100, c_actual=110,
            b_under_b=100, b_under_c=110,
            c_under_b=100, c_under_c=110,
        )]
        d = decompose(rows)
        assert d.verbosity_ratio == pytest.approx(1.0, abs=1e-4)
        assert d.tokenizer_ratio == pytest.approx(1.1, abs=1e-4)
        assert d.observed_ratio == pytest.approx(1.1, abs=1e-4)

    def test_pure_verbosity_effect_zero_tokenizer(self):
        # Both tokenizers agree; challenger just writes 50% more.
        rows = [_row(
            b_chars=100, c_chars=150,
            b_actual=100, c_actual=150,
            b_under_b=100, b_under_c=100,
            c_under_b=150, c_under_c=150,
        )]
        d = decompose(rows)
        assert d.tokenizer_ratio == pytest.approx(1.0, abs=1e-4)
        assert d.verbosity_ratio == pytest.approx(1.5, abs=1e-4)
        assert d.observed_ratio == pytest.approx(1.5, abs=1e-4)

    def test_combined_effect_multiplicative(self):
        # 1.5x verbosity * 1.1x tokenizer = 1.65x observed (approximately).
        rows = [_row(
            b_chars=100, c_chars=150,
            b_actual=100, c_actual=165,
            b_under_b=100, b_under_c=110,
            c_under_b=150, c_under_c=165,
        )]
        d = decompose(rows)
        assert d.verbosity_ratio == pytest.approx(1.5, abs=1e-4)
        assert d.tokenizer_ratio == pytest.approx(1.1, abs=1e-4)
        assert d.observed_ratio == pytest.approx(1.65, abs=1e-4)
        assert abs(d.multiplicative_residual) < 1e-4

    def test_tokenizer_ratio_averages_both_sides(self):
        # Baseline side shows 1.10x; challenger side shows 1.20x. Average is 1.15.
        rows = [_row(
            b_under_b=100, b_under_c=110,
            c_under_b=100, c_under_c=120,
        )]
        d = decompose(rows)
        assert d.tokenizer_ratio_on_baseline == pytest.approx(1.10, abs=1e-4)
        assert d.tokenizer_ratio_on_challenger == pytest.approx(1.20, abs=1e-4)
        assert d.tokenizer_ratio == pytest.approx(1.15, abs=1e-4)

    def test_aggregates_across_rows(self):
        # Two rows; ratio must be computed on sums, not per-row means.
        rows = [
            _row(i=0, b_actual=100, c_actual=150),
            _row(i=1, b_actual=200, c_actual=250),
        ]
        d = decompose(rows)
        # (150 + 250) / (100 + 200) = 400 / 300 = 1.3333
        assert d.observed_ratio == pytest.approx(1.3333, abs=1e-4)
        assert d.n == 2

    def test_returns_decomposition_dataclass(self):
        rows = [_row()]
        d = decompose(rows)
        assert isinstance(d, Decomposition)


class TestCostAttribution:
    def test_unknown_model_returns_zeros(self):
        attr = cost_attribution([_row()], "not-a-model", "also-not")
        assert attr == CostAttribution(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def test_components_sum_to_delta_within_family(self):
        # Opus 4.6 → 4.7: same list price, so the price component is zero
        # and tokenizer + verbosity must sum exactly to the delta.
        rows = [_row(
            b_chars=1000, c_chars=1500,
            b_actual=1000, c_actual=1650,
            b_under_b=1000, b_under_c=1100,
            c_under_b=1500, c_under_c=1650,
        )]
        attr = cost_attribution(rows, "claude-opus-4-6", "claude-opus-4-7")
        assert attr.price_change_component_usd == pytest.approx(0.0, abs=1e-9)
        summed = (
            attr.tokenizer_component_usd
            + attr.verbosity_component_usd
            + attr.price_change_component_usd
        )
        assert summed == pytest.approx(attr.delta_usd, abs=1e-6)

    def test_components_sum_to_delta_cross_family(self):
        # Cross-family (Opus vs Sonnet): the price-change component
        # carries the list-price delta. Components must still sum exactly.
        rows = [_row(
            b_chars=1000, c_chars=1500,
            b_actual=1000, c_actual=1650,
            b_under_b=1000, b_under_c=1100,
            c_under_b=1500, c_under_c=1650,
        )]
        attr = cost_attribution(rows, "claude-sonnet-4-6", "claude-opus-4-7")
        summed = (
            attr.tokenizer_component_usd
            + attr.verbosity_component_usd
            + attr.price_change_component_usd
        )
        assert summed == pytest.approx(attr.delta_usd, abs=1e-6)
        # Sonnet ($15/Mtok out) → Opus ($75/Mtok out): the price jump
        # alone dominates and must be positive.
        assert attr.price_change_component_usd > 0

    def test_pure_tokenizer_case(self):
        # No verbosity change: the verbosity component should be ~0,
        # and all the delta should be attributed to the tokenizer.
        rows = [_row(
            b_chars=1000, c_chars=1000,
            b_actual=1000, c_actual=1100,
            b_under_b=1000, b_under_c=1100,
            c_under_b=1000, c_under_c=1100,
        )]
        attr = cost_attribution(rows, "claude-opus-4-6", "claude-opus-4-7")
        # tokens are equal on both outputs under baseline tokenizer → zero verbosity component.
        assert attr.verbosity_component_usd == pytest.approx(0.0, abs=1e-6)
        assert attr.tokenizer_component_usd > 0

    def test_enterprise_multiplier_scales_linearly(self):
        rows = [_row()]
        full = cost_attribution(rows, "claude-opus-4-6", "claude-opus-4-7")
        half = cost_attribution(
            rows, "claude-opus-4-6", "claude-opus-4-7",
            enterprise_multiplier=0.5,
        )
        # All USD fields must halve when the multiplier halves.
        assert half.baseline_output_cost_usd == pytest.approx(
            full.baseline_output_cost_usd / 2, abs=1e-6
        )
        assert half.delta_usd == pytest.approx(full.delta_usd / 2, abs=1e-6)

    def test_zero_rows_yields_zero_costs(self):
        attr = cost_attribution([], "claude-opus-4-6", "claude-opus-4-7")
        assert attr.baseline_output_cost_usd == 0.0
        assert attr.challenger_output_cost_usd == 0.0
        assert attr.delta_usd == 0.0
