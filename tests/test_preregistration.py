"""Pre-registration: pinning the primary endpoint before the run."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from rift.cli import main
from rift.comparator import compare_runs
from rift.preregistration import (
    PreregError,
    Preregistration,
    evaluate,
    load_preregistration,
)

EXAMPLE = Path(__file__).parent.parent / "examples" / "preregistration_example.yaml"


class TestLoad:
    def test_loads_example(self):
        p = load_preregistration(EXAMPLE)
        assert p.primary == "accuracy"
        assert p.direction == "regression"
        assert p.min_cases == 50

    def test_missing_file_raises(self):
        with pytest.raises(PreregError, match="not found"):
            load_preregistration("/no/such/prereg.yaml")

    def test_invalid_field_raises(self, tmp_path):
        f = tmp_path / "p.yaml"
        f.write_text("primary: nonsense\n")
        with pytest.raises(PreregError, match="Invalid"):
            load_preregistration(f)

    def test_defaults(self):
        p = Preregistration()
        assert p.primary == "accuracy"
        assert p.alpha == 0.05
        assert p.min_cases == 0


def _regression_drift():
    # 10 clean regressions, 0 improvements -> McNemar p tiny, delta -0.5.
    b = [1.0] * 20
    c = [0.0] * 10 + [1.0] * 10
    return compare_runs(b, c, "claude-opus-4-7", "claude-opus-4-8", "reasoning")


def _flat_drift():
    b = [1.0] * 10 + [0.0] * 10
    return compare_runs(b, list(b), "m-a", "m-b", "reasoning")


class TestEvaluateAccuracy:
    def test_regression_confirmed(self):
        d = _regression_drift()
        o = evaluate(Preregistration(primary="accuracy", direction="regression",
                                     min_cases=10), d, d.n_cases)
        assert o.primary_significant is True
        assert o.adverse_confirmed is True
        assert o.honored is True

    def test_no_drift_not_confirmed(self):
        d = _flat_drift()
        o = evaluate(Preregistration(direction="regression"), d, d.n_cases)
        assert o.adverse_confirmed is False

    def test_regression_not_confirmed_as_improvement(self):
        # A real regression is NOT an improvement; improvement gate stays off.
        d = _regression_drift()
        o = evaluate(Preregistration(direction="improvement"), d, d.n_cases)
        assert o.adverse_confirmed is False

    def test_two_sided_fires_on_regression(self):
        d = _regression_drift()
        o = evaluate(Preregistration(direction="two_sided"), d, d.n_cases)
        assert o.adverse_confirmed is True


class TestViolations:
    def test_min_cases_violation(self):
        d = _regression_drift()
        o = evaluate(Preregistration(min_cases=1000), d, d.n_cases)
        assert o.honored is False
        assert any("underpowered" in v for v in o.violations)
        # Violation qualifies the claim but the statistical gate still fires.
        assert o.adverse_confirmed is True

    def test_suite_mismatch(self):
        d = _regression_drift()
        o = evaluate(Preregistration(suite="extraction"), d, d.n_cases)
        assert any("suite mismatch" in v for v in o.violations)

    def test_model_mismatch(self):
        d = _regression_drift()
        o = evaluate(
            Preregistration(baseline="gpt-4o", challenger="gpt-4"),
            d, d.n_cases, baseline_model="claude-opus-4-7",
            challenger_model="claude-opus-4-8",
        )
        assert any("baseline mismatch" in v for v in o.violations)
        assert any("challenger mismatch" in v for v in o.violations)


class TestEvaluateCost:
    def test_cost_regression_confirmed(self):
        # Same accuracy, challenger costs 2x per call -> higher $/correct.
        b = [1.0] * 10 + [0.0] * 5
        c = [1.0] * 10 + [0.0] * 5
        d = compare_runs(b, c, "m-a", "m-b", "reasoning",
                         baseline_costs=[0.01] * 15, challenger_costs=[0.02] * 15)
        o = evaluate(Preregistration(primary="cost_per_correct",
                                     direction="regression"), d, d.n_cases)
        assert o.primary_significant is True
        assert o.adverse_confirmed is True

    def test_cost_ci_undefined_not_significant(self):
        # No cost data -> CI undefined -> cannot confirm.
        d = _flat_drift()
        o = evaluate(Preregistration(primary="cost_per_correct"), d, d.n_cases)
        assert o.primary_significant is False
        assert o.adverse_confirmed is False


class TestCLI:
    def test_help_lists_preregister(self):
        result = CliRunner().invoke(main, ["compare", "--help"])
        assert result.exit_code == 0
        assert "--preregister" in result.output

    def test_missing_prereg_file_clean_error(self):
        result = CliRunner().invoke(main, [
            "compare", "--baseline", "stub", "--challenger", "stub2",
            "--suite", "reasoning", "--preregister", "/no/such.yaml",
        ])
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_end_to_end_gate_binds_to_prereg(self, tmp_path, monkeypatch):
        # Baseline always right, challenger always wrong -> accuracy regression.
        from rift.providers import Completion

        class _Side:
            def __init__(self, out):
                self.out = out
                self.model = "stub"

            async def complete(self, prompt, **kwargs):
                return Completion(model="stub", input_text=prompt,
                                  output_text=self.out, latency_ms=1.0,
                                  input_tokens=1, output_tokens=1,
                                  raw_response={}, provider_fingerprint="fp")

            async def close(self):
                pass

        def fake_get_provider(cfg):
            return _Side("4" if cfg.model == "stubgood" else "wrong")

        monkeypatch.setattr("rift.runner._get_provider", fake_get_provider)
        suite = tmp_path / "s.yaml"
        suite.write_text(
            "name: reasoning\nscoring: exact_match\ncases:\n"
            + "".join(f"  - input: q{i}\n    expected: '4'\n" for i in range(12))
        )
        prereg = tmp_path / "prereg.yaml"
        prereg.write_text("primary: accuracy\ndirection: regression\n"
                          "alpha: 0.05\nmin_cases: 5\n")
        result = CliRunner().invoke(main, [
            "compare", "--baseline", "stubgood", "--challenger", "stubbad",
            "--suite", str(suite), "--preregister", str(prereg),
            "--cache-dir", str(tmp_path / "cache"), "--no-refusal", "--no-power",
        ])
        assert "PRE-REGISTERED REGRESSION CONFIRMED" in result.output
        assert "EXPLORATORY" in result.output
        assert result.exit_code == 1  # gate fired on the pre-registered endpoint
