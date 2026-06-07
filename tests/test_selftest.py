"""Null-calibration self-test: the gate's false-positive rate on an unchanged model."""

from __future__ import annotations

import numpy as np
import pytest
from click.testing import CliRunner

from rift.cli import main
from rift.providers import Completion
from rift.selftest import self_test


class TestSelfTest:
    def test_requires_two_trials(self):
        with pytest.raises(ValueError, match="at least one case with >=2"):
            self_test([[1.0], [0.0]], "m", "s")

    def test_stable_model_never_false_flags(self):
        # Every case is perfectly reproducible (all trials identical), so any
        # self-vs-self split has zero discordant pairs -> the gate never fires.
        cases = [[1.0] * 6 for _ in range(10)] + [[0.0] * 6 for _ in range(10)]
        st = self_test(cases, "m", "s", reps=200)
        assert st.false_positive_rate == 0.0
        assert st.false_regression_rate == 0.0
        assert st.mean_abs_delta == 0.0

    def test_pure_noise_stays_near_alpha(self):
        # Maximally noisy model: every trial an independent coin flip. Under the
        # null the two-sided false-positive rate must still sit near alpha —
        # that is exactly what "well-calibrated" means.
        rng = np.random.default_rng(0)
        cases = [list(rng.integers(0, 2, size=8).astype(float)) for _ in range(40)]
        st = self_test(cases, "m", "s", alpha=0.05, reps=500)
        assert 0.0 <= st.false_positive_rate <= 0.15
        # And the regression-only rate (the gate's exit-1 condition) is lower.
        assert st.false_regression_rate <= st.false_positive_rate
        assert st.mean_abs_delta > 0.0

    def test_deterministic_under_seed(self):
        cases = [[1.0, 0.0, 1.0, 0.0] for _ in range(15)]
        a = self_test(cases, "m", "s", reps=100, seed=7)
        b = self_test(cases, "m", "s", reps=100, seed=7)
        assert a.false_positive_rate == b.false_positive_rate
        assert a.p95_abs_delta == b.p95_abs_delta

    def test_noise_band_ordering(self):
        rng = np.random.default_rng(1)
        cases = [list(rng.integers(0, 2, size=6).astype(float)) for _ in range(20)]
        st = self_test(cases, "m", "s", reps=300)
        assert st.mean_abs_delta <= st.p95_abs_delta <= st.max_abs_delta

    def test_skips_single_trial_cases(self):
        # Mixed: some cases have only one trial; they're dropped, not fatal.
        cases = [[1.0, 0.0, 1.0], [0.0], [1.0, 1.0, 0.0]]
        st = self_test(cases, "m", "s", reps=50)
        assert st.n_cases == 2
        assert st.n_trials == 3


class _VaryingProvider:
    """Returns one of two outputs, alternating per call, to create within-case
    variance across trials for the self-test to chew on."""

    def __init__(self):
        self.model = "stub"
        self._i = 0

    async def complete(self, prompt, **kwargs):
        out = "4" if self._i % 2 == 0 else "wrong"
        self._i += 1
        return Completion(
            model="stub", input_text=prompt, output_text=out, latency_ms=1.0,
            input_tokens=1, output_tokens=1, raw_response={},
            provider_fingerprint="fp",
        )

    async def close(self):
        pass


class TestSelfTestCLI:
    def test_listed_in_help(self):
        result = CliRunner().invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "selftest" in result.output

    def test_rejects_one_trial(self):
        result = CliRunner().invoke(main, [
            "selftest", "--model", "stub", "--suite", "reasoning", "--trials", "1",
        ])
        assert result.exit_code != 0
        assert ">= 2" in result.output

    def test_end_to_end(self, tmp_path, monkeypatch):
        monkeypatch.setattr("rift.runner._get_provider",
                            lambda cfg: _VaryingProvider())
        suite = tmp_path / "s.yaml"
        suite.write_text(
            "name: s\nscoring: exact_match\ncases:\n"
            + "".join(f"  - input: q{i}\n    expected: '4'\n" for i in range(6))
        )
        result = CliRunner().invoke(main, [
            "selftest", "--model", "stub", "--suite", str(suite),
            "--trials", "4", "--reps", "100",
            "--cache-dir", str(tmp_path / "cache"),
        ])
        assert result.exit_code == 0, result.output
        assert "null calibration" in result.output.lower()
        assert "False-regression rate" in result.output
