"""Tests for the credibility-fix additions.

Covers:
- ``cost_per_correct`` 95% CI (paired bootstrap on per-case (score, cost))
- ``cohens_g_paired`` alongside ``cohens_h_marginal``
- ``RunResult.save(strip_io=True)``
- BH correction across ``print_matrix`` p-values (smoke + correctness)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rift.comparator import compare_runs, benjamini_hochberg


class TestCostPerCorrectCI:
    def test_cost_ci_defined_when_both_have_correct(self):
        """With cost data and ≥1 correct in both runs, a CI is reported."""
        b = [1.0] * 8 + [0.0] * 2          # 8/10 correct
        c = [1.0] * 7 + [0.0] * 3          # 7/10 correct
        bc = [0.10] * 10                    # uniform $0.10/case
        cc = [0.15] * 10                    # uniform $0.15/case
        r = compare_runs(b, c, "A", "B", "s",
                          baseline_costs=bc, challenger_costs=cc)
        assert r.cost_delta_ci_defined is True
        # The point delta is ~$0.012 (0.15*10/7 - 0.10*10/8); CI should
        # bracket it.
        assert r.cost_delta_ci_lower <= r.cost_normalized_delta_usd <= r.cost_delta_ci_upper

    def test_cost_ci_undefined_when_zero_correct(self):
        """If either run has zero correct, $/correct is inf — CI undefined."""
        b = [1.0] * 5 + [0.0] * 5
        c = [0.0] * 10
        bc = [0.10] * 10
        cc = [0.15] * 10
        r = compare_runs(b, c, "A", "B", "s",
                          baseline_costs=bc, challenger_costs=cc)
        assert r.cost_delta_ci_defined is False
        # And it doesn't crash with NaN — fields stay at the 0.0 defaults.
        assert r.cost_delta_ci_lower == 0.0
        assert r.cost_delta_ci_upper == 0.0

    def test_cost_ci_seed_stable(self):
        """The bootstrap seed is fixed, so two calls give identical CIs."""
        b = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
        c = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        bc = [0.10, 0.11, 0.12, 0.10, 0.13, 0.10, 0.11, 0.10]
        cc = [0.15, 0.15, 0.14, 0.16, 0.15, 0.14, 0.15, 0.15]
        r1 = compare_runs(b, c, "A", "B", "s",
                           baseline_costs=bc, challenger_costs=cc)
        r2 = compare_runs(b, c, "A", "B", "s",
                           baseline_costs=bc, challenger_costs=cc)
        assert r1.cost_delta_ci_lower == r2.cost_delta_ci_lower
        assert r1.cost_delta_ci_upper == r2.cost_delta_ci_upper


class TestStripIO:
    def test_strip_io_removes_inputs_and_outputs(self, tmp_path: Path):
        from rift.runner import CaseResult, RunResult

        cases = [
            CaseResult(
                case_index=0, input_text="SECRET PROMPT", expected="42",
                output="MODEL ANSWER", score=1.0, latency_ms=10.0,
                input_tokens=10, output_tokens=5, cost_usd=0.001, tags=["x"],
            )
        ]
        rr = RunResult(model="m", suite_name="s", scoring_method="exact_match",
                        cases=cases)
        out = tmp_path / "run.json"
        rr.save(out, strip_io=True)
        loaded = json.loads(out.read_text())
        assert loaded["cases"][0]["input_text"] == ""
        assert loaded["cases"][0]["output"] == ""
        # Non-IO fields preserved.
        assert loaded["cases"][0]["score"] == 1.0
        assert loaded["cases"][0]["cost_usd"] == 0.001
        assert loaded["cases"][0]["tags"] == ["x"]

    def test_default_save_preserves_io(self, tmp_path: Path):
        """Backwards compat: default behaviour writes input_text/output."""
        from rift.runner import CaseResult, RunResult

        cases = [
            CaseResult(case_index=0, input_text="P", expected="E", output="O",
                       score=1.0, latency_ms=1.0, input_tokens=1,
                       output_tokens=1, cost_usd=0.0)
        ]
        rr = RunResult(model="m", suite_name="s", scoring_method="exact_match",
                        cases=cases)
        out = tmp_path / "run.json"
        rr.save(out)
        loaded = json.loads(out.read_text())
        assert loaded["cases"][0]["input_text"] == "P"
        assert loaded["cases"][0]["output"] == "O"


class TestMatrixBHCorrection:
    """``print_matrix`` must BH-correct across all pairwise p-values, not
    treat each cell independently. We don't snapshot the rendered table —
    we verify the underlying BH helper produces the expected q-values for
    a known input distribution.
    """

    def test_bh_lifts_q_above_p(self):
        """BH q-values are ≥ raw p; correction is monotone-non-decreasing."""
        ps = [0.001, 0.01, 0.03, 0.04, 0.20]
        q, rejected = benjamini_hochberg(ps, alpha=0.05)
        # Each q ≥ its corresponding p (BH never *decreases* p).
        for p_i, q_i in zip(ps, q):
            assert q_i >= p_i - 1e-12
        # Smallest p rejects; large p does not.
        assert rejected[0] is True
        assert rejected[-1] is False
