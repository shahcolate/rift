"""Tests for statistical comparison."""

from rift.comparator import compare_runs


class TestCompareRuns:
    def test_no_drift(self):
        """Identical scores should show no significant drift."""
        scores = [0.9, 0.8, 0.85, 0.9, 0.95, 0.88, 0.92, 0.87, 0.9, 0.91]
        result = compare_runs(
            baseline_scores=scores,
            challenger_scores=scores,
            baseline_model="model-a",
            challenger_model="model-b",
            suite_name="test",
        )
        assert not result.significant
        assert result.delta == 0.0
        assert result.p_value == 1.0

    def test_clear_regression(self):
        """Obviously worse scores should be significant."""
        baseline = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
        challenger = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        result = compare_runs(
            baseline_scores=baseline,
            challenger_scores=challenger,
            baseline_model="model-a",
            challenger_model="model-b",
            suite_name="test",
        )
        assert result.significant
        assert result.delta < 0
        assert result.drift_direction == "regression"
        assert result.p_value < 0.05

    def test_clear_improvement(self):
        """Obviously better scores should be significant."""
        baseline = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        challenger = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
        result = compare_runs(
            baseline_scores=baseline,
            challenger_scores=challenger,
            baseline_model="model-a",
            challenger_model="model-b",
            suite_name="test",
        )
        assert result.significant
        assert result.delta > 0
        assert result.drift_direction == "improvement"

    def test_regressed_case_tracking(self):
        """Should correctly identify which cases regressed."""
        baseline = [1.0, 1.0, 0.5, 1.0, 0.5]
        challenger = [1.0, 0.0, 0.5, 0.0, 0.5]
        result = compare_runs(
            baseline_scores=baseline,
            challenger_scores=challenger,
            baseline_model="model-a",
            challenger_model="model-b",
            suite_name="test",
        )
        assert 1 in result.regressed_cases
        assert 3 in result.regressed_cases
        assert len(result.improved_cases) == 0

    def test_confidence_interval(self):
        """CI should contain the observed delta for stable data."""
        baseline = [0.8, 0.85, 0.82, 0.79, 0.83, 0.81, 0.84, 0.80, 0.82, 0.83]
        challenger = [0.7, 0.75, 0.72, 0.69, 0.73, 0.71, 0.74, 0.70, 0.72, 0.73]
        result = compare_runs(
            baseline_scores=baseline,
            challenger_scores=challenger,
            baseline_model="model-a",
            challenger_model="model-b",
            suite_name="test",
        )
        assert result.ci_lower <= result.delta <= result.ci_upper
