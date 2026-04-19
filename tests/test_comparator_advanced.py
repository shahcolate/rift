"""Tests for comparator features added in v0.2 (McNemar, cost, subgroups)."""

from rift.comparator import compare_by_subgroup, compare_runs


class TestTestSelection:
    def test_binary_scores_use_mcnemar(self):
        b = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0]
        c = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        r = compare_runs(b, c, "A", "B", "suite")
        assert r.test_used == "mcnemar_exact"

    def test_continuous_scores_use_ttest(self):
        b = [0.8, 0.82, 0.85, 0.79, 0.81, 0.84]
        c = [0.72, 0.75, 0.73, 0.70, 0.74, 0.76]
        r = compare_runs(b, c, "A", "B", "suite")
        assert r.test_used == "paired_t+bootstrap"


class TestCostNormalized:
    def test_cost_per_correct_improves_when_cheaper(self):
        # Same quality, challenger costs half as much.
        b_scores = [1.0] * 10
        c_scores = [1.0] * 10
        b_costs = [0.10] * 10   # $0.10/case → $1.00 total, $0.10/correct
        c_costs = [0.05] * 10   # $0.05/case → $0.50 total, $0.05/correct
        r = compare_runs(b_scores, c_scores, "A", "B", "suite",
                         baseline_costs=b_costs, challenger_costs=c_costs)
        assert r.baseline_cost_per_correct == 0.1
        assert r.challenger_cost_per_correct == 0.05
        assert r.cost_normalized_delta_usd == -0.05

    def test_cost_infinite_when_no_correct(self):
        b_scores = [0.0] * 5
        c_scores = [1.0] * 5
        b_costs = [0.01] * 5
        c_costs = [0.01] * 5
        r = compare_runs(b_scores, c_scores, "A", "B", "suite",
                         baseline_costs=b_costs, challenger_costs=c_costs)
        assert r.baseline_cost_per_correct == float("inf")
        assert r.challenger_cost_per_correct == 0.01


class TestSubgroup:
    def test_split_by_tag_prefix(self):
        b = [1.0, 1.0, 0.0, 0.0]
        c = [1.0, 0.0, 1.0, 0.0]
        tags = [["distractor:0k"], ["distractor:8k"],
                ["distractor:0k"], ["distractor:8k"]]
        groups = compare_by_subgroup(
            b, c, tags, "distractor:", "A", "B", "suite"
        )
        assert set(groups.keys()) == {"distractor:0k", "distractor:8k"}
        assert groups["distractor:0k"].n_cases == 2
        assert groups["distractor:8k"].n_cases == 2
