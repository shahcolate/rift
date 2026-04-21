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


class TestSuccessRate:
    def test_binary_success_rate_matches_mean(self):
        b = [1.0, 1.0, 0.0, 1.0]
        c = [1.0, 0.0, 0.0, 1.0]
        r = compare_runs(b, c, "A", "B", "suite")
        # For binary scores at the default 0.999 threshold, success
        # rate is just the mean of the 1.0s.
        assert r.baseline_success_rate == 0.75
        assert r.challenger_success_rate == 0.5
        assert r.success_rate_delta == -0.25
        assert r.success_threshold == 0.999

    def test_graded_success_rate_uses_threshold(self):
        # Mean says "roughly flat"; success rate at 0.8 says
        # "challenger completes fewer cases fully."
        b = [0.95, 0.90, 0.85, 0.80, 0.75]  # mean 0.85, 4/5 ≥0.8
        c = [0.95, 0.70, 0.90, 0.70, 0.75]  # mean 0.80, 2/5 ≥0.8
        r = compare_runs(b, c, "A", "B", "suite", success_threshold=0.8)
        assert r.baseline_success_rate == 0.8
        assert r.challenger_success_rate == 0.4
        assert r.success_rate_delta == -0.4
        assert r.success_threshold == 0.8

    def test_threshold_default_matches_legacy_cost_per_correct(self):
        # Pre-feature behavior used a hard-coded 0.999 threshold for
        # the cost-per-correct denominator. New `success_threshold`
        # defaults to 0.999 so unchanged callers keep the same
        # cost numbers.
        b = [1.0, 1.0, 1.0, 1.0]
        c = [1.0, 1.0, 1.0, 1.0]
        r = compare_runs(
            b, c, "A", "B", "suite",
            baseline_costs=[0.10] * 4,
            challenger_costs=[0.20] * 4,
        )
        assert r.baseline_cost_per_correct == 0.1
        assert r.challenger_cost_per_correct == 0.2

    def test_subgroup_surfaces_long_end_success_gap(self):
        # Peter's scenario: challenger matches baseline on short
        # workloads but pulls ahead on long ones. The mean barely
        # moves; success rate at the long end tells the real story.
        b = [1.0, 1.0, 0.0, 0.0]                     # 50% overall
        c = [1.0, 1.0, 1.0, 1.0]                     # 100% overall
        tags = [["len:short"], ["len:short"],
                ["len:long"], ["len:long"]]
        groups = compare_by_subgroup(
            b, c, tags, "len:", "A", "B", "suite",
        )
        assert groups["len:short"].baseline_success_rate == 1.0
        assert groups["len:short"].challenger_success_rate == 1.0
        assert groups["len:long"].baseline_success_rate == 0.0
        assert groups["len:long"].challenger_success_rate == 1.0
        assert groups["len:long"].success_rate_delta == 1.0

    def test_backward_compat_default_fields_present(self):
        # Legacy callers that don't pass success_threshold still get
        # the new fields populated with the sensible default.
        r = compare_runs([1.0, 0.0], [1.0, 1.0], "A", "B", "suite")
        assert hasattr(r, "success_threshold")
        assert hasattr(r, "baseline_success_rate")
        assert hasattr(r, "challenger_success_rate")
        assert r.success_threshold == 0.999
