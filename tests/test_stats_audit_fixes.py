"""Regression tests for the statistical-audit fixes.

Each test here pins a behavior that was found wrong (or undefined) in an
adversarial statistics review: single-case "significance", NaN leaks at
n=0, the ICC noise floor, alpha-level CIs, and undefined percent deltas.
"""

from __future__ import annotations

import numpy as np

from rift.comparator import compare_runs, compare_by_subgroup, variance_components


def test_single_case_is_never_significant():
    # One continuous pair used to hit the "deterministic" branch and report
    # p=0.0, significant=True. No paired test is defined at n=1.
    drift = compare_runs([0.7], [0.5], "a", "b", "s")
    assert drift.test_used == "insufficient_data"
    assert drift.p_value == 1.0
    assert not drift.significant


def test_single_binary_case_is_never_significant():
    drift = compare_runs([1.0], [0.0], "a", "b", "s")
    assert drift.test_used == "insufficient_data"
    assert not drift.significant


def test_empty_comparison_returns_explicit_zeros_not_nan():
    drift = compare_runs([], [], "a", "b", "s")
    assert drift.n_cases == 0
    assert drift.test_used == "insufficient_data"
    assert drift.baseline_mean == 0.0
    assert drift.challenger_mean == 0.0
    assert drift.delta == 0.0
    assert drift.delta_pct is None
    assert not drift.significant
    assert not np.isnan(drift.p_value)


def test_deterministic_branch_still_fires_at_n2():
    # Identical non-zero diffs across >=2 cases: the zero-variance limit of
    # the paired t. This stays significant — only n<2 was the bug.
    drift = compare_runs([0.5, 0.5], [0.7, 0.7], "a", "b", "s")
    assert drift.test_used == "deterministic"
    assert drift.p_value == 0.0
    assert drift.significant


def test_subgroup_with_one_case_reports_insufficient_data():
    drift = compare_by_subgroup(
        baseline_scores=[1.0, 1.0, 0.4],
        challenger_scores=[0.0, 1.0, 0.9],
        tags_per_case=[["task:big"], ["task:big"], ["task:solo"]],
        subgroup_prefix="task:",
        baseline_model="a", challenger_model="b", suite_name="s",
    )
    solo = drift["task:solo"]
    assert solo.n_cases == 1
    assert solo.test_used == "insufficient_data"
    assert not solo.significant


def test_delta_pct_undefined_when_baseline_mean_zero():
    drift = compare_runs([0.0, 0.0, 0.0], [1.0, 0.0, 1.0], "a", "b", "s")
    assert drift.delta > 0
    assert drift.delta_pct is None  # not a false "+0.0%"


def test_ci_level_follows_alpha():
    b = [0.1, 0.4, 0.5, 0.9, 0.3, 0.6, 0.2, 0.8]
    c = [0.2, 0.5, 0.4, 0.8, 0.5, 0.7, 0.4, 0.9]
    d95 = compare_runs(b, c, "a", "b", "s", alpha=0.05)
    d99 = compare_runs(b, c, "a", "b", "s", alpha=0.01)
    assert d95.ci_level == 0.95
    assert d99.ci_level == 0.99
    # A 99% interval must be at least as wide as the 95% one on the same
    # bootstrap draws (same fixed seed).
    assert (d99.ci_upper - d99.ci_lower) >= (d95.ci_upper - d95.ci_lower)


def test_cost_ci_level_follows_alpha():
    b = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    c = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    costs_b = [0.01 * (i + 1) for i in range(10)]
    costs_c = [0.02 * (i + 1) for i in range(10)]
    d95 = compare_runs(b, c, "a", "b", "s", alpha=0.05,
                       baseline_costs=costs_b, challenger_costs=costs_c)
    d99 = compare_runs(b, c, "a", "b", "s", alpha=0.01,
                       baseline_costs=costs_b, challenger_costs=costs_c)
    assert d95.cost_delta_ci_defined and d99.cost_delta_ci_defined
    w95 = d95.cost_delta_ci_upper - d95.cost_delta_ci_lower
    w99 = d99.cost_delta_ci_upper - d99.cost_delta_ci_lower
    assert w99 >= w95


def test_icc_pure_noise_reads_near_zero():
    # A metric that is pure resampling noise: every case has the same true
    # score, all observed spread is within-case. The naive ratio
    # between/(between+within) has a k-dependent floor (0.33 at k=2); the
    # ANOVA ICC(1) estimator must read ~0.
    rng = np.random.default_rng(7)
    trials = [list(rng.normal(0.5, 0.1, size=4)) for _ in range(60)]
    vc = variance_components(trials)
    assert vc["icc"] < 0.15, vc


def test_icc_pure_signal_reads_near_one():
    # Zero within-case variance, real between-case differences.
    trials = [[0.1, 0.1, 0.1], [0.9, 0.9, 0.9], [0.5, 0.5, 0.5]]
    vc = variance_components(trials)
    assert vc["icc"] == 1.0
    assert vc["within_case_var"] == 0.0


def test_icc_mixed_signal_and_noise_between_bounds():
    rng = np.random.default_rng(11)
    # True case means spread 0..1 (signal), plus per-trial noise.
    trials = [
        list(np.clip(rng.normal(mu, 0.05, size=5), 0, 1))
        for mu in np.linspace(0.1, 0.9, 40)
    ]
    vc = variance_components(trials)
    assert 0.8 < vc["icc"] <= 1.0


def test_drift_direction_labels():
    reg = compare_runs([1.0] * 12, [0.0] * 12, "a", "b", "s")
    assert reg.significant and reg.drift_direction == "regression"
    imp = compare_runs([0.0] * 12, [1.0] * 12, "a", "b", "s")
    assert imp.significant and imp.drift_direction == "improvement"
    none = compare_runs([1.0, 0.0] * 6, [1.0, 0.0] * 6, "a", "b", "s")
    assert not none.significant and none.drift_direction == "none"
