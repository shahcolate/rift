"""Statistical comparison of two eval runs."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import stats


@dataclass
class DriftResult:
    """Result of comparing two runs."""

    baseline_model: str
    challenger_model: str
    suite_name: str
    n_cases: int
    baseline_mean: float
    challenger_mean: float
    delta: float
    delta_pct: float
    p_value: float
    ci_lower: float
    ci_upper: float
    significant: bool
    regressed_cases: list[int]  # indices of cases that got worse
    improved_cases: list[int]   # indices of cases that got better

    @property
    def drift_direction(self) -> str:
        if not self.significant:
            return "none"
        return "regression" if self.delta < 0 else "improvement"


def compare_runs(
    baseline_scores: list[float],
    challenger_scores: list[float],
    baseline_model: str,
    challenger_model: str,
    suite_name: str,
    alpha: float = 0.05,
    bootstrap_n: int = 1000,
) -> DriftResult:
    """Compare two sets of scores using paired bootstrap + t-test."""
    assert len(baseline_scores) == len(challenger_scores), "Score lists must be same length"

    n = len(baseline_scores)
    b = np.array(baseline_scores)
    c = np.array(challenger_scores)

    baseline_mean = float(np.mean(b))
    challenger_mean = float(np.mean(c))
    delta = challenger_mean - baseline_mean
    delta_pct = (delta / baseline_mean * 100) if baseline_mean != 0 else 0.0

    # Paired t-test
    diffs = c - b
    if n >= 2 and np.std(diffs) > 1e-10:
        t_stat, p_value = stats.ttest_rel(c, b)
        p_value = float(p_value)
    elif abs(float(np.mean(diffs))) > 1e-10:
        # All diffs identical and non-zero: deterministic change
        p_value = 0.0
    else:
        p_value = 1.0

    # Paired bootstrap confidence interval on the difference
    rng = np.random.default_rng(42)
    boot_means = []
    for _ in range(bootstrap_n):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means.append(float(np.mean(sample)))

    boot_means.sort()
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    significant = p_value < alpha

    # Identify regressed and improved cases
    regressed = [i for i in range(n) if c[i] < b[i]]
    improved = [i for i in range(n) if c[i] > b[i]]

    return DriftResult(
        baseline_model=baseline_model,
        challenger_model=challenger_model,
        suite_name=suite_name,
        n_cases=n,
        baseline_mean=round(baseline_mean, 4),
        challenger_mean=round(challenger_mean, 4),
        delta=round(delta, 4),
        delta_pct=round(delta_pct, 2),
        p_value=round(p_value, 6),
        ci_lower=round(ci_lower, 4),
        ci_upper=round(ci_upper, 4),
        significant=significant,
        regressed_cases=regressed,
        improved_cases=improved,
    )
