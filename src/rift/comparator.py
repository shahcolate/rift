"""Statistical comparison of two eval runs.

This module turns two score vectors into a defensible claim. Two
things matter here and nothing else: **the test matches the data**,
and **the effect size is expressed in units a reader can act on**.

Test selection
--------------
Rift runs one of two paired tests depending on the score
distribution:

* **McNemar's exact test** when both vectors are binary
  (``{0.0, 1.0}`` exact-match outcomes). A paired t-test on a
  Bernoulli variable has well-known size inflation; McNemar's test on
  the 2x2 discordant-pairs table is the textbook correct choice and
  is exact at small n. We call ``scipy.stats.binomtest`` on the
  discordant pairs rather than the chi-squared approximation so the
  test remains valid even when the discordant count is <25.
* **Paired bootstrap + paired t-test** otherwise. Continuous or
  graded scores (semantic similarity, rubric-style llm_judge) get the
  t-test for the p-value and a non-parametric bootstrap for the
  confidence interval — the CI is what a reader actually reads, and
  it does not assume normality.

Both paths produce the same :class:`DriftResult` dataclass so callers
need not branch.

Effect size
-----------
We report four complementary numbers:

1. ``delta`` — raw difference in means. Directly interpretable
   ("dropped 7 points").
2. ``delta_pct`` — relative, for models with very different baseline
   levels.
3. ``success_rate`` (per side) and ``success_rate_delta`` — fraction
   of cases at-or-above ``success_threshold``. For graded scores this
   surfaces the "does it actually finish the workload" signal that
   the mean can hide: a model with mean 0.72 may be succeeding on 80%
   of short cases and 20% of long ones. Paired with subgroup
   breakdown, this is the metric that tells you whether a more
   expensive model is earning its keep on the hard end of the
   distribution.
4. ``cost_normalized_delta`` — change in USD-per-correct-answer. This
   is the number that matters for production budget decisions: two
   models with the same quality are not the same if one costs 3x
   more.

The CI is always on the raw delta. Cost-normalized metrics are point
estimates with a derivation readers can re-run from the underlying
cost and score data stored in the RunResults.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


# Default threshold for counting a case as a "success" and for the
# n_correct denominator in cost-per-correct. Set high so binary
# exact-match (scores in {0.0, 1.0}) behaves exactly as it used to:
# only full-credit cases count. Graded-scorer callers typically
# override this to something like 0.8.
DEFAULT_SUCCESS_THRESHOLD = 0.999


@dataclass
class DriftResult:
    """Result of comparing two runs on the same suite."""

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
    test_used: str                       # "mcnemar_exact" | "paired_t+bootstrap"
    regressed_cases: list[int]
    improved_cases: list[int]
    # Cost-normalized metrics (populated when cost data is supplied).
    baseline_cost_usd: float = 0.0
    challenger_cost_usd: float = 0.0
    baseline_cost_per_correct: float = 0.0
    challenger_cost_per_correct: float = 0.0
    cost_normalized_delta_usd: float = 0.0  # challenger - baseline, per correct
    # Success-rate metrics (threshold-based fraction correct).
    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD
    baseline_success_rate: float = 0.0
    challenger_success_rate: float = 0.0
    success_rate_delta: float = 0.0      # challenger - baseline
    # Per-tag subgroup drift (optional).
    subgroups: dict[str, "DriftResult"] = field(default_factory=dict)

    @property
    def drift_direction(self) -> str:
        if not self.significant:
            return "none"
        return "regression" if self.delta < 0 else "improvement"


def _is_binary(xs: np.ndarray, ys: np.ndarray) -> bool:
    """True iff both vectors contain only {0.0, 1.0}."""
    return bool(np.all(np.isin(xs, (0.0, 1.0))) and np.all(np.isin(ys, (0.0, 1.0))))


def _mcnemar_exact(baseline: np.ndarray, challenger: np.ndarray) -> float:
    """Two-sided McNemar exact p-value via the binomial on discordant pairs.

    Discordant pairs are indices where baseline != challenger. Under
    H0 (no effect) each discordant pair is equally likely to be a
    regression or an improvement, so the count of improvements among
    discordants is Binomial(n_disc, 0.5). The two-sided p-value is
    ``binomtest(k, n, 0.5).pvalue``.
    """
    diff = challenger - baseline
    n_regress = int(np.sum(diff < 0))
    n_improve = int(np.sum(diff > 0))
    n_disc = n_regress + n_improve
    if n_disc == 0:
        return 1.0
    return float(stats.binomtest(n_improve, n_disc, p=0.5).pvalue)


def _bootstrap_ci(diffs: np.ndarray, n: int, bootstrap_n: int, seed: int = 42
                  ) -> tuple[float, float]:
    """Paired bootstrap 95% CI on the mean of ``diffs``.

    Seeded so re-running a comparison gives the same CI. The seed is
    intentionally fixed at the call site — do not expose it as a
    user-tunable; reproducibility of historical reports depends on it.
    """
    rng = np.random.default_rng(seed)
    # Vectorized resample: bootstrap_n × n matrix of indices.
    idx = rng.integers(0, n, size=(bootstrap_n, n))
    sample_means = diffs[idx].mean(axis=1)
    return float(np.percentile(sample_means, 2.5)), float(np.percentile(sample_means, 97.5))


def compare_runs(
    baseline_scores: list[float],
    challenger_scores: list[float],
    baseline_model: str,
    challenger_model: str,
    suite_name: str,
    alpha: float = 0.05,
    bootstrap_n: int = 1000,
    baseline_costs: list[float] | None = None,
    challenger_costs: list[float] | None = None,
    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD,
) -> DriftResult:
    """Compare two paired score vectors.

    Returns a :class:`DriftResult` with the p-value, 95% CI on the
    mean difference, the list of regressed/improved case indices, and
    — when cost vectors are supplied — cost-normalized metrics.

    ``alpha`` controls only the ``significant`` flag; p-value is
    always reported unmodified so callers can apply their own
    threshold.

    ``success_threshold`` governs the fraction-correct / success-rate
    metric: a case counts as a success when its score is
    ``>= success_threshold``. The same threshold drives the
    ``n_correct`` denominator in cost-per-correct, so lowering it for
    a graded scorer keeps both metrics consistent.
    """
    assert len(baseline_scores) == len(challenger_scores), \
        "Score lists must be same length"
    n = len(baseline_scores)
    b = np.asarray(baseline_scores, dtype=float)
    c = np.asarray(challenger_scores, dtype=float)

    baseline_mean = float(b.mean())
    challenger_mean = float(c.mean())
    delta = challenger_mean - baseline_mean
    delta_pct = (delta / baseline_mean * 100) if baseline_mean != 0 else 0.0

    diffs = c - b

    # --- Test selection ---
    if _is_binary(b, c):
        p_value = _mcnemar_exact(b, c)
        test_used = "mcnemar_exact"
    elif n >= 2 and float(np.std(diffs)) > 1e-10:
        _, p = stats.ttest_rel(c, b)
        p_value = float(p)
        test_used = "paired_t+bootstrap"
    elif abs(float(diffs.mean())) > 1e-10:
        # All diffs identical and non-zero: deterministic change.
        p_value = 0.0
        test_used = "deterministic"
    else:
        p_value = 1.0
        test_used = "no_variation"

    # --- CI: bootstrap regardless of test used (non-parametric, robust) ---
    if n >= 2 and float(np.std(diffs)) > 1e-10:
        ci_lower, ci_upper = _bootstrap_ci(diffs, n, bootstrap_n)
    else:
        ci_lower = ci_upper = float(diffs.mean()) if n > 0 else 0.0

    significant = p_value < alpha

    regressed = [int(i) for i in range(n) if c[i] < b[i]]
    improved = [int(i) for i in range(n) if c[i] > b[i]]

    # --- Success-rate metrics (fraction of cases at-or-above threshold) ---
    # Computed unconditionally so every report has them; the threshold
    # used is echoed in DriftResult so readers can reproduce.
    n_b_correct = int(np.sum(b >= success_threshold))
    n_c_correct = int(np.sum(c >= success_threshold))
    b_success_rate = n_b_correct / n if n > 0 else 0.0
    c_success_rate = n_c_correct / n if n > 0 else 0.0

    # --- Cost-normalized metrics ---
    total_baseline_cost = 0.0
    total_challenger_cost = 0.0
    baseline_cpc = 0.0
    challenger_cpc = 0.0
    cost_delta = 0.0
    if baseline_costs is not None and challenger_costs is not None:
        assert len(baseline_costs) == len(challenger_costs) == n
        total_baseline_cost = float(sum(baseline_costs))
        total_challenger_cost = float(sum(challenger_costs))
        baseline_cpc = (
            total_baseline_cost / n_b_correct if n_b_correct else float("inf")
        )
        challenger_cpc = (
            total_challenger_cost / n_c_correct if n_c_correct else float("inf")
        )
        if baseline_cpc != float("inf") and challenger_cpc != float("inf"):
            cost_delta = challenger_cpc - baseline_cpc

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
        test_used=test_used,
        regressed_cases=regressed,
        improved_cases=improved,
        baseline_cost_usd=round(total_baseline_cost, 4),
        challenger_cost_usd=round(total_challenger_cost, 4),
        baseline_cost_per_correct=round(baseline_cpc, 6) if baseline_cpc != float("inf") else float("inf"),
        challenger_cost_per_correct=round(challenger_cpc, 6) if challenger_cpc != float("inf") else float("inf"),
        cost_normalized_delta_usd=round(cost_delta, 6),
        success_threshold=success_threshold,
        baseline_success_rate=round(b_success_rate, 4),
        challenger_success_rate=round(c_success_rate, 4),
        success_rate_delta=round(c_success_rate - b_success_rate, 4),
    )


def compare_by_subgroup(
    baseline_scores: list[float],
    challenger_scores: list[float],
    tags_per_case: list[list[str]],
    subgroup_prefix: str,
    baseline_model: str,
    challenger_model: str,
    suite_name: str,
    alpha: float = 0.05,
    baseline_costs: list[float] | None = None,
    challenger_costs: list[float] | None = None,
    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD,
) -> dict[str, DriftResult]:
    """Partition cases by a tag prefix and compare each subgroup.

    Example: to split a context-rot run by distractor level, pass
    ``subgroup_prefix="distractor:"``. Only cases tagged with that
    prefix contribute; untagged cases are ignored.

    Each subgroup gets its own success-rate pair, which is where the
    headline "does model X carry the long end of the distribution"
    signal lives — use ``success_threshold`` to set the bar.
    """
    buckets: dict[str, list[int]] = {}
    for i, tags in enumerate(tags_per_case):
        for t in tags:
            if t.startswith(subgroup_prefix):
                buckets.setdefault(t, []).append(i)
                break

    out: dict[str, DriftResult] = {}
    for tag, idxs in buckets.items():
        out[tag] = compare_runs(
            baseline_scores=[baseline_scores[i] for i in idxs],
            challenger_scores=[challenger_scores[i] for i in idxs],
            baseline_model=baseline_model,
            challenger_model=challenger_model,
            suite_name=f"{suite_name}[{tag}]",
            alpha=alpha,
            baseline_costs=[baseline_costs[i] for i in idxs] if baseline_costs else None,
            challenger_costs=[challenger_costs[i] for i in idxs] if challenger_costs else None,
            success_threshold=success_threshold,
        )
    return out
