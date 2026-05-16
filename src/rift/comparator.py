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
We report three complementary numbers:

1. ``delta`` — raw difference in means. Directly interpretable
   ("dropped 7 points").
2. ``delta_pct`` — relative, for models with very different baseline
   levels.
3. ``cost_normalized_delta`` — change in USD-per-correct-answer. This
   is the number that matters for production budget decisions: two
   models with the same quality are not the same if one costs 3x
   more.

The CI is always on the raw delta. Cost-normalized metrics are point
estimates with a derivation readers can re-run from the underlying
cost and score data stored in the RunResults.

Effect size, multiplicity, and power
------------------------------------
``DriftResult`` also carries an effect-size number on the appropriate
scale for the test that ran: **Cohen's h** for binary score vectors
(proportions) and **Hedges' g** for continuous scores (a
small-sample-corrected standardized mean difference). These are the
numbers reviewers compare across suites — raw deltas confound with
baseline level and within-pair variance.

When many cases or many subgroups are compared in one report, the
naive per-test p-value will declare drift somewhere by chance. The
:func:`benjamini_hochberg` helper turns a list of p-values into a
list of q-values controlling the false discovery rate (BH 1995);
callers can show q-values alongside p-values without changing the
underlying tests.

:func:`power_analysis` answers the question every eval team
eventually asks: "we did not see drift, but could we have?" — given
the observed effect size and N, it reports the minimum detectable
effect at 80% power and the N needed to detect a given target effect.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


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
    # Effect size on the test's natural scale (Cohen's h for binary,
    # Hedges' g for continuous). Always populated; for non-applicable
    # cases (n<2, no variation) it is 0.0 with effect_size_kind="none".
    effect_size: float = 0.0
    effect_size_kind: str = "none"           # "cohens_h" | "hedges_g" | "none"
    effect_size_magnitude: str = "negligible"  # "negligible"|"small"|"medium"|"large"
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


def _cohens_h(p1: float, p2: float) -> float:
    """Cohen's h for two proportions: 2*(arcsin(√p2) − arcsin(√p1)).

    Sign convention matches Rift's ``delta``: positive h means the
    challenger has the higher proportion. Conventional magnitude
    thresholds (Cohen 1988): |h|<0.2 small, <0.5 medium, ≥0.8 large.
    """
    # Clip so √ of −0 / >1 from float roundoff doesn't raise.
    p1 = float(np.clip(p1, 0.0, 1.0))
    p2 = float(np.clip(p2, 0.0, 1.0))
    return float(2.0 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1))))


def _hedges_g(baseline: np.ndarray, challenger: np.ndarray) -> float:
    """Hedges' g — small-sample-corrected paired standardized mean diff.

    For paired data we use the standard deviation of the paired
    differences (not the pooled SD across groups), which is the
    appropriate denominator for a repeated-measures effect size and
    matches how the t-statistic was computed. The Hedges correction
    factor J ≈ 1 − 3/(4·df−1) un-biases g at small N; for n<2 we
    return 0.0.
    """
    diffs = challenger - baseline
    n = diffs.size
    if n < 2:
        return 0.0
    sd = float(np.std(diffs, ddof=1))
    if sd <= 1e-12:
        return 0.0
    d = float(diffs.mean()) / sd
    df = n - 1
    j = 1.0 - 3.0 / (4.0 * df - 1.0) if df > 0 else 1.0
    return d * j


def _effect_magnitude(value: float, kind: str) -> str:
    """Bucket an effect-size value into negligible/small/medium/large.

    Uses Cohen's conventional thresholds. Both Cohen's h and the
    standardized mean difference share the same |.2|, |.5|, |.8|
    cutoffs, so one table covers both.
    """
    if kind == "none":
        return "negligible"
    a = abs(value)
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05
                       ) -> tuple[list[float], list[bool]]:
    """Benjamini–Hochberg FDR control.

    Returns ``(q_values, rejected)`` where ``q_values[i]`` is the
    smallest FDR at which test ``i`` would be rejected and
    ``rejected[i]`` is True iff ``q_values[i] ≤ alpha``. Order is
    preserved (same as ``p_values``).

    Use this when a single report contains many tests (per-subgroup,
    per-suite, per-axis) — the naive per-test p-value over-rejects.
    """
    m = len(p_values)
    if m == 0:
        return [], []
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    # Raw BH-adjusted: p_(k) * m / k, then enforce monotonicity from
    # the top so q-values are non-decreasing in p.
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    q_sorted = adj
    q = np.empty(m, dtype=float)
    q[order] = q_sorted
    return q.tolist(), [bool(qi <= alpha) for qi in q.tolist()]


def power_analysis(
    baseline_scores: list[float],
    challenger_scores: list[float],
    alpha: float = 0.05,
    power: float = 0.8,
    target_effect: float | None = None,
) -> dict:
    """Post-hoc power and minimum-detectable-effect for a paired comparison.

    Returns a dict with:

    * ``observed_effect`` — Cohen's h (binary) or paired-diff
      standardized mean difference (continuous).
    * ``observed_power`` — power to detect that effect at this N and α.
    * ``min_detectable_effect`` — smallest effect detectable at the
      requested ``power``, on the same scale as ``observed_effect``.
    * ``n_for_target`` — N needed to detect ``target_effect`` at
      ``power``. ``None`` if ``target_effect`` is not given.

    This is the "we did not see drift, but could we have?" answer.
    For paired binary data we use the standard normal approximation
    on Cohen's h (Cohen 1988 §6); for continuous data we use the
    paired-difference SMD with a normal approximation, which is
    accurate for n≳20 and conservative below that.
    """
    b = np.asarray(baseline_scores, dtype=float)
    c = np.asarray(challenger_scores, dtype=float)
    n = b.size
    if n < 2:
        return {
            "observed_effect": 0.0,
            "observed_effect_kind": "none",
            "observed_power": 0.0,
            "min_detectable_effect": float("inf"),
            "n_for_target": None,
        }

    z_alpha = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z_power = float(stats.norm.ppf(power))

    if _is_binary(b, c):
        eff = _cohens_h(float(b.mean()), float(c.mean()))
        kind = "cohens_h"
        # Observed power: Pr(|Z| > z_α/2 | true effect = eff, n)
        # Test statistic ≈ h*√n under H1.
        ncp = abs(eff) * np.sqrt(n)
    else:
        eff = _hedges_g(b, c)
        kind = "smd"
        ncp = abs(eff) * np.sqrt(n)

    observed_power = float(
        stats.norm.cdf(ncp - z_alpha) + stats.norm.cdf(-ncp - z_alpha)
    )
    # Minimum detectable effect at requested power: solve ncp = z_α + z_β
    mde = float((z_alpha + z_power) / np.sqrt(n))
    n_for_target: int | None = None
    if target_effect is not None and target_effect > 0:
        n_for_target = int(np.ceil(((z_alpha + z_power) / target_effect) ** 2))

    return {
        "observed_effect": round(eff, 4),
        "observed_effect_kind": kind,
        "observed_power": round(observed_power, 4),
        "min_detectable_effect": round(mde, 4),
        "n_for_target": n_for_target,
    }


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
) -> DriftResult:
    """Compare two paired score vectors.

    Returns a :class:`DriftResult` with the p-value, 95% CI on the
    mean difference, the list of regressed/improved case indices, and
    — when cost vectors are supplied — cost-normalized metrics.

    ``alpha`` controls only the ``significant`` flag; p-value is
    always reported unmodified so callers can apply their own
    threshold.
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

    # --- Effect size on the test's natural scale ---
    if n < 2:
        effect_size = 0.0
        effect_size_kind = "none"
    elif test_used == "mcnemar_exact":
        effect_size = _cohens_h(baseline_mean, challenger_mean)
        effect_size_kind = "cohens_h"
    elif test_used == "paired_t+bootstrap":
        effect_size = _hedges_g(b, c)
        effect_size_kind = "hedges_g"
    else:
        # deterministic / no_variation — effect size is not defined on
        # a t- or h-scale; surface 0.0 rather than NaN.
        effect_size = 0.0
        effect_size_kind = "none"
    effect_size_magnitude = _effect_magnitude(effect_size, effect_size_kind)

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
        n_b_correct = int(np.sum(b >= 0.999))
        n_c_correct = int(np.sum(c >= 0.999))
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
        effect_size=round(effect_size, 4),
        effect_size_kind=effect_size_kind,
        effect_size_magnitude=effect_size_magnitude,
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
) -> dict[str, DriftResult]:
    """Partition cases by a tag prefix and compare each subgroup.

    Example: to split a context-rot run by distractor level, pass
    ``subgroup_prefix="distractor:"``. Only cases tagged with that
    prefix contribute; untagged cases are ignored.
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
        )
    return out
