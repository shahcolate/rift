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

# NOTE: ``scipy`` is imported lazily inside the functions that need it
# (_mcnemar_exact, power_analysis, compare_runs) rather than at module
# top. Importing scipy.stats costs ~0.5s, and pulling it in eagerly here
# made every CLI invocation — including ``rift demo`` — pay that cost
# before printing anything. See the functions below.


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
    # 95% paired-bootstrap CI on the cost-normalized delta. Populated
    # whenever cost data is supplied AND at least one case is correct
    # in both runs (otherwise both per-correct figures are infinite and
    # the difference is undefined; CI fields stay 0.0 and
    # ``cost_delta_ci_defined`` is False).
    cost_delta_ci_lower: float = 0.0
    cost_delta_ci_upper: float = 0.0
    cost_delta_ci_defined: bool = False
    # Effect size on the test's natural scale.
    # ``cohens_h_marginal``: Cohen's h on the marginal proportions (it does
    # NOT use the paired structure — historical / convenient, not the
    # canonical paired effect size). For paired binary, ``cohens_g_paired``
    # below carries the imbalance among discordant pairs.
    # ``hedges_g``: small-sample-corrected paired standardized mean diff.
    effect_size: float = 0.0
    effect_size_kind: str = "none"           # "cohens_h_marginal" | "hedges_g" | "none"
    effect_size_magnitude: str = "negligible"  # "negligible"|"small"|"medium"|"large"
    # Paired binary only: Cohen's g = P − 0.5 on the discordant cells, where
    # P = b/(b+c) (range [-0.5, 0.5]). Reported alongside Cohen's h_marginal
    # so a reviewer can verify both.
    cohens_g_paired: float | None = None
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
    from scipy import stats  # deferred — see module-top note
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


def _bootstrap_cost_per_correct_delta_ci(
    b_scores: np.ndarray, c_scores: np.ndarray,
    b_costs: np.ndarray, c_costs: np.ndarray,
    bootstrap_n: int, seed: int = 42,
) -> tuple[float, float] | None:
    """Paired bootstrap 95% CI on the cost-per-correct delta.

    Resamples paired ``(b_score_i, b_cost_i, c_score_i, c_cost_i)`` tuples
    with replacement, recomputes per-correct $ on each resample, returns the
    2.5/97.5 percentiles of (challenger_cpc − baseline_cpc). Returns
    ``None`` when fewer than 10% of bootstrap samples yield ≥1 correct in
    BOTH runs (CI undefined; better than reporting a wildly wide interval).
    """
    n = b_scores.size
    if n == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(bootstrap_n, n))
    b_correct_per_sample = (b_scores[idx] >= 0.999).sum(axis=1)
    c_correct_per_sample = (c_scores[idx] >= 0.999).sum(axis=1)
    valid = (b_correct_per_sample > 0) & (c_correct_per_sample > 0)
    if valid.sum() < bootstrap_n * 0.10:
        return None
    b_cost_per_sample = b_costs[idx].sum(axis=1)
    c_cost_per_sample = c_costs[idx].sum(axis=1)
    # Avoid div-by-zero on the invalid mask; compute then filter.
    with np.errstate(divide="ignore", invalid="ignore"):
        b_cpc = b_cost_per_sample / b_correct_per_sample
        c_cpc = c_cost_per_sample / c_correct_per_sample
    deltas = (c_cpc - b_cpc)[valid]
    if deltas.size == 0:
        return None
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def _cohens_h(p1: float, p2: float) -> float:
    """Cohen's h for two proportions: 2*(arcsin(√p2) − arcsin(√p1)).

    Convention: positive h means the challenger has the higher proportion.
    Magnitude thresholds (Cohen 1988): |h|<0.2 small, <0.5 medium, ≥0.8 large.

    **Caveat for paired binary data.** Cohen's h was defined for *independent*
    proportions. When applied to the marginal proportions of a paired binary
    comparison (Rift's default) it characterizes the marginal change but
    ignores the paired structure (e.g. it is zero whenever the marginal means
    tie, even when half the pairs flipped). For paired binary, prefer
    :func:`_cohens_g_paired` on the discordant cells, or interpret h alongside
    the McNemar p-value to recover the paired view.
    """
    # Clip so √ of −0 / >1 from float roundoff doesn't raise.
    p1 = float(np.clip(p1, 0.0, 1.0))
    p2 = float(np.clip(p2, 0.0, 1.0))
    return float(2.0 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1))))


def _cohens_g_paired(baseline: np.ndarray, challenger: np.ndarray) -> float | None:
    """Cohen's g for paired binary data: P − 0.5, where P is the proportion
    of discordant pairs that are improvements (P = n_improve / n_discordant).

    Equivalently ``(n_improve − n_regress) / (2·n_discordant)``. This is the
    canonical Cohen's g (Cohen 1988) for the sign/McNemar setting, ranging in
    [-0.5, 0.5]. Magnitude thresholds (Cohen 1988): |g|<0.05 negligible,
    <0.15 small, <0.25 medium, ≥0.25 large — these are defined on this
    [-0.5, 0.5] scale, so the divisor MUST be ``2·n_disc`` (not ``n_disc``)
    for the thresholds to apply. Returns ``None`` when there are no discordant
    pairs (test is uninformative).
    """
    diff = challenger - baseline
    n_improve = int(np.sum(diff > 0))
    n_regress = int(np.sum(diff < 0))
    n_disc = n_improve + n_regress
    if n_disc == 0:
        return None
    return (n_improve - n_regress) / (2 * n_disc)


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

    * ``observed_effect`` — for binary data the marginal risk difference
      δ = (n_improve − n_regress)/n (= ``challenger_mean − baseline_mean``,
      the quantity McNemar tests); for continuous data the paired-diff
      standardized mean difference (Hedges' g).
    * ``observed_power`` — power to detect that effect at this N and α.
    * ``min_detectable_effect`` — smallest effect detectable at the
      requested ``power``, on the same scale as ``observed_effect``.
    * ``n_for_target`` — N needed to detect ``target_effect`` at
      ``power``. ``None`` if ``target_effect`` is not given (binary: also
      ``None`` when no discordant pairs were observed, so the discordant
      rate cannot be estimated).

    This is the "we did not see drift, but could we have?" answer.

    **Binary (McNemar).** Power for a paired binary test depends on the
    *discordant* pairs, not the marginal proportions: two comparisons with
    identical marginals but different pair-agreement have different power.
    We therefore use the McNemar normal approximation. The noncentrality is
    the McNemar z-statistic ``|n_improve − n_regress| / √n_discordant``, and
    the minimum-detectable / required-N formulas are on the risk-difference
    scale with the observed discordant rate ``p_d = n_discordant/n`` as the
    nuisance parameter (the standard G*Power "McNemar" parameterization).
    This replaces an earlier marginal-Cohen's-h approximation that ignored
    the paired structure McNemar actually uses.

    **Continuous.** We use the paired-difference SMD (Hedges' g) with a
    normal approximation, accurate for n≳20 and conservative below that.
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

    from scipy import stats  # deferred — see module-top note
    z_alpha = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z_power = float(stats.norm.ppf(power))
    n_for_target: int | None = None

    if _is_binary(b, c):
        diff = c - b
        n_improve = int(np.sum(diff > 0))
        n_regress = int(np.sum(diff < 0))
        n_disc = n_improve + n_regress
        # Effect McNemar tests: the marginal risk difference (= compare_runs'
        # ``delta``). Same units as the headline number, so the MDE reads as
        # "the smallest accuracy swing we could reliably detect".
        eff = (n_improve - n_regress) / n
        kind = "mcnemar_risk_diff"
        p_d = n_disc / n  # discordant rate — the nuisance parameter
        # Noncentrality = the McNemar z-statistic at the observed split.
        # Zero when no pairs are discordant (the test has no information).
        ncp = abs(n_improve - n_regress) / np.sqrt(n_disc) if n_disc else 0.0
        observed_power = float(
            stats.norm.cdf(ncp - z_alpha) + stats.norm.cdf(-ncp - z_alpha)
        )
        # MDE / required-N on the risk-difference scale, planning at the
        # observed discordant rate. Undefined without any discordant pairs.
        if p_d > 0.0:
            mde = float((z_alpha + z_power) * np.sqrt(p_d / n))
            if target_effect is not None and target_effect > 0:
                n_for_target = int(
                    np.ceil(p_d * ((z_alpha + z_power) / target_effect) ** 2)
                )
        else:
            mde = float("inf")
    else:
        eff = _hedges_g(b, c)
        kind = "smd"
        ncp = abs(eff) * np.sqrt(n)
        observed_power = float(
            stats.norm.cdf(ncp - z_alpha) + stats.norm.cdf(-ncp - z_alpha)
        )
        # MDE on the SMD scale: solve ncp = z_α + z_β.
        mde = float((z_alpha + z_power) / np.sqrt(n))
        if target_effect is not None and target_effect > 0:
            n_for_target = int(np.ceil(((z_alpha + z_power) / target_effect) ** 2))

    return {
        "observed_effect": round(eff, 4),
        "observed_effect_kind": kind,
        "observed_power": round(observed_power, 4),
        "min_detectable_effect": round(mde, 4),
        "n_for_target": n_for_target,
    }


def variance_components(trial_scores_per_case: list[list[float]]) -> dict:
    """Decompose replicated per-case scores into case vs. run-to-run variance.

    Given one list of per-trial scores per case (from a ``trials>1`` run),
    estimate where the variance lives:

    * ``within_case_var`` — mean of the per-case sample variances. This is the
      **run-to-run (generation) noise**: how much one case's score wobbles when
      you ask the *same* model the *same* prompt again. A single-trial paired
      test assumes this is zero; it is not (MoE routing, batch-dependent
      kernels, non-associative float reduction make even temperature-0
      decoding non-deterministic).
    * ``between_case_var`` — variance of the per-case mean scores: real,
      stable differences between prompts.
    * ``icc`` — intraclass correlation ``between / (between + within)``: the
      fraction of total variance that is signal (stable case differences)
      rather than noise. Near 1 ⇒ scores are reproducible; near 0 ⇒ the metric
      is mostly resampling noise and any single-run drift verdict is suspect.
    * ``noise_floor`` — the standard error of the *run-level mean accuracy*
      attributable purely to generation noise, ``sqrt(mean_within_var / (n*k))``.
      A drift delta smaller than a couple of these is within the noise band of
      re-running an unchanged model.

    Returns zeros (and ``icc=1.0`` — no measurable noise) when there are fewer
    than two trials per case to estimate within-case variance from.
    """
    cases = [xs for xs in trial_scores_per_case if xs]
    n = len(cases)
    if n == 0:
        return {
            "n_cases": 0, "mean_trials": 0.0, "within_case_var": 0.0,
            "between_case_var": 0.0, "icc": 1.0, "mean_within_sd": 0.0,
            "noise_floor": 0.0,
        }
    case_means = np.array([float(np.mean(xs)) for xs in cases])
    # Per-case sample variance (ddof=1); 0 for any single-trial case.
    case_vars = np.array([
        float(np.var(xs, ddof=1)) if len(xs) > 1 else 0.0 for xs in cases
    ])
    total_trials = sum(len(xs) for xs in cases)
    mean_trials = total_trials / n
    within = float(np.mean(case_vars))
    between = float(np.var(case_means, ddof=1)) if n > 1 else 0.0
    denom = between + within
    icc = float(between / denom) if denom > 1e-12 else 1.0
    noise_floor = float(np.sqrt(within / total_trials)) if total_trials else 0.0
    return {
        "n_cases": n,
        "mean_trials": round(mean_trials, 2),
        "within_case_var": round(within, 6),
        "between_case_var": round(between, 6),
        "icc": round(icc, 4),
        "mean_within_sd": round(float(np.sqrt(within)), 4),
        "noise_floor": round(noise_floor, 4),
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
        from scipy import stats  # deferred — see module-top note
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
    cohens_g_paired = None
    if n < 2:
        effect_size = 0.0
        effect_size_kind = "none"
    elif test_used == "mcnemar_exact":
        # Cohen's h on the marginal proportions — historical; ignores paired
        # structure. The paired-binary canonical effect (Cohen's g) is
        # reported alongside so a reviewer can interpret both.
        effect_size = _cohens_h(baseline_mean, challenger_mean)
        effect_size_kind = "cohens_h_marginal"
        cohens_g_paired = _cohens_g_paired(b, c)
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
    cost_delta_ci_lower = 0.0
    cost_delta_ci_upper = 0.0
    cost_delta_ci_defined = False
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
            ci = _bootstrap_cost_per_correct_delta_ci(
                b, c, np.asarray(baseline_costs, dtype=float),
                np.asarray(challenger_costs, dtype=float),
                bootstrap_n=bootstrap_n,
            )
            if ci is not None:
                cost_delta_ci_lower, cost_delta_ci_upper = ci
                cost_delta_ci_defined = True

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
        cost_delta_ci_lower=round(cost_delta_ci_lower, 6),
        cost_delta_ci_upper=round(cost_delta_ci_upper, 6),
        cost_delta_ci_defined=cost_delta_ci_defined,
        effect_size=round(effect_size, 4),
        effect_size_kind=effect_size_kind,
        effect_size_magnitude=effect_size_magnitude,
        cohens_g_paired=(round(cohens_g_paired, 4)
                          if cohens_g_paired is not None else None),
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
