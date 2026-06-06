"""Null calibration: how often does the drift gate fire on an unchanged model?

The single most important number a drift detector can publish about itself is
its false-positive rate under the null — comparing a model to *itself* and
asking how often the "significant regression" verdict (the exit-1 condition the
CI gate blocks deploys on) fires anyway. If that rate is well above ``alpha``,
the gate is mis-calibrated for this suite/model and a green run means less than
it looks like.

We get the null for free from a replicated run (:func:`rift.runner.run_suite`
with ``trials>1``). Each repetition randomly partitions every case's trials into
two arms, takes each arm's mean as a pseudo-run, and feeds the two pseudo-runs
through the *exact* :func:`rift.comparator.compare_runs` the gate uses. Across
many repetitions the fraction flagged significant estimates the empirical
false-positive rate, and the spread of the deltas is the run-to-run noise band a
real drift claim has to clear.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .comparator import compare_runs


@dataclass
class SelfTestResult:
    """Empirical null behaviour of the drift gate for one model + suite."""

    model: str
    suite_name: str
    n_cases: int
    n_trials: int
    reps: int
    alpha: float
    # Fraction of repetitions the gate flagged *any* significant drift
    # (two-sided). Should sit near ``alpha`` if the test is well-calibrated.
    false_positive_rate: float
    # Fraction flagged as a significant *regression* (delta < 0) — the exact
    # condition ``compare``/the CI Action exits 1 on. This is the number that
    # answers "how often would this gate block a deploy on an unchanged model?".
    false_regression_rate: float
    # The noise band: distribution of |delta| across repetitions.
    mean_abs_delta: float
    p95_abs_delta: float
    max_abs_delta: float


def self_test(
    trial_scores_per_case: list[list[float]],
    model: str,
    suite_name: str,
    alpha: float = 0.05,
    reps: int = 500,
    seed: int = 42,
) -> SelfTestResult:
    """Estimate the drift gate's false-positive rate from replicated scores.

    ``trial_scores_per_case`` is one list of per-trial scores per case (from a
    ``trials>=2`` run). Each of ``reps`` repetitions randomly splits every
    case's trials into two arms, compares the arm means with the production
    :func:`compare_runs`, and tallies how often the gate fires.

    The seed is fixed so the reported rate is reproducible. Raises
    ``ValueError`` if fewer than two trials are available to split.
    """
    cases = [list(xs) for xs in trial_scores_per_case if len(xs) >= 2]
    if not cases:
        raise ValueError(
            "self_test needs at least one case with >=2 trials; "
            "run with --trials 2 or more."
        )
    n_trials = min(len(xs) for xs in cases)
    rng = np.random.default_rng(seed)

    n_sig = 0
    n_reg = 0
    abs_deltas: list[float] = []
    for _ in range(reps):
        arm_a: list[float] = []
        arm_b: list[float] = []
        for xs in cases:
            perm = rng.permutation(len(xs))
            half = len(xs) // 2
            a_idx, b_idx = perm[:half], perm[half: 2 * half]
            arm_a.append(float(np.mean([xs[i] for i in a_idx])))
            arm_b.append(float(np.mean([xs[i] for i in b_idx])))
        # bootstrap_n=0: we read only .significant/.delta, so skip the CI
        # resample compare_runs would otherwise run on every one of `reps`.
        drift = compare_runs(arm_a, arm_b, model, model, suite_name,
                             alpha=alpha, bootstrap_n=0)
        if drift.significant:
            n_sig += 1
            if drift.delta < 0:
                n_reg += 1
        abs_deltas.append(abs(drift.delta))

    arr = np.asarray(abs_deltas)
    return SelfTestResult(
        model=model,
        suite_name=suite_name,
        n_cases=len(cases),
        n_trials=n_trials,
        reps=reps,
        alpha=alpha,
        false_positive_rate=round(n_sig / reps, 4),
        false_regression_rate=round(n_reg / reps, 4),
        mean_abs_delta=round(float(arr.mean()), 4),
        p95_abs_delta=round(float(np.percentile(arr, 95)), 4),
        max_abs_delta=round(float(arr.max()), 4),
    )
