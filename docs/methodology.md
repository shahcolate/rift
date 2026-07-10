# Rift: Statistical Methodology

*Technical methodology report for the Rift drift-detection gate: what a
drift verdict is, which statistical procedures produce it, and —
deliberately foregrounded — where those procedures are approximate,
conservative, or mismatched. Every mechanism is cited to its
implementation. Applies to the codebase as of July 2026; the code is the
normative reference wherever the two disagree.*

---

## 1. Overview and scope

A **drift verdict** is a statement about paired behavior change: given one
eval suite (a fixed list of cases) executed against two model endpoints — a
*baseline* and a *challenger* — Rift tests whether the per-case score
differences are consistent with the null hypothesis of no behavioral change.
The verdict is produced by `comparator.compare_runs`
(`src/rift/comparator.py`), which returns a `DriftResult` carrying the
p-value, a confidence interval on the mean score difference (95% at the
default α, level recorded in `DriftResult.ci_level`), effect
sizes, and cost-normalized metrics. The CLI exit code binds to the verdict:
exit 0 for no significant regression, exit 1 for a statistically significant
regression (`src/rift/cli.py`), which is what makes the verdict usable as a
CI/CD gate.

**Why paired.** Both models answer the *same* cases, so each case serves as
its own control. The test statistic is computed from within-pair differences
`d_i = c_i − b_i`; between-case difficulty variance — usually the dominant
variance component in an eval suite — cancels out of the comparison rather
than inflating the error term. For binary outcomes the paired design admits
an exact small-sample test (McNemar) that an unpaired two-proportion test
does not. The assumptions inherited by all downstream inference are:

1. Cases are independent of one another (violated by near-duplicate cases —
   deduplication is the suite author's responsibility).
2. Both runs used identical inputs and sampling parameters (enforced by the
   runner; the completion cache is keyed on `(model, model_params,
   input_hash)`).
3. Scores are exchangeable across arms under the null.

**What a verdict is not.** A drift verdict is a claim about behavior *on
this suite*, at this sample size, under this scoring function. It is not a
general capability claim, and a non-significant result is not evidence of
stability without the power analysis of §6 (and the small-suite floor
documented there).

## 2. Test selection

Rift selects the significance test from the observed score distribution;
the choice is recorded in `DriftResult.test_used` so a report can never be
ambiguous about which procedure ran.

**Binary scores** (both vectors ⊆ {0.0, 1.0}, detected by
`comparator._is_binary`) use **McNemar's exact test**
(`comparator._mcnemar_exact`). Let `b` = number of pairs where the
challenger regressed (`c_i < b_i`) and `c` = number where it improved.
Under H₀ each discordant pair is a fair coin, so the improvement count is
`Binomial(n_d, 0.5)` with `n_d = b + c`. The reported p-value is the
**two-sided exact binomial** p-value, `scipy.stats.binomtest(c, n_d,
0.5).pvalue` — no chi-squared or continuity-corrected approximation, so the
test is valid at arbitrarily small `n_d`. With zero discordant pairs the
test carries no information and p = 1 is returned. (A paired t-test on
Bernoulli outcomes has well-documented size inflation at small n; McNemar
on the discordant table is the textbook-correct paired procedure.)

**Continuous or graded scores** (semantic similarity, rubric `llm_judge`,
fuzzy match) use a **paired t-test** (`scipy.stats.ttest_rel`) for the
p-value and a **paired percentile bootstrap** for the CI
(`comparator._bootstrap_ci`): 1,000 resamples (default) of the difference
vector with replacement, CI endpoints taken at the `α/2` and `1 − α/2`
percentiles of the resampled means (2.5th/97.5th at the default α = 0.05;
the interval's level follows the comparison's α and is recorded in
`DriftResult.ci_level`). Two properties are stated explicitly:

- The bootstrap is the **percentile** method, *not* BCa. It does not
  correct for bias or skew in the resampling distribution of the mean; for
  markedly skewed score differences at small n the percentile interval can
  under-cover relative to nominal 95%.
- The RNG seed is **fixed at 42** at the call site and intentionally not
  user-tunable: re-running any historical comparison reproduces its CI
  bit-for-bit. Reproducibility is bought at the price of one arbitrary but
  disclosed seed.

Degenerate branches exist for completeness: all differences identical and
non-zero yields `test_used = "deterministic"` (p = 0); all differences zero
yields `"no_variation"` (p = 1); an empty or single-case pairing yields
`"insufficient_data"` (never significant).

## 3. Known caveats (internal statistical audit findings)

These limitations were identified in an internal statistical audit of the
gate. They are documented here as standing caveats; none is hidden by the
tooling, and the first two are conservative in the direction that matters
for a regression gate.

**(a) The binary p-value and the binary CI are different procedures and
can disagree.** For binary data the headline p-value is the exact McNemar
test — *conditional* on the discordant pairs (concordant pairs contribute
no information to the test statistic). The CI shown next to it is the
*unconditional* paired bootstrap of the risk difference
(`comparator._bootstrap_ci` over `d_i = c_i − b_i`), which resamples **all**
pairs including concordant ones. These two procedures answer subtly
different questions about the same data, and near the significance boundary
they can disagree: the CI may exclude zero while McNemar's p ≥ α, or
vice versa. **The exact test governs the verdict** — `significant` and the
exit code derive from `p_value < α`, never from the CI. Reports should be
read accordingly: the CI is an unconditional interval estimate of the
marginal risk difference, not an inversion of the test that produced the
p-value.

**(b) Directional gates run at an effective level of about α/2.** The
default regression gate (`drift.significant and drift.delta < 0`,
`src/rift/cli.py`) and directional pre-registrations
(`direction: regression` in `preregistration.evaluate`) are one-sided
*decisions* evaluated with the **two-sided** p-value. A pre-registration
declaring a one-sided α = 0.05 therefore operates at an effective
false-positive level of roughly α/2 ≈ 0.025 for the declared direction:
under the null, two-sided rejections split evenly between regressions and
improvements, and only the regression half trips the gate. This is
*conservative* (the gate fires less often than the declared α suggests, at
some cost in power), it is not silently corrected, and it is exactly what
`rift selftest` reports empirically — the false-*regression* rate on an
unchanged model sits near α/2, not α (§7). Users who want a true one-sided
test at α should be aware the current implementation does not provide one.

**(c) `rift selftest` calibrates whichever test path the trial-split
produces — match `--trials` to your production gate.** The self-test (§7)
splits each case's `k` trials into two pseudo-arms of `⌊k/2⌋` trials each
and scores each arm by its mean (`selftest.self_test`). With `--trials 2`
or `3`, each pseudo-arm holds a **single** trial, arm scores stay binary,
and the calibration exercises the **McNemar path** — the correct null for
a production gate that runs `trials=1` exact-match comparisons. With
higher trial counts (`--trials ≥ 4`, including odd counts, which discard
one trial per repetition), arm scores are means over ≥ 2 trials, take
fractional values, and the calibration exercises the **continuous
(paired-t + bootstrap) path** instead. The two paths do not necessarily
have the same null behavior at small n. A published selftest number is a
calibration of the path it actually ran; choose `--trials` so that path
matches the gate you deploy.

## 4. Multiplicity

**Benjamini–Hochberg FDR control.** Any report containing many tests —
per-subgroup tables, the N×N `rift matrix`, an Observatory panel pass —
adjusts p-values with `comparator.benjamini_hochberg`. The implementation
computes `q_(i) = min_{j ≥ i} ( p_(j) · m / j )` over the sorted p-values,
i.e. the raw BH-adjusted values with **monotonicity enforced from the
largest p downward** (`np.minimum.accumulate` on the reversed array),
clipped to [0, 1]. This matches R's `p.adjust(method = "BH")`, and
`rejected[i]` is `q_i ≤ α`. Application points:

- Subgroup tables show raw `p` and adjusted `q` side by side
  (`src/rift/reporter.py`).
- `rift matrix` corrects across **all off-diagonal pairwise cells** before
  coloring anything significant: a 4-model matrix runs 12 tests, and
  without correction the expected count of false positives at α = 0.05
  under a global null is 0.6 — i.e. "something significant somewhere" is
  the default outcome, not a finding.
- The Observatory pools BH across every paired test in a whole panel pass
  (`src/rift/observatory.py`); a `score_drift` event requires BH-adjusted
  significance, so the drift feed cannot cherry-pick.

BH controls the false discovery rate, not the family-wise error rate, and
its standard validity conditions assume independent or positively dependent
(PRDS) p-values. Matrix cells share underlying runs and are therefore
dependent; positive dependence is plausible here but not proven, and this
is acknowledged rather than assumed away.

**Pre-registration as the forking-paths defense.** Correction handles the
tests you ran; it does not handle the tests you *chose to headline after
looking*. `rift compare --preregister spec.yaml`
(`src/rift/preregistration.py`) pins, before the run, **one** primary
endpoint (`accuracy` or `cost_per_correct`), a direction, an α, and a
minimum sample size. The headline and the exit code bind to that endpoint
(`preregistration.evaluate`); every other number in the report is labeled
exploratory (hypothesis-generating, not confirmatory). Protocol violations
— fewer cases than `min_cases`, a different suite or model pair than
declared — are recorded in the output rather than silently tolerated. For
the `cost_per_correct` primary there is no p-value; significance is defined
as the paired-bootstrap CI excluding zero (§9). Designating one test in
advance is a cleaner multiplicity defense than correcting twenty
after the fact.

## 5. Effect sizes

Every `DriftResult` carries a standardized effect size; raw deltas confound
with baseline level and within-pair variance and should not be compared
across suites.

**Binary — two effect sizes, deliberately.**

- **Cohen's h on the marginal proportions**
  (`comparator._cohens_h`): `h = 2·(arcsin √p_c − arcsin √p_b)`. This is
  the conventional two-proportion effect size, but it was defined for
  *independent* proportions: applied to paired marginals it ignores the
  paired structure entirely — h is exactly 0 whenever the marginal means
  tie, even if half the cases flipped in each direction.
- **Cohen's g on the discordant cells**
  (`comparator._cohens_g_paired`): `g = P − 0.5` where
  `P = n_improve / n_d`, equivalently `(n_improve − n_regress)/(2·n_d)`,
  ranging over [−0.5, 0.5]. This is the canonical paired effect size for
  the sign/McNemar setting, with its own magnitude thresholds
  (|g| < 0.05 negligible, < 0.15 small, < 0.25 medium, ≥ 0.25 large —
  defined on the ±0.5 scale, hence the `2·n_d` divisor). It is `None`
  when every pair is concordant (the test is uninformative).

Both are reported side by side (`DriftResult.effect_size` with
`effect_size_kind = "cohens_h_marginal"`, plus
`DriftResult.cohens_g_paired`) because they measure different things and
**can carry different verdicts on the same data**: a modest h can coexist
with a strongly one-sided discordant split (few flips, all regressions),
and a non-negligible h can coexist with nearly balanced discordants (large
marginal move built on offsetting churn). A reviewer should read h as "how
far the marginal accuracy moved" and g as "how one-sided the flips were" —
the latter is what McNemar actually tests.

**Continuous — Hedges' g** (`comparator._hedges_g`): the paired
standardized mean difference `d = mean(d_i) / sd(d_i)` (SD of the paired
differences, ddof = 1 — the repeated-measures denominator matching the
t-statistic), multiplied by the small-sample correction
`J = 1 − 3/(4·df − 1)` with `df = n − 1`.

Magnitude bucketing (`comparator._effect_magnitude`) uses Cohen's
conventional |0.2| / |0.5| / |0.8| cutoffs for h and Hedges' g. These are
conventions, not laws; they are labels for orientation, not substitutes for
the CI.

## 6. Power and minimum detectable effect

`comparator.power_analysis` answers "we did not see drift — could we
have?". All formulas use the two-sided critical value
`z_{α/2} = Φ⁻¹(1 − α/2)` and `z_β = Φ⁻¹(power)`.

**Binary (McNemar parameterization).** Power for a paired binary test
depends on the **discordant** pairs, not the marginals: two comparisons
with identical marginal accuracies but different pair-agreement have
different power. Rift therefore uses the McNemar normal approximation with
the observed discordant rate `p_d = n_d / n` as the nuisance parameter
(the standard G*Power-style McNemar parameterization):

- observed effect: the marginal risk difference
  `δ = (n_improve − n_regress)/n` (same units as the headline delta);
- noncentrality: the McNemar z-statistic
  `ncp = |n_improve − n_regress| / √n_d`;
- observed power: `Φ(ncp − z_{α/2}) + Φ(−ncp − z_{α/2})`;
- **minimum detectable effect** at the requested power:
  `MDE = (z_{α/2} + z_β) · √(p_d / n)` on the risk-difference scale;
- required n for a target effect δ*:
  `n = ⌈ p_d · ((z_{α/2} + z_β)/δ*)² ⌉`.

With zero observed discordant pairs, `p_d` cannot be estimated and the MDE
is reported as infinite rather than fabricated.

**Continuous.** The paired SMD (Hedges' g) with `ncp = |g|·√n`,
`MDE_SMD = (z_{α/2} + z_β)/√n`. The normal approximation is accurate for
n ≳ 20 and conservative below that.

**The small-suite hard floor.** The exact McNemar p-value has a
combinatorial minimum: with `n_d` discordant pairs all flipping in one
direction, the smallest attainable two-sided p is `2 · 0.5^{n_d}`.
Consequently at least **6 discordant pairs** are required for p < 0.05 to
be *reachable at all* — a **5-case suite can never produce p < 0.05 under
McNemar, even under total collapse** (all five cases regress:
p = 2·0.5⁵ = 0.0625). Suites of n ≤ 10 are severely power-limited even in
the best case, since `n_d ≤ n`. Practical guidance: treat ~30 cases as a
floor for detecting ~15–20 pp marginal swings, 100+ for single-digit
swings, and always read a green gate on a small suite together with its
reported MDE. A non-significant result on an underpowered suite is an
absence of evidence, not evidence of absence — this is precisely what
`min_cases` in a pre-registration (§4) exists to guard.

## 7. Null calibration: `rift selftest`

The most informative number a gate can publish about itself is its error
rate under the null. `selftest.self_test` (`src/rift/selftest.py`)
estimates it empirically, using the *production* comparison code path
rather than a theoretical calculation:

1. Run the suite once with `trials ≥ 2` per case (replication, §8).
2. For each of `reps` repetitions (default 500, seed fixed at 42), randomly
   partition every case's trials into two pseudo-arms of `⌊k/2⌋` trials and
   take each arm's mean as that case's score — two pseudo-runs of the
   *same* model on the *same* suite.
3. Feed the pseudo-runs through the exact `comparator.compare_runs` the
   gate uses (`bootstrap_n = 0`, since only `significant` and `delta` are
   read).
4. Report, across repetitions:
   - `false_positive_rate` — fraction flagged significant in *either*
     direction; should sit near α for a calibrated two-sided test;
   - `false_regression_rate` — fraction flagged as a significant
     **regression** (`delta < 0`), i.e. the exact exit-1 condition; per
     §3(b) this sits near **α/2** for the current gate;
   - the empirical noise band: mean, 95th-percentile, and max |Δ| across
     repetitions — the magnitude of drift that re-running an unchanged
     model produces by chance.

Interpretation: a green gate is only meaningful if this number is low, and
a red gate on a delta inside the selftest noise band deserves suspicion.
Observatory dashboards cite the selftest false-regression rate next to
every alarm for exactly this reason (`src/rift/observatory_site.py`).
Caveat §3(c) applies: the calibration covers the test path the trial split
produced, so match `--trials` to the production gate.

## 8. Replication and the noise floor

LLM inference is not deterministic even at temperature 0 (MoE routing,
batch-dependent kernels, non-associative float reductions). A single-trial
paired test implicitly assumes zero generation noise; `--trials k`
re-samples each case k times so that assumption can be checked instead of
made. `comparator.variance_components` decomposes replicated scores:

- `within_case_var` (σ²_w) — mean of the per-case sample variances
  (ddof = 1): pure run-to-run generation noise;
- `between_case_var` (σ²_b) — variance attributable to stable differences
  between cases, estimated with the ANOVA ICC(1) estimator with the
  sampling-noise correction subtracted:
  `σ̂²_b = max(0, var(case_means) − σ²_w / k̄)`, where k̄ is the mean
  trial count — the naive variance of case means overstates σ²_b because
  each case mean still carries `σ²_w/k̄` of sampling noise;
- `icc = σ²_b / (σ²_b + σ²_w)` — the fraction of total variance that is
  reproducible signal. ICC near 1 means scores are stable; ICC near 0
  means the metric is mostly resampling noise and any single-run drift
  verdict on it is suspect;
- `noise_floor = √(σ²_w / (n·k))` — the standard error of the run-level
  mean score attributable purely to generation noise.

**The 2× rule.** The reporter (`src/rift/reporter.py`) compares the
observed drift delta against this floor and warns when
`|Δ| ≲ 2 × noise_floor`: a delta within roughly two noise floors is inside
the band that re-running *unchanged* models produces, and **may not survive
a re-run**. This is a heuristic screen, not a test — the formal analogue is
the selftest noise band of §7 — but it catches the most common
over-interpretation: headlining a 3 pp swing on a metric whose run-to-run
wobble is 2 pp.

## 9. Cost methodology

**$/correct with a CI.** Every case result carries `input_tokens`,
`output_tokens`, and `cost_usd` (priced from `src/rift/pricing.py`,
optionally scaled by `--enterprise-multiplier`). Cost-per-correct is
`total run cost / number of fully-correct cases` (score ≥ 0.999). The delta
`challenger − baseline` ships with a CI (95% at the default α) from
`comparator._bootstrap_cost_per_correct_delta_ci`: a **paired percentile
bootstrap on `(score, cost)` tuples** (seed = 42) — pairs are resampled
jointly so the score–cost dependence within a case is preserved — with
per-correct dollars recomputed on each resample.

**Undefined-CI honesty.** Per-correct cost is infinite when a run has zero
correct cases, and the bootstrap delta is meaningless when correctness is
that fragile. The CI is declared **undefined** (rather than reported
wildly wide) when fewer than 10% of bootstrap resamples yield ≥ 1 correct
case in *both* arms; `DriftResult.cost_delta_ci_defined` is set `False`
and renderers are required to omit the line. A pre-registered
`cost_per_correct` primary is judged non-significant in this state (§4).

**List price is one cell of a matrix.** `pricing.py` is standard-mode list
price only. Providers sell the *same* model at other prices — batch
discounts, fast-mode premiums, cache-read pricing, regional multipliers —
and these can change which model is "the expensive one". Any published cost
comparison must state **which cell of the serving-configuration matrix**
its dollar figures assume, and a headline cost multiple must be situated
against the adjacent configurations a buyer could choose instead. The same
disclosure applies to model *configuration*: thinking/effort defaults
differ across models (omitting the setting means off on some, always-on on
others), so a cost or quality comparison must state what each side actually
ran with. See `benchmarks/fable5_vs_opus47/analysis.md` for the disclosure
template.

## 10. Judge validation

Judged metrics inherit the judge's error. The faithfulness articulation
verdict ("the model was swayed but its reasoning never credited the cue")
is decided by an LLM judge, so `rift validate-judge`
(`src/rift/judge_validation.py`) scores that judge against a committed,
hand-labeled gold set (`judge_validation.GOLD_ARTICULATION`): 14 realistic
`(cue, reasoning, answer, target)` tuples, deliberately **balanced** 7/7
between acknowledged and unacknowledged so that chance-corrected agreement
is meaningful. Agreement is reported as **Cohen's kappa**
(`comparator.cohens_kappa`): `κ = (p_o − p_e)/(1 − p_e)` with the
convention that a degenerate expected agreement (a constant rater) yields
κ = 1 iff observed agreement is perfect, else 0. Kappa, not raw accuracy,
is the headline because accuracy flatters a judge on any imbalanced set.
Buckets follow Landis & Koch (1977); the full confusion matrix and every
disagreement are printed.

**Rule:** any published faithfulness (or other judged) number must cite the
judge's kappa on this gold set ("judge validated at κ = X on n = Y").
**Limitations:** the gold set is small (n = 14) and synthetic; κ on it is a
sanity bound on the judge's competence at the articulation task, not a
guarantee of transfer to arbitrary domains, and the binomial uncertainty on
a 14-item κ is substantial.

## 11. Fingerprints and silent-swap detection

A completion cache keyed on the request alone has an integrity hole: if the
provider swaps the weights behind a model alias, cached results silently
describe a model that no longer exists. Rift closes this by capturing the
**server-reported model version** on every completion
(`Completion.provider_fingerprint`, `src/rift/providers/__init__.py`):
OpenAI's `system_fingerprint` (falling back to the echoed `model`), Gemini's
`modelVersion`, the resolved dated `model` string Anthropic echoes back, and
for RiftLM the checkpoint's sha256 digest. Fingerprints are persisted
through the cache and stamped into run metadata (`src/rift/runner.py`).

Downstream detections:

- **Alias collision** — both sides of a comparison resolve to the *same*
  fingerprint: the "comparison" is a self-test, flagged in the report
  (`src/rift/reporter.py`).
- **Mid-run rollout** — more than one fingerprint observed within a single
  run (`metadata["fingerprint_rollout"]`): the run's scores straddle a
  server-side change and are not internally comparable.
- **Observatory events** (`src/rift/observatory.py`): a fingerprint change
  between passes is classified as `fingerprint_change` (scores also moved,
  or no comparison ran) or `silent_swap` (fingerprint changed, paired tests
  ran, and no BH-significant score drift was found). A `silent_swap` is
  only emitted when score comparisons actually executed for that endpoint —
  "the scores held" is not asserted on an endpoint that was never tested.

**Limitation, stated plainly:** a `silent_swap` event asserts only that
**no statistically significant change was detected** on the panel — on a
small panel this may reflect limited power (§6), not genuine behavioral
stability. It is a claim about the absence of detected drift at the
panel's MDE, never a certification that the new snapshot behaves
identically.

## 12. Provenance rules

Rift's outputs are designed to be published; the following disclosure rules
are enforced mechanically wherever possible.

- **Synthetic vs. live labeling.** Recorded/synthetic benchmark replays
  (`--mode record` in `benchmarks/run_context_rot.py`) emit a provenance
  warning at the top of every regenerated report. The repository policy is
  explicit: the live capture (`benchmarks/opus47_live.md`) is
  authoritative; the synthetic replay calibrated to it
  (`benchmarks/context_rot_opus47.md`) must never be quoted without
  flagging its provenance.
- **Custom prompt and scorer disclosure.** Suites may override probe
  prompts (`prompts:` / `cues:`, validated by `src/rift/prompts.py`) and
  supply custom scorers (`scoring: custom`). The runner stamps
  `metadata["custom_prompts"]` (the overridden registry keys) and
  `metadata["custom_scorer"]` (the `target:callable` spec) into every run
  (`src/rift/runner.py`), so a published report discloses any non-default
  measurement instrument. A comparison run under a custom judge rubric is
  not comparable to one under the default rubric; the stamp makes that
  checkable.
- **Discovered-suite selection bias.** Suites built by `rift discover`
  carry full provenance in their `description` — proposer model,
  target/achieved power, per-stage counts — including the explicit caveat
  that cases were *selected on divergence*: the achieved-power figure
  measures the suite's sensitivity for this model pair, not an unbiased
  population estimate.
- **Reproducibility constants.** Bootstrap and selftest seeds are fixed
  (42) and not user-tunable; historical reports re-run to the same numbers.
- **`--strip-io`** empties per-case input/output text in saved JSONs for
  proprietary suites. It is a publishing safety, not a privacy primitive:
  `tags` and `expected` fields still ship.
- **Pre-registration on the record.** The `PreregOutcome` — including any
  protocol violations — is serialized into the saved comparison
  (`src/rift/preregistration.py`), so a claim of "pre-registered primary
  endpoint" is auditable from the artifact itself.

---

*Summary of standing limitations: percentile (non-BCa) bootstrap CIs;
binary p-value/CI procedure mismatch near the boundary (§3a); directional
gates at effective α/2 (§3b); selftest calibrates the path its trial split
produces (§3c); BH dependence conditions assumed, not proven, for matrix
cells (§4); exact-test floor makes n ≤ 10 suites structurally underpowered
(§6); judge gold set is small and synthetic (§10); `silent_swap` is
absence-of-detection, not stability (§11). If a future audit finds more,
they will be added here rather than fixed quietly in prose.*
