# Rift

**You upgraded your model. What broke?**

Rift detects behavioral regressions between LLM model versions. Run structured eval suites against any two models, get statistically rigorous drift reports.

No vibes. No "it feels dumber." Just p-values and confidence intervals.

## Quick Start

```bash
pip install rift-eval

# Compare two models (with short aliases — opus-4-7, sonnet-4-6, etc.)
rift compare --baseline opus-4-6 --challenger opus-4-7 --suite reasoning

# Stress-test reasoning under distractor context (0k/2k/8k/32k)
rift compare --baseline opus-4-6 --challenger opus-4-7 \
    --suite context_rot_reasoning --context-rot --subgroup distractor:

# Compare 3+ models at once — prints an NxN drift matrix
rift matrix --models opus-4-7,sonnet-4-6,gpt-4o --suite reasoning

# Diff two saved runs
rift diff results/before.json results/after.json

# Enterprise contract pricing: apply your negotiated multiplier
rift compare --baseline opus-4-6 --challenger opus-4-7 \
    --suite reasoning --enterprise-multiplier 0.65
```

## What You Get

Real output from `rift compare --baseline opus-4-6 --challenger opus-4-7 --suite context_rot_reasoning --context-rot --subgroup distractor:` on 32 cases (live API run, n=32, McNemar's exact test):

```
╭─────────────────────────────────────────────────╮
│  Rift Drift Report                              │
│                                                 │
│    baseline:   claude-opus-4-6                  │
│    challenger: claude-opus-4-7                  │
│    suite:      context_rot_reasoning (32 cases) │
│                                                 │
│    Status: NO SIGNIFICANT DRIFT                 │
│    Test:   mcnemar_exact                        │
│                                                 │
│    Baseline mean:    0.8125                     │
│    Challenger mean:  0.8750                     │
│    Delta:            +0.0625 (+7.7%)            │
│    p-value:          0.687500                   │
│    95% CI:           [-0.0633, +0.2188]         │
│                                                 │
│    Regressed cases:  2                          │
│    Improved cases:   4                          │
│                                                 │
│    Spend:      $4.72  →  $6.84                  │
│    $/correct:  $0.1815 →  $0.2444  (+35%)       │
╰─────────────────────────────────────────────────╯
```

Followed by a per-subgroup breakdown and a table of regressed cases with
per-case score deltas. Use `-r report.md` to emit the same data as
markdown (see [`benchmarks/context_rot_opus47.md`](benchmarks/context_rot_opus47.md)
for a real example).

### How to read it

Three numbers carry the story:

1. **`Delta` + `95% CI`** — the accuracy change and the range the data is
   consistent with. If the CI crosses zero, the direction is not
   established. Don't report a delta without its CI.
2. **`p-value` + `Test`** — whether the delta is unlikely under the null.
   Rift picks the test automatically: McNemar's exact for binary
   (exact-match) scores, paired t-test + bootstrap for continuous ones.
3. **`$/correct`** — USD per fully-correct case. This is the number a
   budget owner can defend. Two models with the same accuracy aren't
   equivalent if one costs 3× more; `$/correct` folds quality and price
   into one line.

## Define Your Own Eval Suite

```yaml
# my_suite.yaml
name: customer_support_triage
description: Classify support tickets by urgency and category
scoring: exact_match
cases:
  - input: "My account was charged twice for the same order #8812"
    expected:
      urgency: high
      category: billing
  - input: "How do I change my notification preferences?"
    expected:
      urgency: low
      category: settings
```

```bash
rift compare --baseline gpt-4 --challenger gpt-4o --suite my_suite.yaml
```

## Scoring Methods

| Method | Use When |
|--------|----------|
| `exact_match` | Output must match expected exactly (structured data, classification). Tolerates a trailing `Confidence: X` line so the same suite can drive calibration. |
| `fuzzy_match` | Character-sequence similarity via `difflib` (tolerates whitespace, capitalization, minor rewording). **Not** embedding-based — for that, see the roadmap. |
| `llm_judge` | Open-ended outputs (summaries, explanations, code) scored on a 0-1 scale by a separate judge model. Supports both **reference-answer** scoring (`expected: "..."`) and **rubric** scoring (`expected: {rubric: "..."}`). The judge model, judge prompt, and a one-sentence judge reasoning per case are all surfaced for auditability. See `suites/open_ended_qa.yaml` for a worked example. |

### `llm_judge` setup

```bash
# Configure once (or set per-suite via the `judge_model` field):
export RIFT_JUDGE_MODEL=claude-sonnet-4-6

# Compare two models on an open-ended suite:
rift compare --baseline gpt-4o --challenger claude-opus-4-7 \
             --suite open_ended_qa
```

Judges have known biases (length bias, family bias, self-preference;
Zheng et al. 2023). Rift mitigates by asking for a 0-1 numeric score
on a fixed scale (not pairwise A-vs-B), instructing the judge to
ignore wording differences, and caching every judgment by `(judge,
prompt)` so re-runs are deterministic. Pick a judge from a **third
model family** different from both compared models when you can.

## CI/CD Integration

Rift returns exit code 1 when significant drift is detected. Drop it in your deployment pipeline:

```yaml
# GitHub Actions
- name: Check for model drift
  run: rift compare --baseline $CURRENT_MODEL --challenger $NEW_MODEL --suite production_evals
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

## Why Rift?

Every model update is a silent deployment to your production system. Providers don't publish granular changelogs. "Improved reasoning" could mean your extraction pipeline now returns different field names. "Better instruction following" could mean your carefully-tuned prompts behave differently.

Rift gives you the audit trail.

## Executive readout: a worked Opus 4.6 → 4.7 study

Live paired run against the Anthropic API. 32 cases (8 reasoning
prompts × 4 distractor regimes: 0k, 2k, 8k, 32k tokens). Same
scorer, same prompts, byte-identical inputs.

| Signal | Opus 4.6 | Opus 4.7 | Δ |
|---|---|---|---|
| Accuracy | 26/32 (81.2%) | 28/32 (87.5%) | +6.25pp, p=0.69 (**not significant**) |
| Input tokens (byte-identical prompts) | 313,717 | 453,957 | **+44.7%** |
| Total spend | $4.72 | $6.84 | +45% |
| **$/correct** | $0.1815 | $0.2444 | **+35%** |

Three takeaways a leader can act on today:

- **The tokenizer changed; the list price didn't.** Opus 4.7 emits
  1.21–1.62× more input tokens than 4.6 for byte-identical prompts
  (mean 1.43×). At $5/Mtok list, the effective rate on real
  prompts is ~$7.15/Mtok. At 10M daily input tokens, a silent
  default-upgrade costs ~$7.85k/year with zero workload change.
- **The quality lift is directional, not established.** +6.25pp
  overall with the CI `[-0.06, +0.22]` — the data is consistent
  with anything from a small regression to a 22-point improvement.
  The lift concentrates at 8k/32k distractor tokens (both +12.5pp)
  where robustness matters most. Run at n≥50 to move the p-value.
- **`$/correct` is the number to watch.** +35% per fully-correct
  answer on this suite. Even if the quality lift is real, it
  doesn't pay for the tokenizer inflation.

**Action list (cheapest first):** pin model routing to an explicit
`claude-opus-4-6` until you've run the same comparison on your own
prompts; re-baseline your token budgets (multiply committed annual
spend by your observed ratio); renegotiate contracts on
`tokens/prompt × prompts/day`, not `$/Mtok` alone.

Full writeup with reproduction steps, per-subgroup tables, and the
tooling bug Rift caught along the way:
[`benchmarks/context_rot_opus47_analysis.md`](benchmarks/context_rot_opus47_analysis.md).
Raw report: [`benchmarks/context_rot_opus47.md`](benchmarks/context_rot_opus47.md).

## Cost as a first-class signal

Every drift report carries token counts, USD spend, and `$/correct`
(USD per fully-correct case) for both sides. Token-based Enterprise
pricing means quality and price have to be compared together — Rift
reports both so you don't have to reconcile spreadsheets after the
run. See `src/rift/pricing.py` for the catalog; pass
`--enterprise-multiplier` to apply your contracted rate.

## Output-token decomposition

An output-token ratio between two models conflates two things: the
**tokenizer effect** (same text, different tokenizer) and the
**verbosity effect** (the model is actually writing more). They have
different fixes — a tokenizer change is a pricing-tier conversation;
verbosity is a prompt-engineering fix — so Rift splits them rather
than pick one story.

```bash
python benchmarks/analyze_output_tokens.py \
    --baseline  runs/opus46_reasoning.json \
    --challenger runs/opus47_reasoning.json \
    --output benchmarks/output_token_decomposition.md
```

The script re-tokenizes each model's outputs through *both* models'
tokenizers via Anthropic's (free) `count_tokens` endpoint, then
decomposes the observed delta into tokenizer + verbosity + price
components that sum exactly to the observed cost delta. See
`src/rift/output_tokens.py` for the math.

## Context-rot benchmark

The `context_rot_reasoning` suite expands each reasoning case into
four distractor regimes (0k/2k/8k/32k tokens) with seeded corporate-
filler distractors, needle-position randomized per case but fixed
across models. Use `--subgroup distractor:` to get a per-regime
breakdown of where a model starts to fail. See
[`benchmarks/context_rot_opus47_analysis.md`](benchmarks/context_rot_opus47_analysis.md)
for a worked example.

## Statistical tests

Rift picks the test that matches the score distribution:

- **Binary scores (exact-match):** McNemar's exact test on paired
  discordant pairs. Valid at small n; no chi-squared approximation.
- **Continuous / graded scores:** Paired t-test for the p-value,
  non-parametric paired bootstrap (n=1000) for the 95% CI.

Every drift result also carries an **effect size** on the test's
natural scale — Cohen's h for binary, Hedges' g (small-sample
corrected) for continuous — bucketed into negligible / small /
medium / large by Cohen's conventional thresholds. Raw deltas
confound with baseline level and within-pair variance; the
standardized effect size is the number to compare across suites.

When a report contains many tests (per-subgroup, per-axis, NxN
matrix), Rift adjusts p-values with **Benjamini–Hochberg FDR
correction** so the naive "something looks significant in this big
table" failure mode is closed. Subgroup tables show both raw `p`
and adjusted `q (BH)`.

Every comparison also gets a **post-hoc power analysis**: observed
power, minimum detectable effect at 80% power, and (optionally) the
N needed to detect a target effect — the answer to "we did not see
drift, but could we have?".

## Power-stratified case discovery

Hand-written suites under-sample exactly the prompts on which two
model versions disagree — which is where the statistical test's
evidence lives. `rift discover` flips this around: given a
`(baseline, challenger)` pair and a seed suite, it uses a strong
proposer model to generate candidate prompts, runs both models on
each, and keeps the cases that contribute most to the paired test's
power on the discovered suite.

```bash
rift discover \
  --baseline opus-4-6 --challenger opus-4-7 \
  --seed-suite reasoning \
  --proposer-model opus-4-7 \
  --target-power 0.9 --target-effect 0.05 \
  --max-cases 50 \
  --output discovered_reasoning_drift.yaml

# Then feed the discovered suite straight into compare:
rift compare --baseline opus-4-6 --challenger opus-4-7 \
             --suite discovered_reasoning_drift.yaml
```

The output YAML carries full provenance in `description`: proposer
model, target / achieved power, discordant rate, per-stage counts
(proposed → parsed → dedup → validity → kept), and the explicit
caveat that **cases were selected on divergence** — the
achieved-power figure measures the suite's sensitivity, not an
unbiased population estimate.

The framing — "discover cases such that the paired test is powered
at ≥0.9 to detect a 5pp drop" — is the methodological hook nobody
else does. See `src/rift/discovery.py` for the McNemar
information-contribution math.

## Beyond accuracy: refusal, sycophancy, calibration

Three behavioral axes that move independently of accuracy and that
release notes typically hand-wave around:

- **Refusal drift** (`rift refusal a.json b.json`) — classifies each
  output for refusal language and reports over-refusal cases
  (challenger refused prompts the baseline answered correctly) and
  new-compliance cases (baseline refused, challenger answered).
  Fully offline — no extra API calls.
- **Calibration drift** (`rift calibration a.json b.json`) — parses
  stated confidence from outputs (`Confidence: 0.85`, `I am 85%
  sure`, etc.) and reports Brier score, ECE, and overconfidence
  deltas. Cases without parseable confidence are surfaced, not
  silently coerced.
- **Sycophancy probe** (`rift sycophancy --model X --suite Y`) —
  runs the suite twice; the second pass pushes back on each of the
  model's answers and measures the **flip rate** among
  originally-correct cases. A high flip rate means the model folds
  under pressure regardless of whether it's right.

## Roadmap

- [x] CLI with compare, run, diff, matrix commands
- [x] Anthropic + OpenAI providers
- [x] Built-in eval suites + context-rot expansion
- [x] Statistical significance testing with test selection
- [x] Cost-per-correct metrics + Enterprise pricing multiplier
- [x] Effect sizes (Cohen's h / Hedges' g) on every drift result
- [x] Benjamini–Hochberg FDR correction for multi-test reports
- [x] Post-hoc power analysis + minimum detectable effect
- [x] Refusal / over-refusal drift detection
- [x] Calibration drift (Brier / ECE / overconfidence)
- [x] Sycophancy probe (pushback flip rate)
- [x] `llm_judge` scorer for open-ended outputs (reference + rubric)
- [x] Power-stratified auto-adversarial case discovery (`rift discover`)
- [ ] Reasoning faithfulness perturbations
- [ ] Embedding-based semantic scoring
- [ ] User-defined `custom` scoring functions
- [ ] Hosted monitoring (continuous drift alerts)
- [ ] CI/CD plugins (GitHub Actions, Jenkins)
- [ ] Observability integrations (Datadog, W&B)

## License

MIT
