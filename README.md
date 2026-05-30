# Rift

**You upgraded your model. What broke?
You're picking a vendor. Who actually wins?**

Rift compares any two (or three+) LLM endpoints on structured eval
suites and returns statistically rigorous drift reports with
cost-per-correct. Use it to catch silent regressions on a same-family
upgrade — or to settle a cross-vendor procurement call with real
numbers instead of list-price math.

No vibes. No "it feels dumber." Just p-values, confidence intervals,
and `$/correct`.

## Try the demo (no API key needed)

```bash
pip install rift-eval
rift demo
```

A 30-second guided walkthrough modelled on one real model upgrade
(Opus 4.6 → 4.7): accuracy ticks up, but cost-per-correct rises +35%
in the live run from a silent tokenizer change. The demo replays a
**synthetic reproduction** calibrated to the live 2026-04-21 capture
([`benchmarks/opus47_live.md`](benchmarks/opus47_live.md)) and will
display roughly +40% — within the documented calibration tolerance.
Fully offline, reproducible, no keys. For the authoritative live
numbers, see that file.

Forward the one-page memo to your VP:

```bash
rift demo --export-html demo.html      # self-contained executive memo
rift demo --export-md  demo.md         # for Notion/Slack/email
rift demo --paced                      # press Enter between acts (live)
```

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/demo.svg">
  <img alt="Rift demo screenshot — four-act terminal walkthrough" src="assets/demo.svg">
</picture>

## Quick Start

```bash
pip install rift-eval

# One-time: add your provider API key(s). Paste when prompted; saved to
# ~/.rift/.env and loaded automatically from then on. (The demo needs none.)
rift setup

# Compare two models (with short aliases — opus-4-8, opus-4-7, sonnet-4-6, etc.)
rift compare --baseline opus-4-7 --challenger opus-4-8 --suite reasoning

# Stress-test reasoning under distractor context (0k/2k/8k/32k)
rift compare --baseline opus-4-7 --challenger opus-4-8 \
    --suite context_rot_reasoning --context-rot --subgroup distractor:

# Compare 3+ models at once — prints an NxN drift matrix
rift matrix --models opus-4-8,opus-4-7,opus-4-6 --suite reasoning

# Diff two saved runs
rift diff results/before.json results/after.json

# Enterprise contract pricing: apply your negotiated multiplier
rift compare --baseline opus-4-6 --challenger opus-4-7 \
    --suite reasoning --enterprise-multiplier 0.65
```

## What You Get

Output from `rift compare --baseline opus-4-6 --challenger opus-4-7 --suite context_rot_reasoning --context-rot --subgroup distractor:` on 32 cases — numbers below are from the **live Anthropic API run on 2026-04-21** (authoritative capture: [`benchmarks/opus47_live.md`](benchmarks/opus47_live.md), n=32, paired, McNemar's exact, $3.85 total spend, 0 errors; dollar figures reflect the current $5/$25 Opus 4.5-generation list price):

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
│    Spend:      $1.57  →  $2.28                  │
│    $/correct:  $0.0605 →  $0.0815  (+35%)       │
╰─────────────────────────────────────────────────╯
```

Followed by a per-subgroup breakdown and a table of regressed cases with
per-case score deltas. Use `-r report.md` to emit the same data as
markdown.

> **Reproducibility note.** The committed
> [`benchmarks/context_rot_outcomes.yaml`](benchmarks/context_rot_outcomes.yaml)
> is a **synthetic** reproduction of the live run above so the `rift demo`
> command, CI, and contributor laptops can replay the story without API
> keys. Calibration fit (synthetic vs. live, as measured 2026-04-21):
> per-model $/correct levels within ±10% (+3.7% baseline, +7.6%
> challenger); top-level accuracy within ~3pp (baseline 0.8438 vs 0.8125;
> challenger 0.8750 vs 0.8750); the headline Δ $/correct % within ~5pp
> (+39.7% vs +34.7%). Subgroup-level numbers can diverge materially —
> the live capture shows a baseline regression at 32k context that the
> synthetic does not reproduce. **For procurement or roadmap decisions,
> cite the authoritative live capture
> [`opus47_live.md`](benchmarks/opus47_live.md), not the demo replay
> [`context_rot_opus47.md`](benchmarks/context_rot_opus47.md).** The
> calibration constants are documented in
> [`generate_synthetic_outcomes.py`](benchmarks/generate_synthetic_outcomes.py).

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

## Worked studies

Paired runs against live APIs, one for each question in the
tagline. Run-level reports (markdown) and per-case completion JSONs
are committed under `benchmarks/`; re-running offline from those
captures requires the cache to be re-primed (the offline `rift demo`
replays the same headline numbers from a calibrated synthetic file —
see the reproducibility note above).

### Did the upgrade regress? — Opus 4.7 → 4.8

Live paired run against the Anthropic API on Opus 4.8 launch day
(2026-05-29), 4.8 compared against 4.7 and 4.6 across six suites.
**4.8 is a statistically indistinguishable sidegrade on five standard
suites (reasoning, extraction, code generation, open-ended QA,
summarization) — and slightly cheaper per correct.** But on
long-context reasoning with injected distractors it regresses:

| Signal | Opus 4.7 | Opus 4.8 | Δ |
|---|---|---|---|
| Accuracy (context-rot, n=32) | 87.5% (28/32) | 68.75% (22/32) | **−18.75pp, p=0.031 (significant)** |
| Regressed / improved cases | — | — | **6 / 0** (paired g = −1.000) |
| Total spend | $2.29 | $2.28 | ~flat |
| **$/correct** | $0.0820 | $0.1036 | **+26%** |
| Refusal rate | 0.0% | 0.0% | no over-refusal |

The +26% cost-per-correct is *not* a price story — spend is flat to the
cent. It rises because 4.8 gets fewer answers right for the same money.
All six regressions are cases carrying injected "reference material"
distractors: **4.8 is more distractible by irrelevant long context than
4.7 was**, a regression a green standard-benchmark sheet would have
hidden. Full writeup, per-suite matrices, and the
"what-not-to-claim" caveats:
[`benchmarks/3way_opus48/analysis.md`](benchmarks/3way_opus48/analysis.md).

### Did the upgrade regress? — Opus 4.6 → 4.7

Live paired run against the Anthropic API. 32 cases (8 reasoning
prompts × 4 distractor regimes: 0k, 2k, 8k, 32k tokens). Same
scorer, same prompts, byte-identical inputs.

| Signal | Opus 4.6 | Opus 4.7 | Δ |
|---|---|---|---|
| Accuracy | 26/32 (81.2%) | 28/32 (87.5%) | +6.25pp, p=0.69 (**not significant**) |
| Input tokens (byte-identical prompts) | 313,717 | 453,957 | **+44.7%** |
| Total spend | $1.57 | $2.28 | +45% |
| **$/correct** | $0.0605 | $0.0815 | **+35%** |

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

### Which vendor wins per correct? — gpt-5.5 vs Opus 4.7 vs Gemini 3.5 Flash

> **Test-set contamination caveat.** The suites in `suites/` are public
> in this repository. Frontier models trained on web snapshots after this
> repo went public may have these prompts in training data, which can
> inflate performance on the public suites without reflecting real-world
> behaviour. Treat cross-vendor numbers below as **suggestive, not
> authoritative**. For procurement decisions, run `rift discover` against
> your own private prompts and compare on that (still adversarially-
> selected — see `rift discover`'s output caveat — but at least not
> public).
>
> Exact-match scoring also rewards terse outputs; vendors whose default
> tone is more verbose (e.g. Anthropic) may underperform on this metric
> relative to their actual quality. See [`suites/`](suites/) for the
> exact `expected` outputs each suite enforces.

Three frontier models, three suites (reasoning n=10, structured
extraction n=29, open-ended QA n=5), same scorers, byte-identical
prompts, single trial, temperature 0. 132 live completions; token
counts from the 2026-05-21 live capture, Opus dollar figures
recomputed at the current $5/$25 list price. Recomputed total
spend: **$0.43** *(see
[`benchmarks/3way_full/analysis.md`](benchmarks/3way_full/analysis.md))*.

| Suite | gpt-5.5 $/c | Opus 4.7 $/c | Gemini Flash $/c | Verdict |
|---|---|---|---|---|
| reasoning | $0.0026 | **$0.0019** | $0.0056 | Opus now cheapest, same accuracy (9/10 each) |
| extraction | **$0.0027** | $0.0029 | $0.0061 | gpt-5.5 ≈ Opus (tie), both ~2× cheaper than Gemini |
| open_ended_qa | **$0.0034** | $0.0056 | $0.0163 | Opus uniquely perfect (5/5); gpt-5.5 cheapest |

Three takeaways a leader can act on:

- **The Opus 4.5-generation price cut (to $5/$25) reopens the cost
  race — the cheapest model is now suite-dependent.** Per-Mtok list
  prices are Gemini $1.50/$9, gpt-5.5 $5/$20, Opus $5/$25. Opus and
  gpt-5.5 now share an input price, so the bill is decided by output
  volume: Opus is cheapest on reasoning (terse output, 471 tok vs
  gpt-5.5's 953), tied on extraction, and gpt-5.5 keeps the edge only
  on free-form QA where Opus is the verbose one. The bill is
  `output_tokens × output_price`, not `output_price`.
- **The I:O-ratio mechanism from the prior 2-way writeup reproduces.**
  Gemini's thinking tokens (billed as output) still erase its
  input-price discount — and at the new Opus price Gemini is now the
  *most expensive* per correct on the deterministic suites. Pricing
  decisions on per-token list prices alone are still wrong; multiply
  by *your* observed output volume.
- **Opus retains a judge-scored quality edge on free-form generation**,
  now at a 1.6× cost premium over gpt-5.5 (was 5× at the old price),
  with the same family-bias caveat as before (judge is Claude Sonnet
  4.6). The 3-way data weakens but doesn't refute the caveat — re-run
  with a non-Anthropic judge before treating the gap as settled.

Full writeup with per-suite tables, statistical tests, and an
executive action list:
[`benchmarks/3way_full/analysis.md`](benchmarks/3way_full/analysis.md).
Prior 2-way that this builds on:
[`benchmarks/opus47_vs_gemini35_analysis.md`](benchmarks/opus47_vs_gemini35_analysis.md).

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
| `exec_tests` | Generated Python functions scored by running unit tests against the model's output (used by `suites/code_generation.yaml`). Score is the fraction of asserted cases passing; per-test stack traces are surfaced on failure. |

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

## Providers

| Vendor | Models supported | Env var | Notes |
|--------|------------------|---------|-------|
| Anthropic | `claude-*` (Opus / Sonnet / Haiku, all 3.x / 4.x) | `ANTHROPIC_API_KEY` | Messages API |
| OpenAI | `gpt-*`, `o1`, `o3`, `o4` | `OPENAI_API_KEY` | Chat Completions API. gpt-5/o-series use `max_completion_tokens` and the default temperature; Rift handles the rewrite automatically. |
| Google | `gemini-*` (3.5 Flash and family) | `GEMINI_API_KEY` | Generative Language API (AI Studio key). Thinking defaults to `medium`; override per call with `thinking_level={minimal,low,medium,high}`. Thinking tokens roll into `output_tokens` for cost accounting. |

Short aliases (`opus-4-8`, `opus-4-7`, `sonnet-4-6`, `gemini-flash`, `gpt-5.5`,
etc.) live in `MODEL_ALIASES` in `src/rift/config.py`. Cross-vendor
comparisons work out of the box:

```bash
rift matrix \
  --models gpt-5.5,opus-4-7,gemini-3-5-flash \
  --suite reasoning
```

## CI/CD Integration

Rift returns exit code 1 when significant drift is detected. Drop it in your deployment pipeline:

```yaml
# GitHub Actions
- name: Check for model drift
  run: rift compare --baseline $CURRENT_MODEL --challenger $NEW_MODEL --suite production_evals
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

---

The sections below document the mechanics behind those headlines.
Skip if you only need to use the tool.

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
(proposed → dedup → both-zero rejects → kept), whether the loop
**early-stopped on achieved-power** or ran to `max_cases`, and the
explicit caveat that **cases were selected on divergence** — the
achieved-power figure measures the suite's sensitivity, not an
unbiased population estimate.

The loop is **iterative**: after the first batch, every subsequent
proposer call surfaces the accepted-so-far cases and asks for
*different* failure modes. This drives diversity without manual
prompting. For continuous-score seed suites (`fuzzy_match`,
`llm_judge`), pass `--min-info 0.2` to filter out near-tie cases
that would dilute the discovered suite's power.

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
- [x] Anthropic + OpenAI + Google providers
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
- [x] `exec_tests` scorer for code generation suites
- [x] Power-stratified auto-adversarial case discovery (`rift discover`)
- [ ] Reasoning faithfulness perturbations
- [ ] Embedding-based semantic scoring
- [ ] User-defined `custom` scoring functions
- [ ] Hosted monitoring (continuous drift alerts)
- [ ] CI/CD plugins (GitHub Actions, Jenkins)
- [ ] Observability integrations (Datadog, W&B)

## License

MIT
