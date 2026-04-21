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
| `exact_match` | Output must match expected exactly (structured data, classification) |
| `fuzzy_match` | Character-sequence similarity via `difflib` (tolerates whitespace, capitalization, minor rewording). **Not** embedding-based — for that, see the roadmap. |

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

The CI is always reported, and the chosen test is named in every
report. See `src/rift/comparator.py` for the exact logic.

## Roadmap

- [x] CLI with compare, run, diff, matrix commands
- [x] Anthropic + OpenAI providers
- [x] Built-in eval suites + context-rot expansion
- [x] Statistical significance testing with test selection
- [x] Cost-per-correct metrics + Enterprise pricing multiplier
- [ ] Embedding-based semantic scoring
- [ ] `llm_judge` scorer for open-ended outputs
- [ ] User-defined `custom` scoring functions
- [ ] Multi-metric drift breakdown in a single run
- [ ] Hosted monitoring (continuous drift alerts)
- [ ] CI/CD plugins (GitHub Actions, Jenkins)
- [ ] Observability integrations (Datadog, W&B)

## License

MIT
