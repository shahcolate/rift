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

```
╭─────────────────────────────────────────────────╮
│  Rift Drift Report                              │
│                                                 │
│    baseline:   claude-3-5-sonnet-20241022       │
│    challenger: claude-sonnet-4-20250514         │
│    suite:      structured_extraction (47 cases) │
│                                                 │
│    Status: REGRESSION DETECTED                  │
│                                                 │
│    Baseline mean:    0.9149                     │
│    Challenger mean:  0.8298                     │
│    Delta:            -0.0851 (-9.3%)            │
│    p-value:          0.003421                   │
│    95% CI:           [-0.1243, -0.0459]         │
│                                                 │
│    Regressed cases:  3                          │
│    Improved cases:   0                          │
╰─────────────────────────────────────────────────╯
```

A table of regressed cases (with inputs and per-case score deltas) is printed
below the summary. Use `rift report` to emit the same data as a markdown file.

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
| `semantic` | Fuzzy string similarity (tolerates whitespace, capitalization, minor rewording) |

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
