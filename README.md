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
│  baseline: claude-3-5-sonnet-20241022           │
│  challenger: claude-sonnet-4-20250514           │
│  suite: structured_extraction (47 cases)        │
├─────────────────────────────────────────────────┤
│  Aggregate Drift Score: 0.12 (p=0.003) ⚠️       │
│                                                 │
│  exact_match    0.91 → 0.83  ▼ -8.7%  p=0.01  │
│  field_recall   0.95 → 0.94  ▼ -1.1%  p=0.34  │
│  format_valid   1.00 → 0.96  ▼ -4.3%  p=0.08  │
│                                                 │
│  3 / 47 cases regressed significantly           │
│  See full report: rift_report.md                │
╰─────────────────────────────────────────────────╯
```

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
| `semantic` | Meaning matters more than wording (summaries, explanations) |
| `llm_judge` | Complex quality assessment (creative writing, nuanced reasoning) |
| `custom` | Your own scoring function |

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
- [ ] Hosted monitoring (continuous drift alerts)
- [ ] CI/CD plugins (GitHub Actions, Jenkins)
- [ ] Observability integrations (Datadog, W&B)

## License

MIT
