# Rift

**You upgraded your model. What broke?**

Rift detects behavioral regressions between LLM model versions. Run structured eval suites against any two models, get statistically rigorous drift reports.

No vibes. No "it feels dumber." Just p-values and confidence intervals.

## Quick Start

```bash
pip install rift-eval

# Compare two models
rift compare \
  --baseline claude-3-5-sonnet-20241022 \
  --challenger claude-sonnet-4-20250514 \
  --suite summarization

# Run against a built-in suite
rift run --model gpt-4o --suite extraction

# Diff two saved runs
rift diff results/before.json results/after.json
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

## Roadmap

- [x] CLI with compare, run, diff commands
- [x] Anthropic + OpenAI providers
- [x] Built-in eval suites
- [x] Statistical significance testing
- [ ] Hosted monitoring (continuous drift alerts)
- [ ] CI/CD plugins (GitHub Actions, Jenkins)
- [ ] Observability integrations (Datadog, W&B)

## License

MIT
