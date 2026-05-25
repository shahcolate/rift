# Rift — LLM Drift Detection Tool

## What This Is
Rift is an open-source CLI that detects behavioral regressions between LLM model versions. It runs structured eval suites against two model endpoints, computes statistical significance of output differences, and generates drift reports.

The pitch: "You upgraded your model. What broke?"

## Architecture

```
rift/
├── src/rift/
│   ├── cli.py              # CLI entry: compare, run, diff, matrix
│   ├── runner.py            # Async eval engine (retries, timeouts, cost tagging)
│   ├── comparator.py        # McNemar + paired t-test + bootstrap + cost-normalized
│   ├── reporter.py          # Terminal, markdown, subgroup + NxN matrix rendering
│   ├── pricing.py           # Token price catalog + enterprise multiplier
│   ├── context_rot.py       # Distractor-injection suite expansion
│   ├── scoring/
│   │   ├── exact_match.py
│   │   └── semantic.py
│   ├── providers/
│   │   ├── __init__.py      # Abstract BaseProvider + Completion dataclass
│   │   ├── anthropic.py
│   │   ├── google.py        # Gemini (Generative Language API)
│   │   └── openai.py
│   └── config.py            # YAML parsing + model alias resolution
├── suites/
│   ├── summarization.yaml
│   ├── extraction.yaml
│   ├── reasoning.yaml
│   ├── code_generation.yaml
│   └── context_rot_reasoning.yaml
├── benchmarks/
│   ├── run_context_rot.py              # Reproducible benchmark driver (live|record)
│   ├── generate_synthetic_outcomes.py  # Seeded prior-model outcomes generator
│   ├── context_rot_outcomes.yaml       # Recorded outcomes (committed for repro)
│   ├── context_rot_opus47.md           # Raw Rift drift report
│   └── context_rot_opus47_analysis.md  # Methodology + findings writeup
├── tests/
├── pyproject.toml
├── README.md
└── CLAUDE.md
```

## Core Concepts

- **Suite**: A YAML file defining a set of eval cases (input, expected output, scoring method)
- **Provider**: An LLM API endpoint (Anthropic, OpenAI, or custom)
- **Run**: One execution of a suite against a single model
- **Comparison**: Statistical analysis of two runs (baseline vs challenger)
- **Drift Score**: Per-task and aggregate metric quantifying behavioral change

## CLI Interface

```bash
# Compare two models on a suite
rift compare --baseline claude-3-5-sonnet --challenger claude-sonnet-4 --suite summarization

# Run a single model and save results
rift run --model gpt-4o --suite extraction --output results/gpt4o_extraction.json

# Compare two saved runs
rift diff results/run_a.json results/run_b.json

# Generate a markdown report
rift report results/comparison.json --format markdown --output drift_report.md
```

## Config Format (suite YAML)

```yaml
name: structured_extraction
description: Extract structured data from messy text inputs
scoring: exact_match  # or semantic, llm_judge, custom
cases:
  - input: |
      Invoice #4521, issued Jan 15 2025, total $1,240.00 to Acme Corp
    expected:
      invoice_number: "4521"
      date: "2025-01-15"
      total: 1240.00
      recipient: "Acme Corp"
  - input: |
      ... more cases
```

## Tech Stack

- Python 3.11+
- Click (CLI framework)
- httpx (async HTTP for API calls)
- PyYAML (config parsing)
- numpy + scipy (statistical tests)
- rich (terminal output formatting)

## Design Principles

1. Zero config to start: `rift compare --baseline X --challenger Y --suite Z` should just work
2. Statistically rigorous: every drift claim backed by confidence intervals
3. Suite-driven: evals are data (YAML), not code
4. Provider-agnostic: any model with an HTTP endpoint works
5. Output is publishable: markdown reports designed for blog posts and READMEs

## Key Implementation Notes

- Use async throughout for parallel eval execution. Per-case timeout
  (180s default) and exponential-backoff retries on transient errors
  (429, 5xx, timeouts) live in `runner.py`. Non-transient 4xx errors
  bubble up immediately.
- Cache completions by `(model, model_params, input_hash)`. Changing
  temperature invalidates the cache; rewording a prompt invalidates
  the cache; changing the model obviously invalidates the cache.
- Provider instantiation is lazy — fully cached runs (including
  benchmark replays from recorded outcomes) work without API keys.
- Cache writes are atomic (tmp + rename) so a crashed runner never
  leaves a half-written JSON.
- Every `CaseResult` carries `input_tokens`, `output_tokens`, and
  `cost_usd`. Do not drop any of these — the cost-normalized drift
  metrics depend on them.
- Statistical test selection is automatic: binary scores use
  McNemar's exact test (binomial on discordant pairs); continuous
  scores use paired t-test + paired bootstrap CI. The chosen test is
  stored in `DriftResult.test_used`.
- Every drift claim ships with a confidence interval. Accuracy CIs
  come from `_bootstrap_ci`; the `$/correct` delta CI comes from
  `_bootstrap_cost_per_correct_delta_ci` (paired bootstrap on
  `(score, cost)` tuples, seed=42). Both populate `DriftResult`
  fields — do not surface a delta in a report without its CI. The
  cost CI is undefined when either run has zero correct cases
  (per-correct is infinite); in that case `cost_delta_ci_defined`
  is `False` and renderers must skip the line.
- Effect sizes: binary tests report `cohens_h_marginal` (Cohen's h on
  the marginal proportions) AND `cohens_g_paired` (paired-binary
  effect on discordant cells). Both are required because h ignores
  the paired structure McNemar uses — they can disagree (h=0 with
  half the pairs flipped). Continuous tests report `hedges_g`.
- `rift matrix` applies a Benjamini–Hochberg correction across ALL
  off-diagonal pairwise p-values before colouring "significant"
  cells. A 4-model matrix runs 12 tests; without correction, the
  expected false-positive count at α=0.05 under all-null is 0.6.
  Both raw `p` and BH `q` are shown in each cell.
- Saved JSONs honour `--strip-io` on `compare`, `run`, and `matrix`:
  when set, per-case `input_text` and `output` are emptied. Use this
  for proprietary suites. The flag is a publishing safety, not a
  privacy primitive — secrets in `tags` or `expected` still ship.
- Exit code 0 = no significant drift; exit code 1 = significant
  regression detected (for CI/CD integration).
- Benchmarks live under `benchmarks/`. Any benchmark worth publishing
  should run reproducibly in `--mode record` against a committed
  outcomes file. **`opus47_live.md` is the authoritative live capture;
  `context_rot_opus47.md` is a synthetic replay calibrated to it.**
  Never quote a number from the synthetic file without flagging the
  provenance — the `run_context_rot.py --mode record` path emits a
  warning at the top of every regenerated report.

## Environment Variables

- ANTHROPIC_API_KEY — for Anthropic provider
- OPENAI_API_KEY — for OpenAI provider
- GEMINI_API_KEY — for Google (Gemini) provider
- RIFT_CACHE_DIR — override cache location (default: .rift/cache)

## Development Commands

```bash
# Install in dev mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linting
ruff check src/

# Run type checking
pyright src/
```
