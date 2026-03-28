# Rift — LLM Drift Detection Tool

## What This Is
Rift is an open-source CLI that detects behavioral regressions between LLM model versions. It runs structured eval suites against two model endpoints, computes statistical significance of output differences, and generates drift reports.

The pitch: "You upgraded your model. What broke?"

## Architecture

```
rift/
├── src/
│   ├── cli.py              # CLI entry point (Click-based)
│   ├── runner.py            # Eval execution engine
│   ├── comparator.py        # Statistical comparison (paired bootstrap, t-tests)
│   ├── reporter.py          # Output formatting (terminal, JSON, markdown)
│   ├── scoring/
│   │   ├── exact_match.py   # Exact match scorer
│   │   ├── semantic.py      # Embedding-based semantic similarity
│   │   ├── llm_judge.py     # LLM-as-judge scorer
│   │   └── custom.py        # User-defined scoring functions
│   ├── providers/
│   │   ├── base.py          # Abstract provider interface
│   │   ├── anthropic.py     # Anthropic API provider
│   │   ├── openai.py        # OpenAI API provider
│   │   └── local.py         # Local/custom endpoint provider
│   └── config.py            # YAML config parsing and validation
├── suites/                  # Built-in eval suites
│   ├── summarization.yaml
│   ├── extraction.yaml
│   ├── reasoning.yaml
│   └── code_generation.yaml
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

- Use async throughout for parallel eval execution
- Cache individual completions by (model, input_hash) to avoid redundant API calls
- Store raw responses alongside scores for manual inspection
- Default to paired bootstrap confidence intervals (n=1000) for significance testing
- Use rich for terminal output with progress bars during runs
- Exit code 0 = no significant drift, exit code 1 = significant drift detected (for CI/CD integration)

## Environment Variables

- ANTHROPIC_API_KEY — for Anthropic provider
- OPENAI_API_KEY — for OpenAI provider
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
