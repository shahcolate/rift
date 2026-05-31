# Rift — LLM Drift Detection Tool

## What This Is
Rift is an open-source CLI that detects behavioral regressions between LLM model versions. It runs structured eval suites against two model endpoints, computes statistical significance of output differences, and generates drift reports.

The pitch: "You upgraded your model. What broke?"

## Architecture

```
rift/
├── src/rift/
│   ├── cli.py              # CLI entry: compare, run, diff, matrix, faithfulness, ...
│   ├── runner.py            # Async eval engine (retries, timeouts, cost tagging)
│   ├── comparator.py        # McNemar + paired t-test + bootstrap + cost-normalized
│   ├── reporter.py          # Terminal, markdown, subgroup + NxN matrix rendering
│   ├── observability.py     # Flat metrics export (JSON / Prometheus) for dashboards
│   ├── pricing.py           # Token price catalog + enterprise multiplier
│   ├── prompts.py           # Registry of user-overridable probe prompt templates
│   ├── context_rot.py       # Distractor-injection suite expansion
│   ├── faithfulness.py      # Reasoning-faithfulness probe (biasing-hint articulation)
│   ├── scoring/
│   │   ├── exact_match.py
│   │   ├── custom.py        # Loader for user-supplied `scoring: custom` scorers
│   │   ├── llm_judge.py
│   │   ├── semantic.py      # Embedding-cosine scorer (OpenAI + Google backends)
│   │   └── faithfulness_judge.py  # Articulation judge (did reasoning admit the cue?)
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
│   ├── context_rot_reasoning.yaml
│   └── faithfulness_reasoning.yaml  # Seed suite for `rift faithfulness`
├── benchmarks/
│   ├── run_context_rot.py              # Reproducible benchmark driver (live|record)
│   ├── generate_synthetic_outcomes.py  # Seeded prior-model outcomes generator
│   ├── context_rot_outcomes.yaml       # Recorded outcomes (committed for repro)
│   ├── context_rot_opus47.md           # Raw Rift drift report
│   └── context_rot_opus47_analysis.md  # Methodology + findings writeup
├── tests/
├── .github/
│   ├── actions/rift-drift-check/  # Reusable composite action (drift gate in CI)
│   └── workflows/publish.yml      # Build + PyPI publish on GitHub Release (OIDC)
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
- **Faithfulness**: Whether a model's stated reasoning reflects what drove its
  answer. `rift faithfulness --mode hint|cot|both`:
  - **hint** plants a biasing cue toward a generated plausible-wrong answer; a
    case is *unfaithful* when the model is swayed to that answer but its
    reasoning does not acknowledge the cue (an LLM judge decides
    acknowledgement).
  - **cot** captures each model's chain-of-thought, then re-asks under a
    truncated/corrupted version of it; a case is *faithful* when the answer
    *changes* (the CoT was load-bearing), unfaithful when it stays put
    (post-hoc). The perturbation suite is built per-model from that model's own
    reasoning; no judge/proposer key needed.

  Both modes score only control-correct cases and run the paired drift test on
  the intersection of both models' control-correct cases. Control correctness is
  judged on the *parsed* `Answer:` line (`_score_answer`), not the raw
  step-by-step output. See `faithfulness.py`.

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
scoring: exact_match  # or fuzzy_match, semantic, llm_judge, exec_tests, custom
# Optional per-suite probe-prompt overrides (see prompts.py). Validated at
# load: unknown key or a template missing a required placeholder => error.
prompts:
  judge_rubric: |       # must keep {question} {target_block} {output}
    ... custom grading rubric ...
cues:
  authority: "A domain expert insists the answer is {target}."  # must keep {target}
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

Probe prompts are overridable per suite via `prompts:` (registry keys:
`judge_rubric`, `faithfulness_judge`, `faithfulness_format_instruction`,
`faithfulness_wrong_answer`, `faithfulness_cot_early`,
`faithfulness_cot_mistake`) and `cues:` (faithfulness cue name → hint template).
`rift.prompts` validates required placeholders at load time and resolves
overrides → defaults; the runner stamps `metadata["custom_prompts"]` so a
published report discloses any non-default prompt. Adding a new overridable
prompt is a one-line `PROMPT_REGISTRY` entry + a default in `_default_for`.

Custom scoring: `scoring: custom` + `custom_scorer: "target:callable"` (an
importable `module:fn` or a `./file.py:fn` resolved against the suite file's
dir, tracked via `SuiteConfig._source_dir` — which `_with_cases` in
`context_rot.py` must preserve). The target may be a sync
`score(output, expected)`, an async `ascore(output, expected, context=None)`,
or a Scorer class/instance; plain functions are wrapped in `scoring/custom.py`.
`SuiteConfig` validates the spec at load (custom ⇒ scorer required; scorer ⇒
custom required); the loader executes the target module (trust boundary — only
run suites you trust) and the runner stamps `metadata["custom_scorer"]`.

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
  This keyless guarantee covers `rift demo` and code that calls
  `run_suite` directly (benchmarks, demo replay), which do not
  preflight. The live CLI commands (`compare`/`run`/`matrix`/
  `sycophancy`/`discover`/`faithfulness`) DO preflight keys via
  `ensure_provider_keys`
  for a clean fail-fast prompt, so they require a key even when a run
  would have been fully cached. A missing key raised lazily anywhere
  (e.g. an llm_judge judge key) is re-raised by `run_suite` rather than
  swallowed as a per-case error, so it always shows the clean message.
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
- Effect sizes: binary tests always populate `cohens_h_marginal`
  (Cohen's h on the marginal proportions). `cohens_g_paired`
  (Cohen's g on the discordant cells) is also populated whenever
  there is at least one discordant pair; it is `None` when every
  pair is concordant (test is uninformative). Both are surfaced
  side-by-side because h ignores the paired structure McNemar uses:
  the two measure different things and can carry different verdicts
  on the same data (e.g. modest h with a strongly one-sided
  discordant split, or non-negligible h with discordants nearly
  balanced). Continuous tests report `hedges_g`.
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
- RIFT_JUDGE_MODEL — default judge model for `llm_judge` scoring
- RIFT_EMBEDDING_MODEL — default embedding model for `semantic` scoring
  (default: text-embedding-3-small; OpenAI or Google ids)

Provider keys are also auto-loaded from `~/.rift/.env` then `./.env`
(real env vars always win — `os.environ.setdefault`). `rift setup`
writes `~/.rift/.env` (mode 0600); see `keys.py`. Live commands
(`compare`/`run`/`matrix`/`sycophancy`/`discover`/`faithfulness`)
preflight the
needed keys via `ensure_provider_keys` — interactive TTY prompts for a
missing key, non-interactive raises `MissingAPIKeyError` (a
`ClickException` → clean message, exit 1, never a traceback). The
demo and cached/replay paths stay keyless (lazy provider init).

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
