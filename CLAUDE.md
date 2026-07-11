# Rift — LLM Drift Detection Tool

## What This Is
Rift is an open-source CLI that detects behavioral regressions between LLM model versions. It runs structured eval suites against two model endpoints, computes statistical significance of output differences, and generates drift reports.

The pitch: "You upgraded your model. What broke?"

## Architecture

```
rift/
├── src/rift/
│   ├── __init__.py          # Public library API (lazy PEP 562 exports, semver)
│   ├── cli.py              # CLI entry: compare, run, diff, matrix, report, import, ...
│   ├── runner.py            # Async eval engine (retries, timeouts, cost tagging)
│   ├── comparator.py        # McNemar + paired t-test + bootstrap + cost-normalized
│   ├── reporter.py          # Terminal, markdown, subgroup + NxN matrix rendering
│   ├── brief.py             # `rift report`: reload saved comparisons + exec upgrade brief
│   ├── demo.py              # `rift demo`: recorded walkthrough + HTML/md/SVG exporters
│   ├── observability.py     # Flat metrics export (JSON / Prometheus) for dashboards
│   ├── observatory.py       # Longitudinal monitoring: records, budget guard, drift feed
│   ├── observatory_site.py  # Static observatory dashboard (hand-rolled SVG, zero JS) + RSS
│   ├── selftest.py          # Null calibration: gate false-positive rate (self-vs-self)
│   ├── judge_validation.py  # Articulation-judge gold set + Cohen's kappa
│   ├── preregistration.py   # Pre-registered primary endpoint (anti forking-paths)
│   ├── pricing.py           # Token price catalog + enterprise multiplier
│   ├── prompts.py           # Registry of user-overridable probe prompt templates
│   ├── context_rot.py       # Distractor-injection suite expansion
│   ├── faithfulness.py      # Reasoning-faithfulness probe (biasing-hint articulation)
│   ├── sycophancy.py        # Pushback probe: flip rate on originally-correct answers
│   ├── refusal.py           # Refusal / over-refusal classification + drift
│   ├── calibration.py       # Confidence parsing, Brier / ECE drift
│   ├── discovery.py         # `rift discover`: power-stratified adversarial suite mining
│   ├── keys.py              # API-key preflight, ~/.rift/.env, `rift setup`
│   ├── adapters/            # `rift import`: promptfoo / Inspect / lm-eval / OpenAI evals
│   │   ├── scorers.py       # Bundled contains/regex scorers for imported suites
│   │   └── ...              # one module per source format + shared _common.py
│   ├── lm/                  # RiftLM: built-in tiny GPT, pure numpy (no torch)
│   │   ├── data.py          # Synthetic tasks (cpy/rev/srt/max) + hash train/eval split
│   │   ├── model.py         # TinyGPT: forward, hand-written backprop, Adam, greedy decode
│   │   └── train.py         # Training loop w/ mid-run task-mix shift → ckpt A/B pair
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
│   │   ├── openai.py
│   │   └── riftlm.py        # In-process provider for RiftLM checkpoints (keyless)
│   └── config.py            # YAML parsing + model alias resolution
├── suites/
│   ├── summarization.yaml
│   ├── extraction.yaml
│   ├── reasoning.yaml
│   ├── code_generation.yaml
│   ├── context_rot_reasoning.yaml
│   ├── faithfulness_reasoning.yaml  # Seed suite for `rift faithfulness`
│   ├── hard_reasoning.yaml
│   ├── open_ended_qa.yaml
│   └── riftlm.yaml                  # Held-out RiftLM eval (generated: `rift lm suite`)
├── benchmarks/
│   ├── run_context_rot.py              # Reproducible benchmark driver (live|record)
│   ├── generate_synthetic_outcomes.py  # Seeded prior-model outcomes generator
│   ├── context_rot_outcomes.yaml       # Recorded outcomes (committed for repro)
│   ├── context_rot_opus47.md           # Raw Rift drift report
│   └── context_rot_opus47_analysis.md  # Methodology + findings writeup
├── docs/
│   ├── methodology.md             # Citable statistical methodology (limitations-forward)
│   └── design/agentic-drift.md    # Design doc: tool-use drift (STRATEGY P2, unbuilt)
├── observatory/
│   └── panel.yaml                 # Observatory panel: endpoints, suites, cost cap
├── tests/
├── .github/
│   ├── actions/rift-drift-check/  # Reusable composite action (drift gate in CI)
│   ├── scripts/observatory_selftest.sh  # Monthly per-endpoint selftest refresh
│   └── workflows/
│       ├── publish.yml            # Build + PyPI publish on GitHub Release (OIDC)
│       └── observatory.yml        # Weekly panel → data branch commit → Pages deploy
├── pyproject.toml
├── README.md
├── STRATEGY.md                    # Product strategy: positioning, pivot, roadmap
└── CLAUDE.md
```

## Core Concepts

- **Suite**: A YAML file defining a set of eval cases (input, expected output, scoring method)
- **Provider**: An LLM API endpoint (Anthropic, OpenAI, or custom)
- **Run**: One execution of a suite against a single model
- **Comparison**: Statistical analysis of two runs (baseline vs challenger)
- **Drift Score**: Per-task and aggregate metric quantifying behavioral change
- **RiftLM** (`rift lm train|sample|suite`, model string `riftlm:<ckpt>.npz`):
  Rift's own tiny character-level GPT — pure numpy, hand-written backprop
  (finite-difference-checked in tests), trains from scratch on CPU in minutes
  on synthetic string tasks (cpy/rev/srt/max). Training shifts the task mix at
  `--switch` (rev dropped) and saves `riftlm-a.npz` / `riftlm-b.npz`, a
  baseline/challenger pair with a real subgroup regression — the keyless
  end-to-end demo of the whole pipeline against a model whose weights and data
  you fully control. The eval suite (`suites/riftlm.yaml`) is held out by a
  content-hash split (`is_eval_line`): the training sampler rejects exactly the
  lines the suite generator draws. `resolve_model` bakes the checkpoint's
  sha256 digest into the model string (`riftlm:models/riftlm-a.npz@<digest>`),
  so retraining in place invalidates the completion cache and the digest doubles
  as `provider_fingerprint`. Inference is in-process (`providers/riftlm.py`);
  no key, no network, $0 cost.
- **Replication / noise floor** (`--trials k`): re-sample each case k times to
  estimate run-to-run generation noise. `comparator.variance_components`
  decomposes scores into within-case (noise) vs between-case (signal) + ICC +
  a noise floor; a drift delta within ~2× that floor may not survive a re-run.
- **Self-test / null calibration** (`rift selftest`): compare a model to
  *itself* across trials and report the empirical false-positive rate of the
  drift gate (esp. the false-*regression* rate the CI gate exit-1s on). The
  credibility artifact: a green gate means little if it's red on unchanged
  models. See `selftest.py`.
- **Model fingerprint**: server-reported version (`Completion.provider_fingerprint`
  — OpenAI `system_fingerprint`, Gemini `modelVersion`, the resolved dated
  `model` Anthropic/OpenAI echo back). Captured on every completion, persisted
  through the cache, stamped into run metadata. Closes the integrity hole where
  a cache keyed on the request alone masks a silent server-side weight swap.
  `compare`/`diff` flag alias collisions (both sides → one fingerprint) and
  mid-run rollouts (>1 fingerprint in a run).
- **Judge validation** (`rift validate-judge`): the faithfulness articulation
  judge is scored against a committed, balanced human gold set
  (`judge_validation.py`); reports Cohen's kappa (`comparator.cohens_kappa`),
  not bare accuracy. Cite kappa alongside any faithfulness number.
- **Pre-registration** (`compare --preregister spec.yaml`): pin ONE primary
  endpoint (accuracy | cost_per_correct), direction, alpha, min_cases before
  the run. The headline + exit code bind to it; everything else is exploratory.
  The clean multiplicity defense — designate one test, don't correct twenty.
  See `preregistration.py` and `examples/preregistration_example.yaml`.
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

- **Observatory** (`rift observe` / `rift observatory-site`): scheduled
  longitudinal monitoring of live endpoints. Each pass runs a fixed panel
  (suites + sycophancy probe; calibration/refusal derived offline from the
  same outputs), appends strip-io records + a compact `index.jsonl` line per
  (date, endpoint, suite) to an append-only data dir (the orphan
  `observatory-data` branch in CI), then pairs each observation against the
  previous one via `compare_runs` with BH pooling across the whole pass.
  Event kinds in `events.jsonl`: `score_drift` (BH-significant),
  `silent_swap` (fingerprint changed, scores held), `fingerprint_change`,
  `rollout`, `panel_changed` (suite hash mismatch — pairing restarts, no
  bogus comparison), `notice` (probe shifts ≥10pp, never gated). Paired
  tests exclude cases errored on either side (outage ≠ drift) and skip
  majority-errored (`aborted`) records. `panel_version` = sha256 over
  `(input, expected)` pairs; budget guard = stage-level pre-flight estimates
  vs `max_cost_usd` (prior observation's tokens when available, else
  chars/4 + 300/case). Anything needing raw output text (confidence parse,
  refusal flags, sycophancy flip vectors) is computed pre-strip into the
  record's `derived` block. Replay mode (`--from-runs run.json ...`) builds
  records from saved RunResults, keyless — a run whose suite ends in
  `__pushback` pairs as the sycophancy follow-up. The site renderer
  (`observatory_site.py`) follows demo.py's hand-rolled-SVG idiom: zero JS,
  fingerprint-change markers as dashed verticals, selftest false-regression
  rate cited in the footer. See `observatory/panel.yaml` and
  `.github/workflows/observatory.yml` (weekly cron + monthly selftest
  refresh + Pages deploy).

## CLI Interface

```bash
# Compare two models on a suite
rift compare --baseline claude-3-5-sonnet --challenger claude-sonnet-4 --suite summarization

# Run a single model and save results
rift run --model gpt-4o --suite extraction --output results/gpt4o_extraction.json

# Compare two saved runs
rift diff results/run_a.json results/run_b.json

# Re-render a saved comparison (from compare --output): terminal, full
# markdown report, or a one-page HTML executive "upgrade brief"
rift report results/comparison.json --format markdown --output drift_report.md
rift report results/comparison.json --format brief --output brief.html

# Import suites from other harnesses (Rift becomes their statistics layer)
rift import --from promptfoo promptfooconfig.yaml -o suites/imported.yaml
rift import --from inspect samples.jsonl -o suites/imported.yaml
rift import --from lm-eval task.yaml --dataset docs.jsonl -o suites/imported.yaml
rift import --from openai-evals samples.jsonl -o suites/imported.yaml

# Observatory: one panel pass + render the dashboard
rift observe --panel observatory/panel.yaml --data-dir observatory-data
rift observatory-site --data-dir observatory-data --out _site

# RiftLM: train the built-in tiny GPT, then catch its manufactured regression
rift lm train
rift compare --baseline riftlm:models/riftlm-a.npz \
             --challenger riftlm:models/riftlm-b.npz \
             --suite riftlm --subgroup task:
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
  `sycophancy`/`discover`/`faithfulness`/`selftest`/`validate-judge`/
  `observe` in live mode) DO preflight keys via `ensure_provider_keys`
  for a clean fail-fast prompt, so they require a key even when a run
  would have been fully cached. A fatal, user-fixable `ClickException`
  raised lazily anywhere — a missing key (e.g. an llm_judge judge key),
  an unreadable RiftLM checkpoint — is re-raised by `run_suite` rather
  than swallowed as a per-case error, so it always shows the clean
  message instead of an all-errored "run" drift stats would be computed
  over.
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
- Cost claims must name their serving configuration. `pricing.py` is
  standard-mode list price only; providers also sell the SAME model at
  other prices (Anthropic: Batch −50%, fast mode premium, cache reads
  0.1×, inference_geo 1.1× — and these change which model is "the
  expensive one": Opus 4.8 *fast* = Fable 5 standard; batched Fable ≈
  live Opus 4.7). Any published cost comparison must state which cell
  of that matrix its $ figures assume and, when the headline is a cost
  multiple, situate it against the adjacent configurations a reader
  could buy instead. Same for model *configuration*: thinking/effort
  defaults differ per model (e.g. omitting `thinking` = off on Opus
  4.7, always-on for Fable 5), so disclose what each side actually ran
  with. See benchmarks/fable5_vs_opus47/analysis.md ("The price in
  context" + the provenance configuration note) for the template.
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
- Exit-code contract (the CI surface): 0 = no significant regression;
  1 = significant regression detected (the gate — ONLY this); 2 =
  operational error (unknown model/suite, malformed file, missing API
  key, all cases errored). Operational failures must never exit 1 — a
  CI job treating 1 as "regression" would misclassify infrastructure
  problems as drift. `compare` refuses to compute a verdict (exit 2)
  when every case on either side errored.
- Model strings: provider prefixes (claude/gpt-/o1/o3/o4/gemini),
  aliases, `riftlm:<ckpt>.npz`, or `<model>@<base-url>` for any
  self-hosted OpenAI-compatible server (vLLM, Ollama, llama.cpp).
  Anything else resolves to the lazy `local` pseudo-provider: cached
  runs and replays stay keyless, but a live call raises
  `UnknownModelError` (exit 2) at provider-construction time.
- CI levels follow alpha: `compare_runs(alpha=...)` computes both the
  accuracy-delta CI and the $/correct CI at the `1 − alpha` level and
  stamps `DriftResult.ci_level`; renderers label the interval by its
  real level. A pre-registered `alpha: 0.01` therefore gets genuine
  99% intervals (and `preregistration.evaluate` records a violation if
  the levels don't match).
- Statistical honesty notes renderers must respect: `delta_pct` is
  `None` (omit, don't print +0.0%) when the baseline mean is 0;
  `test_used == "insufficient_data"` (n < 2) is never significant and
  subgroup tables render it as untestable; when the bootstrap CI and
  the exact test disagree in significance, reports append a one-line
  footnote (different procedures; the p-value governs). See
  docs/methodology.md for the full caveats.
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
(`compare`/`run`/`matrix`/`sycophancy`/`discover`/`faithfulness`/
`selftest`/`validate-judge`/`observe` in live mode)
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
