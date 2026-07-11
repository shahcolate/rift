# Changelog

All notable changes to Rift are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Frontier head-to-head benchmark** (`benchmarks/frontier/`): a
  pre-registered, reproducible harness for measuring two frontier
  models against each other (default: Claude Fable 5 vs GPT-5.6 Sol).
  Primary endpoint is pooled accuracy on the judge-free panel
  (`reasoning` + `extraction` + `hard_reasoning`), declared in
  `preregistration.yaml` before the first capture; per-suite drift,
  $/correct, and token profile are exploratory. Live mode has a
  cost-cap pre-flight; replay mode rebuilds the full report (including
  the pre-registration verdict) from committed run JSONs, keyless. A
  `workflow_dispatch` GitHub Actions workflow
  (`frontier-benchmark.yml`) runs the live capture with repo secrets
  and commits the results for replay.
- **OpenAI GPT-5.6 family support** (launched 2026-07-09): pricing
  entries for `gpt-5.6-sol` ($5/$30 per Mtok), `gpt-5.6-terra`
  ($2.50/$15), and `gpt-5.6-luna` ($1/$6), plus aliases — bare
  `gpt-5.6`/`gpt-5-6` pin to `gpt-5.6-sol` (matching OpenAI's own
  routing of the bare alias, but explicit in run metadata), and dashed
  filename-safe forms for all three tiers. Prices are launch list
  rates as reported at release; verify against the OpenAI pricing page
  before publishing cost comparisons.

## [1.1.0] - 2026-07-10

The self-auditing release: everything shipped between 1.0.0 and here —
the drift gate that measures its own error rate, the Observatory, RiftLM,
suite importers, and a statistical-audit hardening pass.

### Added
- **Suite importers** (`rift import --from promptfoo|inspect|lm-eval|openai-evals`).
  Convert existing evals into Rift suites so Rift becomes the statistics
  layer over harnesses teams already use. Conversion is conservative and
  loud about loss: every dropped assertion or flattened chat is warned and
  recorded in the emitted suite's description; bundled binary scorers
  (`rift.adapters.scorers`) make emitted suites self-contained.
- **`rift report`** — re-render a saved `compare --output` payload keyless:
  terminal, full markdown, or a one-page HTML/markdown **model upgrade
  brief** with a rules-based verdict that claims only what the statistics
  support.
- **Public library API.** `import rift` now exposes 18 stable, lazily
  loaded symbols (`load_suite`, `run_suite`, `compare_runs`,
  `DriftResult`, `power_analysis`, ...) with a semver commitment; a
  `py.typed` marker ships types to downstream checkers.
- **Self-hosted endpoints**: `<model>@<base-url>` runs any
  OpenAI-compatible server (vLLM, Ollama, llama.cpp).
- **Observatory RSS feed** (`feed.xml`) with re-render-stable guids;
  pages advertise it via `<link rel=alternate>`.
- **Self-audit suite** (from #48): `rift selftest` (null calibration of
  the gate), `--trials` replication + noise floor, server-fingerprint
  capture with silent-swap detection, judge validation with Cohen's
  kappa, pre-registered primary endpoints.
- **Rift Observatory** (#49): `rift observe` + `rift observatory-site` +
  weekly scheduled workflow — longitudinal, statistically-gated
  monitoring of live endpoints with an append-only public record.
- **RiftLM** (#57): `rift lm train|sample|suite` — a pure-numpy tiny GPT
  that trains on CPU in minutes and manufactures a real regression, so
  the whole pipeline runs keyless against a model whose weights you own.
- **Docs**: `docs/methodology.md` (citable, limitations-forward
  statistical methodology) and `docs/design/agentic-drift.md` (design doc
  for tool-use drift). LICENSE, SECURITY.md, CODE_OF_CONDUCT.md,
  CITATION.cff.

### Changed
- **Exit-code contract**: 1 is now reserved for the drift gate
  (significant regression); all operational errors (unknown model/suite,
  malformed files, missing keys, all-errored runs) exit 2. `compare`
  refuses to compute a verdict when every case on a side errored.
- **CI levels follow alpha**: accuracy and $/correct CIs are computed at
  `1 − alpha` and labeled by their real level (`DriftResult.ci_level`);
  a pre-registered `alpha: 0.01` gets genuine 99% intervals.
- Unknown models fail at provider-construction time with a clean remedy
  instead of producing an all-errored run that could exit 0.

### Fixed
- Statistical audit: a single-case comparison could report `p=0.0,
  significant=True` (now `insufficient_data`, never significant); n=0
  comparisons leaked NaN; `variance_components`' ICC was biased upward
  (now the ANOVA ICC(1) estimator, so pure noise reads ≈0); Cohen's h
  docstring cited wrong magnitude thresholds.
- Engine audit: `pricing.lookup` billed named submodels at family rates
  (`gpt-4o-mini` at `gpt-4o` prices); derived suites (sycophancy
  pushback, context-rot expansion) dropped pinned judge/embedding models
  and prompt overrides, confounding flip rates with grader drift;
  network scorers had no retry and one judge 429 aborted a whole run
  (now retried, then recorded per-case); OpenAI structured refusals
  parsed as empty output and were missed by refusal drift; aliased
  proposer models reported $0 discovery spend; confidence-stripping
  could delete a non-trailing line of the answer.
- Observatory audit: `silent_swap` events no longer overclaim "scores
  held" for merely non-significant panels; the budget guard estimates
  unpriced models at the catalog maximum instead of $0 (the cap was a
  no-op exactly when prices were unknown); budget-cap skips are recorded
  and reported; a truncated `events.jsonl`/`index.jsonl` tail no longer
  wedges every future pass; the Pages job re-grants `contents: read`.

## [1.0.0] - 2026-05-31

First stable release. Rift now covers the full "you upgraded your model — what
broke?" workflow end to end, with a stable CLI, data-driven suites, and
publishable, statistically-backed reports.

### Added
- **Custom scoring functions** (`scoring: custom`). A suite supplies its own
  scorer via `custom_scorer: "target:callable"` — an importable module
  (`mypkg.scorers:score`) or a file path (`./scorer.py:score`, resolved
  relative to the suite file then the CWD). The target may be a sync
  `score(output, expected)`, an async `ascore(output, expected, context=None)`,
  or a `Scorer` class/instance. The chosen scorer is recorded in run metadata.
  (#37)
- **Observability metrics export.** `compare` and `run` gain `--metrics-out` +
  `--metrics-format {json,prometheus}` to emit a flat, stable set of named
  metrics for dashboards and time-series stores, distinct from the rich
  `--output` JSON. Metrics are written even when `compare` exits 1 on a
  regression, so CI can upload them on failure. (#40)
- **GitHub Action** (`.github/actions/rift-drift-check`) — a reusable composite
  action that wraps `rift compare`, writes the drift report to the job summary,
  and gates a PR on a significant regression via the exit code. (#41)
- **Real subprocess end-to-end tests** (`tests/e2e/`) that drive the installed
  `rift` binary through complete workflows against a seeded cache (no network).
  (#38)
- **User-overridable probe prompts** (`prompts:` / `cues:` per suite), validated
  at load and disclosed in published reports. (#36)
- **Embedding-based semantic scoring** (`scoring: semantic`, OpenAI + Google
  backends). (#35)
- **Reasoning-faithfulness probe** (`rift faithfulness --mode hint|cot|both`).
  (#30, #34)
- **Simple API-key onboarding** (`rift setup`, `~/.rift/.env`). (#28, #29)

### Changed
- `compare` and `matrix` now record the **resolved** model id (e.g.
  `claude-opus-4-7`) in the drift artifact, matching `run`/`diff` — so saved
  comparison and run JSONs agree on model naming. (#39)
- A malformed or missing suite now produces a clean CLI message and exit 1
  (`SuiteValidationError` / `SuiteNotFoundError`) instead of a raw traceback.
  (#38)
- `Development Status` classifier promoted to Production/Stable.

## [0.1.1] - 2026

### Added
- Pip-installable package (`rift-eval`) with bundled suites and a Trusted
  Publishing workflow. (#26)
- Faster CLI startup with a demo-loading spinner. (#27)

[Unreleased]: https://github.com/shahcolate/rift/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/shahcolate/rift/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/shahcolate/rift/releases/tag/v1.0.0
[0.1.1]: https://github.com/shahcolate/rift/releases/tag/v0.1.1
