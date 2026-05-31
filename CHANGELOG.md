# Changelog

All notable changes to Rift are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[1.0.0]: https://github.com/shahcolate/rift/releases/tag/v1.0.0
[0.1.1]: https://github.com/shahcolate/rift/releases/tag/v0.1.1
