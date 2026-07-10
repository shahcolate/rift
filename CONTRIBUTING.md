# Contributing to Rift

Thanks for your interest in improving Rift — the "you upgraded your model,
what broke?" drift detector. This guide covers how to get a dev environment
running, the conventions we hold the codebase to, and how to land a change.

Rift's whole pitch is statistically honest answers about model behavior, so
the bar for contributions is the same bar we hold the reports to: no vibes,
backed claims, reproducible results.

## Code of Conduct

Be kind, assume good faith, and keep discussion technical. Harassment or
personal attacks aren't welcome. Maintainers may remove comments, commits, or
contributors that don't meet this standard.

## Ways to contribute

- **Report a bug** — open a [Bug report](https://github.com/shahcolate/rift/issues/new?template=bug_report.yml).
- **Request a feature** — open a [Feature request](https://github.com/shahcolate/rift/issues/new?template=feature_request.yml).
- **Contribute an eval suite** — new `suites/*.yaml` that exercise a real
  drift mode are very welcome (see [Adding a suite](#adding-an-eval-suite)).
- **Improve docs** — README, `CLAUDE.md`, docstrings, this guide.
- **Fix or extend code** — providers, scorers, statistics, reporters.

If you're planning a non-trivial change, please open an issue first so we can
agree on the approach before you invest the time.

## Development setup

Rift targets **Python 3.11+**.

```bash
git clone https://github.com/shahcolate/rift.git
cd rift

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Verify the install with the keyless demo — it replays recorded outcomes, so it
needs no API keys:

```bash
rift demo
```

API keys are only needed for live commands (`compare`, `run`, `matrix`,
`faithfulness`, …). Set them via environment variables
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`) or run
`rift setup` to write `~/.rift/.env`. Never commit keys.

## Before you open a pull request

Run the same checks CI runs (see `.github/workflows/ci.yml`). All three must
pass on Python 3.11 and 3.12:

```bash
ruff check src/ tests/   # lint
pyright src/             # type-check
pytest tests/ -q         # tests
```

CI also smoke-tests the keyless demo (`rift demo`), so keep that path working
without network or keys.

### Conventions

- **Async throughout.** Eval execution is async for parallelism — follow the
  existing patterns in `runner.py` rather than introducing blocking I/O.
- **Type hints everywhere.** `pyright src/` is enforced; new code should be
  fully typed.
- **Line length 100**, enforced by ruff (`pyproject.toml`).
- **Statistically honest.** Every drift claim ships with a confidence
  interval and the test used. Don't surface a delta without its CI, and don't
  drop the per-case `input_tokens` / `output_tokens` / `cost_usd` fields — the
  cost-normalized metrics depend on them. See the "Key Implementation Notes"
  in [`CLAUDE.md`](CLAUDE.md).
- **Suites are data, not code.** Prefer expressing evals as YAML over Python
  where possible.
- **Provider-agnostic.** Keep provider-specific logic behind the
  `BaseProvider` abstraction in `src/rift/providers/`.

### Tests

- New behavior needs tests. Unit tests live in `tests/`; full subprocess
  end-to-end tests live in `tests/e2e/` and drive the installed `rift` binary
  against a seeded cache (no network).
- Tests must not hit the network or require API keys. Use the cache / recorded
  outcomes patterns the existing tests follow.

## Adding an eval suite

Suites live in `suites/*.yaml`. A suite declares a scoring method and a list of
cases:

```yaml
name: my_suite
description: What drift mode this exercises
scoring: exact_match   # or fuzzy_match, semantic, llm_judge, exec_tests, custom
cases:
  - input: |
      ...
    expected: ...
```

See the existing suites and the "Config Format" section of
[`CLAUDE.md`](CLAUDE.md) for the full schema, including overridable probe
prompts (`prompts:` / `cues:`) and custom scorers. If your suite is meant to
back a published benchmark, make it reproducible in `--mode record` against a
committed outcomes file under `benchmarks/`.

## Pull request process

1. Fork the repo and create a topic branch off `main`.
2. Make focused commits with clear messages (imperative mood, e.g.
   "Add Gemini retry backoff"). Keep unrelated changes in separate PRs.
3. Run lint, type-check, and tests locally (see above).
4. Update docs when behavior changes — README, `CLAUDE.md`, and a
   `CHANGELOG.md` entry under an `## [Unreleased]` heading following
   [Keep a Changelog](https://keepachangelog.com/).
5. Open the PR with a description of *what* changed and *why*, and link the
   issue it closes.
6. Make sure CI is green. A maintainer will review; please respond to feedback
   by pushing follow-up commits.

By contributing, you agree that your contributions are licensed under the
project's [MIT License](LICENSE).
