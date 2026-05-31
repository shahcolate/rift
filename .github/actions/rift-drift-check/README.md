# Rift drift check — GitHub Action

Run a [Rift](https://github.com/shahcolate/rift) eval suite in CI comparing a
**baseline** and **challenger** model, and fail the job when a
statistically-significant regression is detected. The markdown drift report is
written to the job summary.

## Usage

```yaml
name: Model drift check
on:
  pull_request:
  workflow_dispatch:

jobs:
  drift:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4   # only needed if your suite YAML lives in the repo
      - uses: shahcolate/rift/.github/actions/rift-drift-check@v1.0.0
        with:
          baseline: opus-4-7
          challenger: opus-4-8
          suite: reasoning           # built-in name, or a path like ./suites/my.yaml
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          # OPENAI_API_KEY / GEMINI_API_KEY as needed by your models/judge.
```

The job **fails (exit 1)** when Rift finds a significant regression, gating your
PR. Set `fail-on-regression: false` to only surface the report without failing.

> **Version coupling.** Pin the action to a released tag (e.g.
> `…/rift-drift-check@v1.0.0`); the default `install-spec` (`rift-eval>=1.0.0`)
> then installs a Rift that supports the flags that action version uses. Using
> `@main` tracks unreleased changes — set `install-spec` to a matching git ref
> (`git+https://github.com/shahcolate/rift@main`) if you do.

## Inputs

| Input | Default | Description |
|-------|---------|-------------|
| `baseline` | _(required)_ | Baseline model identifier. |
| `challenger` | _(required)_ | Challenger model identifier. |
| `suite` | _(required)_ | Suite name or path to a suite YAML. |
| `alpha` | `0.05` | Significance threshold. |
| `fail-on-regression` | `true` | Fail the job on a significant regression. |
| `report-path` | `rift-drift-report.md` | Where to write the markdown report. |
| `metrics-path` | `""` | Write flat metrics here (empty = skip). |
| `metrics-format` | `json` | `json` or `prometheus`. |
| `cache-dir` | `.rift/cache` | Completion cache dir (cache it between runs to save cost). |
| `extra-args` | `""` | Extra raw args appended to `rift compare` (e.g. `--judge-model sonnet-4-6`). Whitespace-split into argv tokens — **not** shell-quoted, so don't pass values containing spaces. |
| `install-spec` | `rift-eval>=1.0.0` | What `pip install` installs. Pin exactly (`rift-eval==1.0.0`) for reproducible CI, or point at source (`git+https://github.com/shahcolate/rift@main`, `.`). Keep in lockstep with the action ref you pin. |
| `python-version` | `3.11` | Python to set up. |

## Outputs

| Output | Description |
|--------|-------------|
| `regression` | `'true'` if a significant regression was detected, else `'false'`. |
| `report-path` | Path to the written markdown report. |
| `metrics-path` | Path to the written metrics file (empty if not requested). |

## Caching completions

Rift caches model completions by `(model, params, input)`. Persist the cache dir
across runs so re-runs are cheap:

```yaml
      - uses: actions/cache@v4
        with:
          path: .rift/cache
          key: rift-cache-${{ hashFiles('suites/**') }}
      - uses: shahcolate/rift/.github/actions/rift-drift-check@v1.0.0
        with:
          baseline: opus-4-7
          challenger: opus-4-8
          suite: reasoning
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

## Uploading metrics

`compare` writes metrics even when it exits 1 on a regression, so the upload
runs on failure too (`if: always()`):

```yaml
      - uses: shahcolate/rift/.github/actions/rift-drift-check@v1.0.0
        id: drift
        with:
          baseline: opus-4-7
          challenger: opus-4-8
          suite: reasoning
          metrics-path: drift-metrics.prom
          metrics-format: prometheus
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: drift-metrics
          path: drift-metrics.prom
```
