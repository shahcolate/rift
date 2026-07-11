# Frontier head-to-head

A reproducible, **pre-registered** benchmark for comparing two frontier
models — by default Claude Fable 5 (baseline) vs OpenAI GPT-5.6 Sol
(challenger), the July 2026 flagship pair.

The whole point of pre-registration: the headline verdict is bound to
ONE primary endpoint declared in [`preregistration.yaml`](preregistration.yaml)
*before* the first capture — pooled accuracy on the frontier panel
(`reasoning` + `extraction` + `hard_reasoning`; 63 paired cases, all
binary-scored and judge-free, pairs errored on either side excluded).
Everything else the report prints (per-suite drift, cost per correct,
token profile, judged suites) is exploratory. No forking paths.

## Run it

### Option A — GitHub Actions (recommended)

The **Frontier benchmark** workflow (`workflow_dispatch`) runs the live
capture on a GitHub runner using the repo secrets `ANTHROPIC_API_KEY` /
`OPENAI_API_KEY`. Dispatch it from the Actions tab (or `gh workflow run
frontier-benchmark.yml`), optionally overriding the pair, trials, cost
cap, or the judged-suite opt-in. The report lands in the job summary,
the raw run JSONs + report are uploaded as an artifact and committed
back to the dispatching branch, so the capture is auditable and
replayable.

### Option B — locally

```bash
export ANTHROPIC_API_KEY=... OPENAI_API_KEY=...   # or: rift setup
python benchmarks/frontier/run_frontier.py --mode live
```

Useful flags: `--trials 3` (noise floor), `--judged` (adds
`open_ended_qa` — see the judge caveat below), `--max-cost 25`
(pre-flight refusal + stage-level cap), `--baseline/--challenger`
(any pair; the pre-registration flags identity mismatches as protocol
violations rather than silently rebinding).

### Reproduce a published capture (keyless)

```bash
python benchmarks/frontier/run_frontier.py --mode replay \
    --from-dir benchmarks/frontier/results/<date>
```

Rebuilds the full report — including the pre-registration verdict —
from the committed run JSONs. No API keys, no network.

## Methodology notes

- **Pooled primary.** The three panel suites are binary-scored, so the
  pooled vector gets McNemar's exact test (the pooling is declared in
  the pre-registration, not chosen after seeing results). With
  `--trials > 1` scores become trial means and the test switches to
  paired-t — the report says which ran (`test_used`).
- **Two-sided.** This is a cross-vendor comparison, not an upgrade
  gate, so the plan tests for a difference in either direction.
- **Errors ≠ drift.** Pairs errored on either side are excluded from
  the primary (same rule as the observatory) and the exclusion count is
  reported; `min_cases: 60` makes an outage-degraded capture dishonor
  the plan instead of confirming on a fragment. Exploratory per-suite
  tables keep `rift compare` semantics (errored cases score 0, warned).
- **Judged suites are opt-in and asymmetric.** `open_ended_qa` uses a
  pinned Anthropic judge (`claude-sonnet-4-6`). In a cross-vendor
  comparison the judge shares a vendor with one contestant, so judge
  bias does not cancel out the way it does in
  `benchmarks/fable5_vs_opus47`. Off by default; loudly disclosed when
  on.
- **Cost claims name their configuration.** All $ figures are
  standard-mode list price; the report's disclosure block covers batch
  / fast-tier / cached-input variants, Fable's ~30% heavier tokenizer,
  and its always-on thinking billed as output. Compare `$/correct`,
  never per-token rates.
- **This is a benchmark, not a CI gate.** The driver always exits 0 on
  a successful run. For gating, use
  `rift compare --preregister` (exit 1 is reserved for a confirmed
  regression there).
