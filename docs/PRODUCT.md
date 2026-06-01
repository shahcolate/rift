# Rift as a product — from CLI to platform

> "You upgraded your model. What broke? You're picking a vendor. Who actually
> wins?"

This doc frames Rift as a product and sequences the work from the open-source
CLI (done) through the self-hosted control plane (this MVP) to a hosted,
multi-tenant platform.

## The wedge

Model vendors ship new versions constantly, with no changelog for *behavior*.
Teams running LLMs in production discover regressions the expensive way — in
prod, or on the invoice. Existing evals are either vibes ("feels dumber") or
leaderboard accuracy that ignores cost and paired structure.

Rift's edge is **statistical rigor on the dimensions that actually decide a
migration**:

- **Cost-normalized drift (`$/correct`)** — catches silent tokenizer inflation
  that leaves accuracy flat but raises the bill (the demo: +35% `$/correct`,
  neutral accuracy, on an Opus point upgrade).
- **Paired significance** — McNemar / paired-t + bootstrap CIs, BH correction
  across an NxN matrix, post-hoc power. Every claim ships a confidence interval.
- **Behavioral probes beyond accuracy** — refusal, sycophancy, calibration, and
  reasoning *faithfulness* (does the stated reasoning reflect what drove the
  answer?).

The CLI proves the engine. The product is **continuous**: watch the models a
team depends on, and tell them — before prod does — when one drifts.

## Where we are

| Layer | Status |
|-------|--------|
| Eval engine (runner, comparator, scorers, probes) | ✅ shipped (v1.0.0, PyPI) |
| Reproducible benchmarks + publishable reports | ✅ shipped |
| CI drift gate (composite GitHub Action) | ✅ shipped |
| **Self-hosted control plane (`rift serve`)** | ✅ **this MVP** |
| Hosted, multi-tenant SaaS | ⏳ roadmap (below) |

### The MVP: `rift serve` (self-hosted, single-tenant)

A thin orchestration + persistence layer over the existing engine
(`run_suite` → `compare_runs` → `observability`). No eval logic is
reimplemented — a monitored run and a `rift compare` produce identical numbers.

- **Monitors** — a standing comparison (suite + baseline + challenger + cron).
- **Scheduler** — APScheduler runs each monitor on its cron; failures are
  recorded, not fatal.
- **History** — SQLite store of every run (drift + both run blobs) and alert.
- **Dashboard** — server-rendered HTML; drift and `$/correct` time-series as
  dependency-free inline SVG.
- **API + `/metrics`** — JSON REST plus a Prometheus endpoint reusing the exact
  metric shape of `rift compare --metrics-out`.
- **Alerts** — Slack-compatible webhook on any regression, where "regression"
  means a significant accuracy drop **or** a significant `$/correct` increase.

Code: `src/rift/server/` (`store`, `service`, `scheduler`, `alerts`, `app`,
`charts`, templates/static). Launch with `rift serve` (extra: `rift-eval[server]`).

The MVP is deliberately built with clean seams for what comes next:

- `store.py` takes `db_path` everywhere and returns plain dicts — the single
  function boundary a multi-tenant store would replace.
- Per-monitor `cache_dir` and the `cache_root` app setting reserve completion-
  cache isolation.
- Provider keys are read from the environment per run — the injection point a
  per-tenant key vault would replace.

## Roadmap to a hosted platform

Each phase is independently shippable and leaves the prior phase working.

### Phase 1 — Multi-tenant + auth
- Orgs/projects; login + API tokens.
- Per-tenant provider-key vault (encrypted at rest), injected per run.
- Per-tenant completion-cache and storage isolation (the `cache_root` seam).
- Postgres behind the same `store` interface; SQLite stays the local default.

### Phase 2 — Managed runners + scale
- Move execution off the web process to a task queue (Celery/RQ/Arq) and a pool
  of runner workers; the scheduler enqueues instead of running inline.
- Concurrency, rate-limit, and budget controls per org.
- Backfills and ad-hoc comparisons as first-class jobs.

### Phase 3 — Product surface
- Email/Slack digests; regression triage with diff drill-down on the cases that
  moved.
- A hosted, versioned suite library (reasoning, extraction, refusal, context-rot,
  faithfulness) teams can subscribe to and pin.
- Vendor "what changed" reports the day a new model ships.

### Phase 4 — Business
- Usage metering (runs × cases × tokens) and billing.
- SSO/SAML, audit log, data-retention controls for enterprise.
- `--strip-io` already lets proprietary suites ship results without prompts;
  extend to a full bring-your-own-suite privacy story.

## Business model (sketch)

- **Open-source CLI + self-hosted control plane** — free; drives adoption and
  trust (the stats are auditable, the suites are data).
- **Hosted platform** — seat + usage based; the value is *not running it
  yourself* plus the managed suite library and vendor-change reports.
- **Enterprise** — SSO, audit, on-prem/VPC, contracted-pricing modeling (the
  enterprise multiplier already exists in `pricing.py`).

## Why this can be a YC company

The hard part — credible, paired, cost-aware drift statistics plus novel
behavioral probes — is already built and tested. The remaining work is product
infrastructure on top of a proven engine, sequenced above. The market is every
team that ships on someone else's model and can't afford to find out about a
regression from a customer or an invoice.
