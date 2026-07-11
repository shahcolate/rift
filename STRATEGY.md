# Rift Product Strategy

*Where Rift is going, why, and what gets built next. Last updated 2026-07-10.*

## One-liner

**Rift is becoming the public record of model behavior over time.**
The CLI stays the engine; the [Observatory](#the-pivot-the-rift-observatory)
makes its output a continuously-growing, citable, statistically-gated
history of how production LLM endpoints actually behave.

## Reading the market honestly

"Compare two models on an eval suite" is a commoditized surface. promptfoo,
Braintrust, LangSmith, DeepEval, and every model vendor's own eval tab all
live within one feature of it. Competing on that frame means competing on
integrations and UI polish — a losing game for an open-source tool with a
statistics-first identity.

What is **not** commoditized is the combination Rift already ships:

1. **A real statistics layer.** Paired McNemar / t-test / bootstrap CIs,
   effect sizes, Benjamini–Hochberg FDR across multi-test reports, post-hoc
   power and minimum-detectable-effect, pre-registered primary endpoints,
   and — uniquely — `rift selftest`, a gate that publishes its own
   false-positive rate under the null. Most eval tools ship *zero*
   statistics; none ship self-calibration.
2. **A behavioral safety panel.** Reasoning-faithfulness probes
   (hint-articulation and CoT-dependence), sycophancy flip rate under
   pushback, calibration drift (Brier/ECE), refusal/over-refusal drift, and
   context-rot stress testing. These map directly onto the questions AI-lab
   safety teams publish about — measured here on production endpoints.
3. **Fingerprint capture / silent-swap detection.** Every completion
   records the server-reported model version; runs flag alias collisions
   and mid-run rollouts. No other eval tool can tell you the model changed
   *underneath* your eval.
4. **Cost-per-correct with confidence intervals.** The procurement number,
   with error bars — `$/correct` deltas from a paired bootstrap, not
   list-price arithmetic.

Each is defensible alone. Together they answer a question nobody else can:
*"did the model behind this endpoint change, in what direction, at what
cost, and how sure are we?"*

## The pivot: the Rift Observatory

A CLI you run once produces a report. A panel you run **every week against
live endpoints** produces a dataset nobody can retroactively recreate.

The Observatory (`rift observe` + `rift observatory-site` +
`.github/workflows/observatory.yml`) runs a fixed behavioral panel —
accuracy suites, the sycophancy probe, calibration and refusal
classification — against pinned endpoints on a weekly cron, appends every
observation to an append-only data branch, and publishes a static
dashboard with a dated **drift feed**:

- **`score_drift`** — this week's scores differ from last week's, by the
  same paired test the CI gate uses, pooled through BH correction across
  the whole panel. No cherry-picking by construction.
- **`silent_swap`** — the server fingerprint changed and the scores held.
  The model was replaced under the alias and an accuracy-only check would
  never have noticed.
- **`rollout`** — the served snapshot changed *mid-run*.
- **`notice`** — probe metrics (flip rate, ECE, refusal rate) moved past a
  threshold; reported, never gated.
- Every verdict is published next to the gate's empirical false-regression
  rate from `rift selftest` — the reader can weigh the alarm against how
  often the alarm goes off for no reason.

### Why this is the right bet

- **Novel.** "Did gpt-5.5 quietly change this week?" has no citable answer
  anywhere today. Reddit threads claim models get nerfed; vendors say
  nothing; nobody publishes continuous, statistically-gated behavioral
  monitoring of production endpoints.
- **Defensible.** The moat is the time series. A competitor can fork the
  code in an afternoon; they cannot fork eighteen months of weekly
  observations with fingerprint provenance. Every week the gap widens.
- **High-impact.** Each "the model got worse" controversy becomes
  confirmable or debunkable with data. Journalists, researchers, and
  procurement teams get a primary source. Real findings (like the
  Opus 4.7 → 4.8 context-rot regression already in `benchmarks/`) become a
  publishable series, not one-off blog posts.
- **Cheap to run.** The panel is budget-capped (~$1–2/pass at list
  pricing, hard cap `$3`) and degrades gracefully: provider outages and
  budget aborts record partial data rather than losing the week.

## Three audiences, one artifact

| Audience | What they come for | What Rift gives them |
|---|---|---|
| **Researchers** | Behavioral drift data + methodology | The Observatory time series, the probe panel, pre-registration, judge validation with Cohen's κ — everything cited with CIs and null calibration |
| **Engineering teams** | "Don't let an upgrade break prod" | The GitHub Action drift gate, `rift compare` exit codes, completion caching, metrics export (JSON/Prometheus) |
| **PMs & executives** | A defensible go/no-go and a budget number | `$/correct` with CIs, the `rift demo` executive memo, the dashboard's plain-language drift feed, "you're paying for a model you can't verify you're getting" (fingerprints) |

The positioning rule: **research-grade rigor is the brand; the CI gate is
the adoption engine; the exec-readable artifact is the wedge into
companies.** Each pillar feeds the others — Observatory findings earn the
credibility that makes teams trust the gate, and gate users supply the
demand for more panel coverage.

## Roadmap

**P1 — compounding the moat**
- Run the Observatory continuously; keep the data branch unbroken. The
  series is the product. *(ongoing — the only P1 item that never
  completes)*
- ~~Suite adapters~~ **Shipped**: `rift import --from
  promptfoo|inspect|lm-eval|openai-evals`. Teams keep their existing
  evals; Rift is the statistics layer on top. Conversion is loud about
  loss (warnings + caveats recorded in the emitted suite), and the
  `import rift` library API covers the no-conversion path —
  `rift.compare_runs` over any harness's paired scores.
- ~~Publish the methodology~~ **Shipped**: `docs/methodology.md`, a
  limitations-forward technical treatment of the drift gate (paired
  tests + BH + selftest null calibration + pre-registration), including
  the known caveats an external statistician would find. The Observatory
  findings report remains open until the series is months deep.

**P2 — widening the panel**
- Agentic / tool-use drift: **design doc shipped**
  (`docs/design/agentic-drift.md`) — trajectories collapse to a scalar
  per case so the existing paired machinery applies; needs provider
  tool-call surface before implementation. The frontier moved to
  agents; drift detection should follow.
- ~~Exec report mode~~ **Shipped**: `rift report --format brief` renders
  a one-page "model upgrade brief" (rules-based verdict, `$/correct`
  delta with CI, risk flags) from any saved comparison, in the demo
  memo's visual language.
- Dated-snapshot twins in the panel (alias vs. pinned date for the same
  family) so alias re-pointing is measured directly, not inferred.

**P3 — distribution**
- ~~Drift-feed RSS~~ **Shipped**: the Observatory site emits `feed.xml`
  with re-render-stable ids. Webhooks remain open (panel-config field +
  best-effort POST, out of the critical path).
- More CI integrations (GitLab, Jenkins) and observability sinks
  (Datadog, W&B) as demand shows up.

## What to lead with (proof points)

1. **A real catch:** Opus 4.7 → 4.8 regressed −18.75pp on context-rot
   reasoning (McNemar p=0.031, $/correct +26%) while looking like a
   sidegrade on five standard suites
   ([`benchmarks/3way_opus48/analysis.md`](benchmarks/3way_opus48/analysis.md)).
2. **A gate that audits itself:** `rift selftest` publishes the
   false-regression rate on an unchanged model — the number every other
   eval gate hides by never measuring it.
3. **Silent-swap detection:** fingerprints on every completion; the
   Observatory turns them into a public changelog of when the model behind
   an alias actually changed.
4. **Statistical honesty as a feature:** pre-registration, BH correction,
   effect sizes, power analysis, and CIs on every number — including the
   cost numbers.

## Principles (unchanged)

- Every drift claim ships with a confidence interval.
- The gate's own error rate is measured and published, not assumed.
- Evals are data (YAML), not code; suites are importable, not bespoke.
- Offline-first: demos, replays, and tests run keyless and reproducible.
- Provenance is loud: synthetic data is labeled synthetic, custom prompts
  and judges are stamped into run metadata, fingerprints are recorded.
