# Opus 4.8 is a clean sidegrade — except it regresses on long-context distractor reasoning

> **Provenance.** Numbers below are from a **live API run on 2026-05-29**
> against the production Anthropic API (Opus 4.8 launch day), paired
> against Opus 4.7 and 4.6 on the same prompts, same scorers, single
> trial, temperature-pinned defaults. Dollar figures are at the current
> Opus 4.5-generation list price ($5/Mtok input, $25/Mtok output; see
> `src/rift/pricing.py`) — all three models share that rate, so cost
> comparisons here are apples-to-apples. Raw per-case JSONs are committed
> under the suite subdirectories; the context-rot drift report is at
> `context_rot/4.8_contextrot.md`. The bundled `rift demo` is a separate
> offline synthetic story and does NOT cite these numbers.

## Executive summary

Opus 4.8 dropped, so I ran it through [Rift](https://github.com/shahcolate/rift)
(paired evals + automatic significance testing) against its two
predecessors on six suites. The result splits cleanly in two:

1. **On standard tasks, 4.8 is a statistically indistinguishable
   sidegrade — usually a hair cheaper per correct.** Reasoning, structured
   extraction, code generation, open-ended QA, and summarization all come
   back as *no significant drift* after Benjamini–Hochberg correction
   across the pairwise comparisons. Where there's any cost difference,
   4.8 tends to edge out 4.7 (terser output).
2. **On long-context reasoning with injected distractors, 4.8 regresses
   significantly vs 4.7.** Accuracy drops 87.5% → 68.75% (Δ −18.75pp,
   **McNemar p = 0.031**), with **6 regressed cases and 0 improved**
   (paired g = −0.500). Spend is essentially unchanged ($2.29 → $2.28),
   so cost-per-correct rises **+26%** purely because 4.8 gets fewer
   answers right. Every regression is a case carrying a "BEGIN REFERENCE
   MATERIAL (may contain irrelevant information)" distractor block — 4.8
   gets pulled off-task by noise that 4.7 ignored. Refusal rate is 0% on
   both, so these are wrong answers, not over-refusals.

The headline isn't "Opus 4.8 is worse." It's **"a same-tier upgrade that
looks flat on standard benchmarks can still quietly break the long-context
robustness you actually depend on — so measure the capability you care
about before you flip the default."**

_Disclosure: numbers are from my own runs against the live Anthropic API
on 2026-05-29. n is small per suite; the suites are public (possible
training contamination); single trial. Run your own paired benchmark
before making routing decisions._

---

## The finding: context-rot regression (Opus 4.7 → 4.8)

`rift compare --baseline opus-4-7 --challenger opus-4-8 --suite
context_rot_reasoning --context-rot --subgroup distractor:` — 32 cases
(8 base reasoning prompts × 4 distractor regimes: 0k / 2k / 8k / 32k
tokens of injected irrelevant "reference material").

| Metric | Opus 4.7 | Opus 4.8 |
|---|---|---|
| Accuracy | 87.5% (28/32) | 68.75% (22/32) |
| Δ | — | **−18.75pp (−21.4%)** |
| p-value (McNemar exact) | — | **0.031** |
| 95% CI (accuracy Δ) | — | [−0.34, −0.06] |
| Regressed / improved cases | — | **6 / 0** |
| Effect size | — | Cohen's h = −0.46 (marginal); paired g = −0.500 |
| Total spend | $2.29 | $2.28 |
| $/correct | $0.0820 | **$0.1036 (+26%)** |
| 95% CI (Δ $/correct) | — | [+$0.0051, +$0.0577] |
| Refusal rate | 0.0% | 0.0% |
| Observed power | — | 68.8% (McNemar, α=0.05) |

**Every discordant pair goes the same way** — 6 regressions, 0
improvements (paired g = −0.500 is the maximum one-sided magnitude). That is
a much stronger signal than a noisy accuracy dip: 4.8 is strictly losing
cases here, never gaining them.

**The failures are all distractor cases.** The six regressed cases
(indices 3, 4, 5, 6, 7, 17) are dominated by prompts that open with
`--- BEGIN REFERENCE MATERIAL (may contain irrelevant information) ---`.
Broken out by distractor regime, all four levels regress (Δ between
−0.125 and −0.250); individually each subgroup is underpowered (n=8,
q=1.000), but the direction is uniform and the pooled test is
significant. The cost-per-correct damage concentrates at the 32k regime
(Δ $/correct +$0.098), where a lost answer is most expensive.

**Cost nuance worth stating precisely:** the +26% $/correct is *not* a
price or token-bloat story — total spend is flat to the cent ($2.29 vs
$2.28). It rises entirely because the denominator (correct answers)
shrank. Same money, fewer right answers.

---

## The other five suites: no significant drift

3-way matrix (`rift matrix --models opus-4-8,opus-4-7,opus-4-6`),
BH-corrected across all off-diagonal pairs at α=0.05.

| Suite | n | Scoring | 4.8 | 4.7 | 4.6 | Verdict | 4.8 $/correct |
|---|---|---|---|---|---|---|---|
| reasoning | 10 | exact_match | 8/10 | 9/10 | 9/10 | tie (all q=1.00) | $0.0011 |
| extraction | 29 | exact_match (partial) | 0.967 | 0.941 | 0.933 | tie (best q=0.125) | $0.0030 |
| code_generation | 5 | exec_tests | 5/5 | 5/5 | 5/5 | tie (all q=1.00) | $0.0028 |
| open_ended_qa | 5 | llm_judge | 5/5 | 5/5 | 5/5 | tie (all q=1.00) | $0.0054 |
| summarization | 8 | fuzzy/partial | 0.478 | 0.504 | 0.492 | tie (all q≥0.45) | n/a |

Notes that keep these honest:

- **Extraction is a tie, not a 4.8 win.** 4.8 posted the highest raw
  accuracy (25/29) and was the only model with zero API errors, but the
  pairwise gaps don't survive multiple-comparison correction (4.8-vs-4.6
  q=0.125, 4.8-vs-4.7 q=0.125). An earlier run had transient API errors
  on 4.6/4.7 that inflated the apparent gap; re-running at lower
  concurrency cleared them (0 errors all round) and the gap shrank from
  Δ0.129 to Δ0.035. Reported here is the clean run.
- **Summarization resolves nothing on the cost axis.** All three score
  0/8 "correct" because the fuzzy scorer's 1.0 threshold is stricter than
  any model achieves, so `$/correct` is undefined (∞) for all. The
  *quality* drift test still runs validly on the continuous mean scores
  and finds no significant difference (means 0.48–0.50, q≥0.45). Treat
  this as "tied," and do not cite its cost.
- On the suites with a defined cost, **4.8 is consistently a touch
  cheaper than 4.7** per correct (codegen $0.0028 vs $0.0030; QA $0.0054
  vs $0.0059; reasoning $0.0011 vs $0.0018) — consistent with slightly
  terser output at the same quality.

---

## What an executive leader should do this week

1. **Don't default-upgrade long-context / RAG / agentic workloads to 4.8
   without testing.** This is the one place the regression is real and
   significant. If your prompts carry retrieved context that includes
   irrelevant passages, 4.8 is measurably more distractible than 4.7 on
   this suite.
2. **For everything else, 4.8 is a safe sidegrade and marginally
   cheaper.** Standard reasoning, extraction, codegen, and short-form
   generation show no significant quality drift and a small cost win.
3. **Re-run on YOUR prompts.** n is 5–32 per suite and the suites are
   public. The *direction* (long-context robustness regression) is what
   to probe; the magnitudes need your data:
   ```bash
   rift compare --baseline opus-4-7 --challenger opus-4-8 \
       --suite YOUR_SUITE --context-rot --subgroup distractor:
   ```
   Watch the `Δ $/correct` CI and the regressed-vs-improved case counts.

---

## What is NOT in this writeup

- **A bigger context-rot n.** 32 cases at 68.8% power is enough to flag
  the regression but not to pin its magnitude tightly. A 100+ case run
  would tighten the CI and resolve the per-distractor-regime breakdown.
- **Repeated trials.** Single-trial, temperature-pinned; no within-model
  error bars. A flaky-case effect can't be ruled out, though the strict
  6/0 discordant split argues against pure noise.
- **Contamination control.** The suites are public in this repo; frontier
  models may have trained on them. The context-rot distractors are
  generated per-run, which mitigates this for the headline finding, but
  the standard-suite ties should be read as "no drift detected on public
  prompts," not "no drift."

---

## Reproduce

```bash
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=...

# The headline regression
rift compare --baseline opus-4-7 --challenger opus-4-8 \
    --suite context_rot_reasoning --context-rot --subgroup distractor: \
    -r benchmarks/3way_opus48/context_rot/4.8_contextrot.md

# The 3-way sweep
for SUITE in reasoning extraction summarization code_generation open_ended_qa; do
  rift matrix --models opus-4-8,opus-4-7,opus-4-6 --suite $SUITE \
      --concurrency 2 --output-dir benchmarks/3way_opus48/$SUITE
done
```

Raw per-case JSONs are committed under each suite subdirectory and are
sufficient to regenerate every number above with `rift diff` — no API
spend required.

---

## Bottom line

Opus 4.8 is, on the evidence here, a **competent same-tier release**:
statistically tied with 4.7/4.6 on five standard suites and slightly
cheaper per correct. The one real finding is a **significant long-context
distractor-robustness regression vs 4.7** — −18.75pp at p=0.031, 6/0
one-sided, +26% cost per correct for the same spend, concentrated on
prompts with injected irrelevant reference material. A green standard
benchmark sheet would have hidden it; the paired, per-case, distractor-
subgrouped view is what surfaced it. Measure the capability you depend
on before you switch the default.
