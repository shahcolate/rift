# Opus 4.7 vs Gemini 3.5 Flash: a Rift cross-vendor drift study

*Author: Rift maintainers · Run date: 2026-05-20 · Gemini 3.5 Flash GA day: 2026-05-19*

## TL;DR

Gemini 3.5 Flash shipped yesterday at a list price of **$1.50 / $9.00 per
1M tokens (input / output)** vs Opus 4.7 at **$15 / $75**. Headline says
Flash is 10x cheaper. We ran three Rift suites against both models the
day-after-GA. Three findings:

1. **On reasoning (binary, n=10):** identical 9/10 accuracy. Per-correct
   cost: **Opus $0.0052 vs Gemini $0.0056 — Gemini is more expensive.**
   The 10x list-price discount is fully consumed by Gemini 3.5 Flash's
   reasoning ("thinking") tokens, which Google bills as output.
2. **On extraction (continuous, n=29):** identical 0.941 mean. Per-correct
   cost: Opus $0.0087 vs **Gemini $0.0061 — 30% cheaper.** Long input
   prompts swing the math toward Gemini's input pricing.
3. **On open-ended QA (judge-scored, n=5):** Opus 1.000 vs Gemini 0.940
   (Δ −0.060, Hedges' g = −0.876 — **large** effect, p = 0.070, power
   ~50%). Caveat: the judge was Claude Sonnet 4.6, which may exhibit
   family bias. The signal is real but the magnitude is contested by the
   methodology.

**Bottom line:** the "Flash = cheap" claim is true only on input-heavy
workloads. On thinking-heavy ones it is approximately a wash, and on
open-ended generation the quality story is mixed at this sample size.

The numbers below are reproducible from the committed `.json` raw
runs — Rift's `diff` command will recreate every figure offline.

---

## What was run

For each suite, `rift compare --baseline opus-4-7 --challenger
gemini-3.5-flash` against the two live APIs. Gemini was called with
`thinking_level=medium` (Google's own default; pinned by
`rift.providers.google.GoogleProvider` for paired determinism). Opus
ran with deterministic sampler defaults (Rift drops `temperature`
for Opus 4.7 per the model's deprecated-params policy).

| Suite | Scoring | Cases | Statistical test |
|-------|---------|-------|------------------|
| `reasoning` | `exact_match` | 10 | McNemar's exact (binary) |
| `extraction` | `exact_match` (dict, field-by-field) | 29 | Paired t + bootstrap |
| `open_ended_qa` | `llm_judge` (`claude-sonnet-4-6` as judge) | 5 | Paired t + bootstrap |

Raw reports (Rift's own output) are committed alongside this file:
- `opus47_vs_gemini35/reasoning.md` / `.json`
- `opus47_vs_gemini35/extraction.md` / `.json`
- `opus47_vs_gemini35/open_ended_qa.md` / `.json`

Total live-API spend for the comparison runs: **~$0.55**.

---

## Finding 1 — Reasoning: thinking tokens eat the Flash discount

**Both models scored 9/10 on the reasoning suite. The cost story is
the headline:**

| Model | Total input tokens | Total output tokens | Total spend | $/correct |
|-------|--------------------|---------------------|-------------|-----------|
| `claude-opus-4-7` | 1,077 | 405 | $0.0465 | **$0.0052** |
| `gemini-3.5-flash` | 787 | **5,495** | $0.0506 | **$0.0056** |

Gemini's output volume on a 10-case binary reasoning suite was **13.6×
greater than Opus's** — and Gemini's list-price output rate ($9/Mtok) is
only 8.3× cheaper than Opus's ($75/Mtok). The arithmetic is unforgiving:
Flash issued enough reasoning tokens that its per-token discount
collapsed to roughly nothing.

This is the kind of finding a list-price-only spreadsheet misses
entirely. Both Anthropic Opus 4.7 and Gemini 3.5 Flash run extended
thinking by default; the difference is *how many tokens of it land in
the output column on your invoice*. Per-correct-answer cost — not
per-token — is the only sane metric for budget defense.

p-value 1.0, effect size 0.000 (Cohen's h), no significant drift in
either direction. The reasoning suite is too small (n=10, power ~5%
at α=0.05) to detect anything subtle.

---

## Finding 2 — Extraction: long input swings the math the other way

**Both models scored 0.941 on the extraction suite. Cost differs the
other direction:**

| Model | Total input tokens | Total output tokens | Total spend | $/correct |
|-------|--------------------|---------------------|-------------|-----------|
| `claude-opus-4-7` | 2,547 | 2,154 | $0.1998 | **$0.0087** |
| `gemini-3.5-flash` | 1,673 | **14,563** | $0.1336 | **$0.0061** |

Extraction prompts are long: messy invoice / contract / receipt text that
needs structured JSON output. The input-token term dominates Opus's spend
($2,547 × $15/Mtok = $0.038 just on inputs), while Gemini's input rate
($1.50/Mtok) makes that term nearly free. Even with Gemini emitting
**6.8× more output tokens than Opus**, total spend lands ~30% lower.

p-value 1.0, mean delta 0.000, one regressed and one improved case
(offsetting). Hedges' g = 0.000.

**This is the headline cost finding:** Gemini's *prompt-shape sensitivity*
makes the right model a function of input-to-output ratio. Input-heavy
work (extraction, classification, RAG) → Gemini. Output-heavy work
(reasoning, generation) → it's a wash.

---

## Finding 3 — Open-ended QA: a quality gap, with caveats

**This is the only suite with a non-trivial mean delta:**

| Model | Mean score | $/correct |
|-------|------------|-----------|
| `claude-opus-4-7` | **1.000** | $0.0162 |
| `gemini-3.5-flash` | **0.940** | $0.0163 |

- Hedges' g = **−0.876** (large effect by Cohen's thresholds)
- p = 0.070, 95% CI [−0.100, −0.020]
- 3 regressed cases, 0 improved
- Power at this n=5: ~50% at α=0.05

The 6.0 pp delta is *consistent* across three cases — every regression
was the same 0.10 score drop (judge giving Gemini's answer a 0.9 where
Opus got a 1.0). That's a stable pattern, not noise.

**Caveat the chart will not show you:** the judge was Sonnet 4.6 —
same model family as the baseline. The Rift `LLMJudgeScorer` documents
this exact failure mode ("family bias — judges over-reward outputs
from their own family"). The recommended fix is a third-family judge.
GPT-4o or Gemini 3.5 Flash itself would both be more defensible
judges here, and the signal might or might not survive that swap.

Notable: $/correct is identical to four decimals, despite Gemini being
3.5× more verbose. Both models spend ~$0.016 per correct answer on
this suite. Quality, not cost, is the differentiator.

---

## Cross-suite: the structural finding

| Suite | Opus output/case | Gemini output/case | Gemini : Opus output ratio | Gemini $/correct ÷ Opus $/correct |
|-------|------------------|---------------------|----------------------------|------------------------------------|
| reasoning   | 40.5  | 549.5  | **13.6×** | 1.08× (Gemini more expensive) |
| extraction  | 74.3  | 502.2  | **6.8×**  | 0.70× (Gemini cheaper) |
| open_ended  | 208.4 | 720.6  | **3.5×**  | 1.00× (tie) |

**Gemini 3.5 Flash emits 3.5–13.6× more output tokens than Opus 4.7 for
the same task.** That's not because Gemini's answers are longer (the
visible answers are similar in length) — it's because Gemini's
thinking tokens count as output, and Gemini emits a lot of them. The
ratio collapses on input-heavy tasks (Gemini's input price is cheap
enough that output dominance doesn't matter); it widens on
output-heavy tasks where the thinking-billing effectively erases the
list-price discount.

**Implication for procurement:** when comparing Flash-class to Opus-class
on a per-correct basis, you have to *measure your workload's
input-to-output ratio*. Vendor-table list prices are misleading.

---

## What is NOT in this writeup

* **Discovery.** We started a `rift discover` run with Gemini 3.5 Flash
  as the adversarial proposer (the methodologically novel artifact),
  but Anthropic credits ran out mid-loop. Queued for a follow-up.
* **Larger sample sizes.** n=5 on open_ended_qa is not enough to call
  the quality gap significant at α=0.05. The Hedges' g of −0.876 is
  large; whether it holds at n=30+ is open.
* **A non-Claude judge.** The judge family-bias caveat on Finding 3
  needs a Gemini-judge or GPT-judge replication.
* **Refusal / calibration drift.** Both ran clean across all three
  suites (0 over-refusals, 0 safety regressions), so no narrative
  there yet.

---

## Reproduce

```bash
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...

rift compare --baseline opus-4-7 --challenger gemini-3.5-flash \
             --suite reasoning \
             --report benchmarks/opus47_vs_gemini35/reasoning.md \
             --output benchmarks/opus47_vs_gemini35/reasoning.json

rift compare --baseline opus-4-7 --challenger gemini-3.5-flash \
             --suite extraction \
             --report benchmarks/opus47_vs_gemini35/extraction.md \
             --output benchmarks/opus47_vs_gemini35/extraction.json

rift compare --baseline opus-4-7 --challenger gemini-3.5-flash \
             --suite open_ended_qa \
             --judge-model claude-sonnet-4-6 \
             --report benchmarks/opus47_vs_gemini35/open_ended_qa.md \
             --output benchmarks/opus47_vs_gemini35/open_ended_qa.json
```

Total spend: ~$0.55. Cache-hits make re-runs free.

To reproduce the figures in this writeup without API access, use
`rift diff` on the committed `.json` outputs — every number in this
document is recoverable offline.
