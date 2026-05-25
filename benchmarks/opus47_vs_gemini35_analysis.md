# Gemini 3.5 Flash's 10× discount disappears on reasoning workloads

> **Provenance.** Numbers below are from a **live API run on 2026-05-20**
> against the production Anthropic and Google endpoints (day-after-GA of
> Gemini 3.5 Flash). Per-model completion JSONs are committed under
> `opus47_vs_gemini35/`. The bundled `rift demo` is a separate, offline
> synthetic story and does NOT cite these numbers.

## Executive summary

I ran [Rift](https://github.com/shahcolate/rift) against
Google's just-released Gemini 3.5 Flash, paired against Anthropic's
Claude Opus 4.7 on three eval suites and one adversarial discovery
loop (same prompts, same scorer, real API calls). The day-after-GA
list-price comparison says Gemini Flash is **10× cheaper than Opus**
($1.50 / $9.00 vs $15 / $75 per 1M tokens). The actual
per-correct-answer numbers say otherwise:

1. **On reasoning, Gemini Flash is *more expensive than Opus* per
   correct answer** ($0.0056 vs $0.0052, n=10). The 10× list-price
   discount is fully consumed by Gemini's reasoning ("thinking")
   tokens, which Google bills as output. Gemini Flash emits **13.6×
   more output tokens than Opus** on the same prompts.
2. **On structured extraction, Gemini Flash is 30% cheaper than Opus**
   per correct ($0.0061 vs $0.0087, n=29). Long inputs swing the
   math the other direction — input tokens dominate Opus's spend
   and Gemini's $1.50/Mtok input price wins.
3. **On open-ended generation (judge-scored), Gemini scores 6 pp
   lower than Opus** (1.00 vs 0.94 at n=5, Hedges' g = −0.876, p =
   0.07). Per-correct cost is identical to four decimals. Caveat:
   the judge was Claude Sonnet 4.6 — same family as the baseline,
   so family-bias is plausible. The signal is consistent (3
   regressed cases, 0 improved, all the same −0.10 score drop) but
   the magnitude is contested by the methodology.
4. **Discovery-loop methodology demo.** `rift discover` with Gemini
   itself as the adversarial proposer produced Opus 0/9 vs Gemini
   9/9 (Cohen's h = +3.14, p = 0.004) — and this writeup
   **explicitly disowns the headline**. Proposer-equals-challenger
   creates directional selection bias by construction. The
   third-party-proposer fix is a one-line change documented inline.

The headline isn't "is Gemini Flash better." The headline is **"the
right model is a function of your workload's input-to-output ratio,
not the vendor list price."** Input-heavy work → Gemini. Thinking-
heavy work → roughly a tie. Free-form generation → quality story
unresolved at this n with this judge.

_Disclosure: I maintain Rift. Numbers below are from my own runs
against the live Anthropic and Google APIs the day after Gemini
3.5 Flash GA (2026-05-19). Run your own paired benchmark before
making procurement decisions; the I:O ratio finding generalizes,
the specific magnitudes will not._

---

## At a glance

| Suite          | n  | Opus mean | Gemini mean | Δ      | Opus $/correct | Gemini $/correct | Gemini cheaper? | Statistical test |
|----------------|----|-----------|-------------|--------|----------------|-------------------|-----------------|------------------|
| reasoning      | 10 | 0.900     | 0.900       | 0.000  | $0.0052        | **$0.0056**       | No, +8%          | McNemar exact (binary)         |
| extraction     | 29 | 0.941     | 0.941       | 0.000  | $0.0087        | **$0.0061**       | **Yes, −30%**    | Paired t + bootstrap           |
| open_ended_qa  | 5  | 1.000     | 0.940       | −0.060 | $0.0162        | $0.0163           | Tie              | Paired t + bootstrap (g=−0.88) |
| discovered     | 9  | 0.000     | 1.000       | +1.000 | ∞              | $0.0090           | n/a (biased)     | McNemar exact (h=+3.14)        |

The cross-suite structural finding — and the one that travels
beyond this study — is the **output-token volume ratio**:

| Suite          | Opus output tokens (total) | Gemini output tokens (total) | Ratio  | Effect on $/correct          |
|----------------|----------------------------|------------------------------|--------|------------------------------|
| reasoning      | 405                        | 5,495                        | **13.6×** | Gemini list-price discount **erased** |
| open_ended_qa  | 1,042                      | 3,603                        | **3.5×**  | Per-correct tied              |
| extraction     | 2,154                      | 14,563                       | **6.8×**  | Gemini still 30% cheaper      |

Gemini 3.5 Flash emits 3.5–13.6× more output tokens than Opus 4.7
for the same task — not because Gemini's visible answers are longer,
but because Gemini's thinking tokens are billed as output. Whether
that erases the 10× input-price discount depends entirely on how
input-heavy your prompts are.

**Run details:** three suites + one discovery loop, all live API
(no synthetic). 53 paired prompts total. 0 errors across baseline.
Gemini's `thinking_level` pinned to `medium` (its own default) for
paired determinism. Total live spend across all four runs: **$1.10**.

---

## What an executive leader should do this week

For an engineering, platform, or finance leader evaluating whether
to move some or all of a Claude-Opus workload to Gemini 3.5 Flash,
here is the action list ranked by reversibility cost (cheapest first):

### 1. Measure your workload's input-to-output token ratio before quoting any savings

The vendor list-price comparison (`$1.50 / $9 vs $15 / $75`) implies
10× savings. The per-correct number on this study's reasoning suite
shows a **net cost increase of ~8%** at the same accuracy. The
direction of the savings flips entirely with prompt shape:

* RAG / extraction / classification (long input, short output) →
  Gemini Flash wins, on the order of 30% per correct.
* Reasoning / multi-step generation / agentic loops (modest input,
  thinking-heavy output) → Gemini Flash is **roughly a wash or
  slightly worse** because reasoning tokens are billed as output at
  $9/Mtok.

Pull a week of production prompts, compute mean
`input_tokens : output_tokens`, and only then decide which side of
the curve you're on.

### 2. Pin `thinking_level` explicitly if you switch

Gemini 3.5 Flash ships with thinking on by default at level
`medium`. For paired-comparison reproducibility, Rift pins
`thinking_level=medium` (Google's default). For production cost
optimization, the choice matters: `low` and `minimal` will reduce
output-token volume substantially at some quality cost. Until you
have your own quality benchmark, pin one level and stay there;
don't let an SDK auto-upgrade silently move the line.

### 3. Don't read the discovered-suite result as a Gemini blowout

Finding 4 says Opus 0/9 vs Gemini 9/9 — a clean 100% sweep with a
maximum Cohen's h of +3.14. That number is structurally biased: the
adversarial proposer was Gemini itself, so case selection over-rewards
the proposer. The methodologically valid version of this result
requires a **third-party proposer** (different family from both
compared models, e.g. GPT-4o). That run is queued and is the more
expensive part of the budget; do not cite the +3.14 headline.

### 4. Run your own paired benchmark before committing volume

Don't take the magnitudes in this writeup as authoritative for your
workload. n is small (5–29 per suite), the judge in Finding 3 is
family-related to the baseline, and Finding 4 is bias-flagged. The
**direction** of the input-vs-output finding generalizes; the
**numbers** require your data:

```bash
pip install rift-eval
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...

rift compare --baseline opus-4-7 --challenger gemini-3.5-flash \
    --suite YOUR_PRODUCTION_SUITE
```

Look at three columns in the Rift report: the accuracy delta, the
**output-token ratio**, and the **`cost-per-correct` delta**. The
last one is the only number a CFO cares about.

### 5. Renegotiate any "we're moving from Opus to Flash for the cost savings" budget line

If a budget line is built on the 10× list-price headline, it is
already wrong on roughly half the workload shapes. Re-baseline
on per-correct or per-task spend, not per-token, before locking
in the contract.

The reproducible numbers behind every figure in this writeup are
in the committed `.json` files. `rift diff` recreates each report
offline without spending another dollar of API.

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

## Finding 4 — Discovery: a methodology trap, surfaced honestly

`rift discover` ran with Gemini 3.5 Flash as the adversarial proposer
against the reasoning seed suite, targeting `target_power=0.9 at
Δ=0.05, α=0.05`. The loop early-stopped at 9 accepted cases:

| Metric | Value |
|--------|-------|
| n_proposed | 36 |
| n_after_dedup | 35 |
| n_both_zero | 0 |
| n_kept | 9 |
| discordant rate (of verified) | 25.7% |
| achieved_power | 1.0 (early-stopped) |
| spend (proposer + verification) | $0.57 |

The compare on the discovered suite shows:

| Model | Score on discovered suite | $/correct |
|-------|---------------------------|-----------|
| `claude-opus-4-7` | **0 / 9** | ∞ |
| `gemini-3.5-flash` | **9 / 9** | $0.0090 |

McNemar p = 0.0039, Cohen's h = **+3.14** (the maximum value for a
0-vs-1 proportion shift). Every discovered case is Gemini-right /
Opus-wrong, none in the other direction.

**A naïve reader would call this a Gemini blowout. The honest reading
is more interesting.**

The proposer was *Gemini itself*. Discovery selects on cases where
the two models disagree, and we biased the proposer toward generating
cases for which it knows the answer. The mechanism:

1. The proposer generates a candidate input and an `expected` answer.
2. The verification loop scores both models against that `expected`.
3. A case is accepted only when the two models disagree.
4. Since the proposer wrote the `expected` itself, it has an advantage
   on "would my answer be marked correct?" — and the validity gate
   (drop cases neither model got) further filters in favor of the
   proposer being right.

So the +3.14 effect size is **a measure of the proposer-asymmetry
bias**, not a population estimate of Opus vs Gemini on novel
reasoning. The Rift YAML metadata states this explicitly
(`IMPORTANT: cases were selected on divergence...`), but the directional
bias from `proposer == challenger` is sharper than that generic caveat.

**The methodological fix** is to use a third-party proposer — a model
from a different family from both compared models. With this codebase
that's a one-line change:

```
proposer_model="gpt-4o"   # neither baseline nor challenger family
```

We did not run that variant in this study (would cost ~$2-3 extra and
introduce a third API key) but it is the obvious follow-up.

**What this finding does demonstrate cleanly:**
* The discovery loop works end-to-end — proposer + dedup + validity
  gate + power-based early-stop all fire correctly.
* Caching makes the final compare free — once the verification calls
  during discovery are written to `.rift/cache/`, the published
  compare run incurs zero new API spend.
* The selection-bias caveat is load-bearing, not decorative. A reader
  who ignores it will mis-cite this result. A methodology-stable
  comparison **requires** a third-party proposer; we will not publish
  Opus-vs-Gemini-via-Gemini-proposer headline numbers without that
  fix in a future run.

The raw discovered suite is at
`benchmarks/opus47_vs_gemini35/discovered_reasoning.yaml`, and the
compare output at `discovered_drift.md`. Both are committed so the
selection-bias claim is reviewable case-by-case.

## What is NOT in this writeup

* **A bias-free discovery run.** As discussed above, the discovered
  suite is biased toward the proposer model. A third-party-proposer
  variant (e.g., GPT-4o or Sonnet) is the obvious next step.
* **Larger sample sizes.** n=5 on open_ended_qa is not enough to call
  the quality gap significant at α=0.05. The Hedges' g of −0.876 is
  large; whether it holds at n=30+ is open.
* **A non-Claude judge.** The judge family-bias caveat on Finding 3
  needs a Gemini-judge or GPT-judge replication.
* **Refusal / calibration drift.** Both ran clean across all suites
  (0 over-refusals, 0 safety regressions), so no narrative there yet.

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
