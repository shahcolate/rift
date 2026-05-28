# Gemini 3.5 Flash's list-price discount vanishes per correct answer — even on input-heavy extraction

> **Provenance.** Token counts, scores, and accuracy below are from a
> **live API run on 2026-05-20** against the production Anthropic and
> Google endpoints (day-after-GA of Gemini 3.5 Flash). Per-model
> completion JSONs are committed under `opus47_vs_gemini35/`.
>
> **The Opus dollar figures have been recomputed at the current Opus
> 4.5-generation list price** ($5/Mtok input, $25/Mtok output; see
> `src/rift/pricing.py`). The original 2026-05-20 capture priced Opus at
> $15/$75, so its Opus dollar figures were 3× higher; Gemini prices did
> not change. This re-pricing **reverses the headline cost ranking** —
> see the executive summary. The bundled `rift demo` is a separate,
> offline synthetic story and does NOT cite these numbers.

## Executive summary

I ran [Rift](https://github.com/shahcolate/rift) against
Google's just-released Gemini 3.5 Flash, paired against Anthropic's
Claude Opus 4.7 on three eval suites and one adversarial discovery
loop (same prompts, same scorer, real API calls). The day-after-GA
list-price comparison said Gemini Flash was **10× cheaper than Opus**
($1.50 / $9.00 vs $15 / $75 per 1M tokens). Two things have since
collapsed that headline: the per-correct-answer arithmetic, and the
Opus 4.5-generation price cut to **$5 / $25**, which shrinks the
list-price gap to roughly **3×** and tips every per-correct comparison
to Opus:

1. **On reasoning, Opus is now ~3.3× cheaper per correct answer than
   Gemini Flash** ($0.0017 vs $0.0056, n=10). Gemini's 10×-then-3×
   list-price discount is more than consumed by its reasoning
   ("thinking") tokens, which Google bills as output. Gemini Flash
   emits **13.6× more output tokens than Opus** on the same prompts.
2. **On structured extraction — the one suite where Gemini used to win
   — Opus is now ~2.1× cheaper per correct** ($0.0029 vs $0.0061,
   n=29). At the old $15/$75 Opus price this suite went the other way
   (Gemini 30% cheaper); the price cut reverses it, because Gemini's
   6.8× output-token volume now outweighs its shrunken input discount.
3. **On open-ended generation (judge-scored), Opus is both higher
   quality *and* ~3× cheaper** per correct ($0.0054 vs $0.0163; mean
   1.00 vs 0.94 at n=5, Hedges' g = −0.876, p = 0.07). Caveat: the
   judge was Claude Sonnet 4.6 — same family as the baseline — so
   family-bias on the quality gap is plausible. The signal is
   consistent (3 regressed cases, 0 improved, all the same −0.10 score
   drop) but the magnitude is contested by the methodology.
4. **Discovery-loop methodology demo.** `rift discover` with Gemini
   itself as the adversarial proposer produced Opus 0/9 vs Gemini
   9/9 (Cohen's h = +3.14, p = 0.004) — and this writeup
   **explicitly disowns the headline**. Proposer-equals-challenger
   creates directional selection bias by construction. The
   third-party-proposer fix is a one-line change documented inline.

The headline isn't "is Gemini Flash better." The headline is **"the
right model is a function of your workload's input-to-output ratio and
the current list prices — and after the Opus 4.5-gen price cut, the
per-correct math favors Opus on every suite it can answer, because
Gemini's thinking-token output volume costs more than its remaining
list-price discount saves."** The I:O-ratio mechanism is the same one
the original capture identified; what moved is the price, and it moved
far enough to flip the conclusion.

_Disclosure: I maintain Rift. Token counts and accuracy below are from
my own runs against the live Anthropic and Google APIs the day after
Gemini 3.5 Flash GA (2026-05-19); Opus dollar figures are recomputed at
the current $5/$25 list price. Run your own paired benchmark before
making procurement decisions; the I:O ratio mechanism generalizes, the
specific magnitudes will not._

---

## At a glance

| Suite          | n  | Opus mean | Gemini mean | Δ      | Opus $/correct | Gemini $/correct | Cheaper per correct | Statistical test |
|----------------|----|-----------|-------------|--------|----------------|-------------------|---------------------|------------------|
| reasoning      | 10 | 0.900     | 0.900       | 0.000  | **$0.0017**    | $0.0056           | Opus, 3.3×          | McNemar exact (binary)         |
| extraction     | 29 | 0.941     | 0.941       | 0.000  | **$0.0029**    | $0.0061           | Opus, 2.1×          | Paired t + bootstrap           |
| open_ended_qa  | 5  | 1.000     | 0.940       | −0.060 | **$0.0054**    | $0.0163           | Opus, 3.0×          | Paired t + bootstrap (g=−0.88) |
| discovered     | 9  | 0.000     | 1.000       | +1.000 | ∞              | $0.0090           | n/a (biased)        | McNemar exact (h=+3.14)        |

The cross-suite structural finding — and the one that travels
beyond this study — is the **output-token volume ratio**:

| Suite          | Opus output tokens (total) | Gemini output tokens (total) | Ratio  | Effect on $/correct          |
|----------------|----------------------------|------------------------------|--------|------------------------------|
| reasoning      | 405                        | 5,495                        | **13.6×** | Gemini discount erased, then some |
| open_ended_qa  | 1,042                      | 3,603                        | **3.5×**  | Opus 3× cheaper               |
| extraction     | 2,154                      | 14,563                       | **6.8×**  | Opus 2× cheaper (was Gemini's win) |

Gemini 3.5 Flash emits 3.5–13.6× more output tokens than Opus 4.7
for the same task — not because Gemini's visible answers are longer,
but because Gemini's thinking tokens are billed as output. At the old
$15/$75 Opus price that volume tax only erased Gemini's discount on
output-heavy suites; at the new $5/$25 price the remaining ~3×
list-price gap is small enough that the output tax wins **everywhere**,
including input-heavy extraction.

**Run details:** three suites + one discovery loop, all live API
(no synthetic). 53 paired prompts total. 0 errors across baseline.
Gemini's `thinking_level` pinned to `medium` (its own default) for
paired determinism. Total live spend across the three comparison runs,
recomputed at the current Opus price: **~$0.33** (it was ~$0.55 at the
old $15/$75 rate). The discovery loop in Finding 4 adds further
proposer + verification spend on top.

---

## What an executive leader should do this week

For an engineering, platform, or finance leader evaluating whether
to move some or all of a Claude-Opus workload to Gemini 3.5 Flash,
here is the action list ranked by reversibility cost (cheapest first):

### 1. Re-baseline against the current Opus price before quoting any Flash savings

The vendor list-price comparison everyone remembers (`$1.50 / $9 vs
$15 / $75`) implied 10× savings. That comparison is doubly stale: Opus
now lists at **$5 / $25** (a ~3× gap, not 10×), and on a per-correct
basis Opus is the cheaper model on all three standard suites here. If a
plan to move workload to Flash was built on the 10× headline, it is now
built on a number that no longer exists.

Prompt shape still decides the *magnitude*, just not the *direction*:

* RAG / extraction / classification (long input, short output) →
  Opus, ~2× cheaper per correct (this used to be Gemini's win).
* Reasoning / multi-step generation / agentic loops (modest input,
  thinking-heavy output) → Opus, ~3× cheaper, because Gemini's
  thinking tokens are billed as output at $9/Mtok against Opus's now
  far more competitive $25/Mtok.

Pull a week of production prompts, compute mean
`input_tokens : output_tokens`, and re-run the comparison at current
prices before quoting anything.

### 2. Pin `thinking_level` explicitly if you switch

Gemini 3.5 Flash ships with thinking on by default at level
`medium`. For paired-comparison reproducibility, Rift pins
`thinking_level=medium` (Google's default). For production cost
optimization, the choice matters: `low` and `minimal` will reduce
output-token volume substantially at some quality cost — and given
that output volume is exactly what loses Gemini the per-correct race
here, this is the single biggest lever on a Flash deployment's bill.
Until you have your own quality benchmark, pin one level and stay
there; don't let an SDK auto-upgrade silently move the line.

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
**numbers** require your data and your current contract prices:

```bash
pip install rift-eval
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...

rift compare --baseline opus-4-8 --challenger gemini-3.5-flash \
    --suite YOUR_PRODUCTION_SUITE
```

Look at three columns in the Rift report: the accuracy delta, the
**output-token ratio**, and the **`cost-per-correct` delta**. The
last one is the only number a CFO cares about.

### 5. Renegotiate any "we're moving from Opus to Flash for the cost savings" budget line

If a budget line is built on the 10× list-price headline, it is now
wrong twice over: the gap is ~3×, and on a per-correct basis Opus is
the cheaper model on every suite here. Re-baseline on per-correct or
per-task spend, not per-token, before locking in the contract.

The reproducible numbers behind every figure in this writeup are
in the committed `.json` files (Opus `cost_usd` fields recomputed at
the current $5/$25 price). `rift diff` recreates each report offline
without spending another dollar of API.

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

Total live-API spend for the three comparison runs, recomputed at the
current $5/$25 Opus price: **~$0.33** (was ~$0.55 at $15/$75).

---

## Finding 1 — Reasoning: thinking tokens eat what's left of the Flash discount

**Both models scored 9/10 on the reasoning suite. The cost story is
the headline:**

| Model | Total input tokens | Total output tokens | Total spend | $/correct |
|-------|--------------------|---------------------|-------------|-----------|
| `claude-opus-4-7` | 1,077 | 405 | $0.0155 | **$0.0017** |
| `gemini-3.5-flash` | 787 | **5,495** | $0.0506 | **$0.0056** |

Gemini's output volume on a 10-case binary reasoning suite was **13.6×
greater than Opus's** — and at the new prices Gemini's list-price output
rate ($9/Mtok) is only 2.8× cheaper than Opus's ($25/Mtok), down from
8.3× at the old $75 rate. The arithmetic is now lopsided: Flash issues
enough reasoning tokens that its per-token discount doesn't just
collapse, it goes negative — Opus is 3.3× cheaper per correct.

This is the kind of finding a list-price-only spreadsheet misses
entirely. Both Anthropic Opus 4.7 and Gemini 3.5 Flash run extended
thinking by default; the difference is *how many tokens of it land in
the output column on your invoice*. Per-correct-answer cost — not
per-token — is the only sane metric for budget defense.

p-value 1.0, effect size 0.000 (Cohen's h), no significant drift in
either direction. The reasoning suite is too small (n=10, power ~5%
at α=0.05) to detect anything subtle.

---

## Finding 2 — Extraction: the price cut flips the one suite Gemini used to win

**Both models scored 0.941 on the extraction suite. At the old Opus
price Gemini was 30% cheaper here; the $5/$25 price cut reverses it:**

| Model | Total input tokens | Total output tokens | Total spend | $/correct |
|-------|--------------------|---------------------|-------------|-----------|
| `claude-opus-4-7` | 2,547 | 2,154 | $0.0666 | **$0.0029** |
| `gemini-3.5-flash` | 1,673 | **14,563** | $0.1336 | **$0.0061** |

Extraction prompts are long: messy invoice / contract / receipt text that
needs structured JSON output. At the old $15/Mtok input rate the input
term was a meaningful chunk of Opus's bill and Gemini's $1.50/Mtok
input price won the suite. At $5/Mtok input, Opus's input cost drops to
~$0.013 and its output cost (2,154 × $25/Mtok ≈ $0.054) dominates its
own bill — but Gemini, emitting **6.8× more output tokens**, pays
~$0.131 on output even at $9/Mtok. The output-volume tax now beats the
input discount, and total spend lands ~2× *lower* for Opus.

p-value 1.0, mean delta 0.000, one regressed and one improved case
(offsetting). Hedges' g = 0.000.

**This is the reversal that matters for procurement:** the suite that
made the "move extraction to Flash for cost" case at the old price now
makes the opposite case. The deciding variable was never just the
input-to-output ratio — it was the ratio *times the current per-token
prices*, and one of those prices just fell 3×.

---

## Finding 3 — Open-ended QA: Opus wins on both quality and cost

**This is the only suite with a non-trivial mean delta — and now Opus
is also the cheaper model:**

| Model | Mean score | $/correct |
|-------|------------|-----------|
| `claude-opus-4-7` | **1.000** | **$0.0054** |
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

Notable: at the old Opus price the two models tied on $/correct here
(~$0.016 each) and the suite was a "quality, not cost" story. At the
new price Opus is **3× cheaper *and* higher quality**, so the only
thing keeping this from being a clean Opus win is the unresolved
judge-family-bias question above.

---

## Cross-suite: the structural finding

| Suite | Opus output/case | Gemini output/case | Gemini : Opus output ratio | Gemini $/correct ÷ Opus $/correct |
|-------|------------------|---------------------|----------------------------|------------------------------------|
| reasoning   | 40.5  | 549.5  | **13.6×** | 3.3× (Gemini more expensive) |
| extraction  | 74.3  | 502.2  | **6.8×**  | 2.1× (Gemini more expensive) |
| open_ended  | 208.4 | 720.6  | **3.5×**  | 3.0× (Gemini more expensive) |

**Gemini 3.5 Flash emits 3.5–13.6× more output tokens than Opus 4.7 for
the same task.** That's not because Gemini's answers are longer (the
visible answers are similar in length) — it's because Gemini's
thinking tokens count as output, and Gemini emits a lot of them. At the
old $15/$75 Opus price this only made Gemini more expensive on the
output-heaviest suite (reasoning) and left extraction to Gemini. At the
new $5/$25 price the list-price gap is small enough that the
output-volume tax makes Gemini more expensive **on every suite**.

**Implication for procurement:** when comparing Flash-class to
Opus-class on a per-correct basis, you have to *measure your workload's
input-to-output ratio and plug in current prices*. Vendor-table list
prices — and last quarter's list prices — are both misleading.

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

The compare on the discovered suite shows:

| Model | Score on discovered suite | $/correct |
|-------|---------------------------|-----------|
| `claude-opus-4-7` | **0 / 9** | ∞ |
| `gemini-3.5-flash` | **9 / 9** | $0.0090 |

McNemar p = 0.0039, Cohen's h = **+3.14** (the maximum value for a
0-vs-1 proportion shift). Every discovered case is Gemini-right /
Opus-wrong, none in the other direction. (Opus's $/correct is ∞ here
because it got zero correct — unchanged by any pricing; Gemini's
$0.0090 is its own unchanged price.)

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

Total spend at current prices: ~$0.33. Cache-hits make re-runs free.

To reproduce the figures in this writeup without API access, use
`rift diff` on the committed `.json` outputs — every number in this
document is recoverable offline. (The committed Opus `cost_usd` values
reflect the current $5/$25 list price; token counts are unchanged from
the 2026-05-20 capture.)
