# Opus 4.7 costs 45% more than 4.6 for the same prompt

## Executive summary

I ran [Rift](https://github.com/shahcolate/rift) against
Anthropic's new Opus 4.7 on a context-rot reasoning suite,
paired against Opus 4.6 (same prompts, same scorer, real API
calls). Two findings:

1. **Short workloads are a wash. The lift, if any, is at the
   long end.** On 0k and 2k distractor tokens, both models
   complete **87.5% (7/8)** of cases — indistinguishable. At
   8k and 32k distractors, success rate moves from **75% → 87.5%
   (+12.5pp)** on each bucket. Directional but not significant at
   n=8 per bucket; +6.25pp overall (26/32 → 28/32, p=0.69).
2. **The tokenizer changed.** For byte-identical prompts,
   Opus 4.7 emits **1.43× more input tokens** than 4.6 (range
   1.21×–1.62× across 32 paired prompts, uniform across all
   distractor levels). At list-price parity, this is a silent
   **~45% per-prompt cost increase** on migration.

The combined effect on `cost-per-correct-answer`: **+35% on
Opus 4.7** (`$0.244` vs `$0.181`). *If* the 12.5pp long-end
success-rate lift holds at larger n, a pure long-workload (all
8k/32k) would see that cost-per-correct delta shrink from +35%
to about **+24%** — still a regression on unit economics, but a
meaningfully smaller one. An 80% long / 20% short mix lands
around **+28%**. On a workload that is mostly 0k/2k, the lift
buys nothing and the full +35% lands.

The headline isn't "is Opus 4.7 better." The headline is "your
token bill goes up 30–45% even if you change nothing else, and
whether the long-context success-rate lift pays any of it back
depends on what fraction of your prompts are actually long."

**Why this benchmark alone can't settle the long-workload
question:** exact-match scoring collapses success rate to mean
score, and n=8 per distractor bucket is underpowered to prove the
12.5pp long-end lift is real. A graded scorer (token-F1 or
embedding similarity) on a larger suite would let success rate
diverge from mean and bring the "does 4.7 actually carry long
workloads better" question into a range where it can be answered,
not just hinted at.

_Disclosure: I maintain Rift. Numbers below are from my own runs
against the Anthropic API; run your own paired benchmark before
making procurement decisions._

---

## At a glance

| Model      | Mean       | Correct    | Errors | Spend     | Cost/correct      |
|------------|------------|------------|--------|-----------|-------------------|
| opus-4-6   | 0.812      | 26/32      | 0      | $4.72     | `$0.1815`         |
| **opus-4-7** | 0.875 (ns) | 28/32 (ns) | **0**  | **$6.84** | **`$0.2444`**     |

- Drift on accuracy: +6.25pp, p=0.69 (McNemar's exact test on 6
  discordant pairs: 4 improved, 2 regressed). Not significant.
- Drift on tokens: **+44.7% input tokens** for byte-identical
  prompts. Highly significant — every paired prompt is at least
  +21% on Opus 4.7.
- Drift on `cost-per-correct`: **+35%** at list pricing.

**Run details:** 32 cases (8 base reasoning prompts × 4 distractor
regimes: 0k, 2k, 8k, 32k tokens). Real Anthropic API, 0 errors.
Total spend: $11.56.

---

## What an executive leader should do this week

For an engineering, platform, or finance leader whose team is
running on Anthropic's Claude Opus today, here is the action
list ranked by reversibility cost (cheapest first):

### 1. Do not enable a default-upgrade flag in your routing layer

If your gateway routes "opus" to "the latest opus," **pin to
`claude-opus-4-6` explicitly** until you have your own paired
benchmark. A silent +30–45% input-token bill on byte-identical
prompts is the kind of finding that lands in a CFO email three
weeks late.

### 2. Re-baseline your token budgets before you migrate

Take your last 7 days of production prompts and re-tokenize a
sample of them through Opus 4.7's API (one call per sample is
enough — `usage.input_tokens` in the response is the source of
truth). Multiply your committed annual token spend by the
observed ratio. Renegotiate your Enterprise contract on
**`tokens/prompt × prompts/day`**, not `$/Mtok` alone.

### 3. Re-test prompt caching assumptions

Anthropic's prompt-caching tier (5-minute and 1-hour ephemeral
windows) prices cached input tokens at a discount. A tokenizer
change invalidates cache entries — your first week post-upgrade
will see lower cache hit rates by construction. Budget for a
2–4 week re-warm period and confirm the cache line items in
your contract apply to the new tokenizer.

### 4. Measure what fraction of your prompts are actually long

The +35% blended cost-per-correct delta assumes your workload
looks like this suite — a uniform mix across distractor regimes.
Yours probably doesn't. Pull a week of production prompts,
bucket by input-token length, and weight the cost-per-correct
deltas by that distribution rather than the uniform one. If
you're mostly short-prompt traffic (chat, classification, small
extractions) the quality lift buys nothing and the full
tokenizer tax applies. If you're mostly long-context (RAG over
large corpora, long-document reasoning) you're in the regime
where the lift might offset some of the cost, and a
bigger-sample follow-up is worth running before ruling 4.7 out.

### 5. Run your own paired drift report before signing off

Don't take this writeup's numbers as authoritative for your
workload. The 1.43× inflation here is on English corporate
filler. Code, JSON, multilingual, or domain-specific text
tokenizes differently and may show 1.1× or 1.7× — measure it.

```bash
pip install rift-eval
rift compare --baseline opus-4-6 --challenger opus-4-7 \
    --suite YOUR_PRODUCTION_SUITE \
    --subgroup length: --success-threshold 0.8
```

Look at four columns in the report: the accuracy delta, the
total-spend delta, the **cost-per-correct delta**, and the
per-subgroup **success-rate delta**. The third is the only one
that integrates quality and price into a number you can defend
in a budget review; the fourth is where Opus-tier pricing earns
its keep, if it earns it at all.

### 6. If you must migrate now, negotiate

You have a real, measurable +30–45% bill increase on the same
workload. That is leverage. Either negotiate an offsetting
discount on the per-token rate, request a cap-protection clause
that locks the contract on `prompts/day` rather than
`tokens/day`, or stage the migration to a low-volume workload
first to characterize the inflation on your actual prompts.

---

## Finding 1: success rate is flat at the short end, directional at the long end

The relevant metric on a reasoning suite is **fraction of cases
the model actually gets right** — not mean score, which happens
to equal fraction-correct here only because the scorer is
binary. Per bucket:

| Distractor  | opus-4-6 success | opus-4-7 success | Δ      | 95% CI            |
|-------------|------------------|------------------|--------|-------------------|
| 0k          | 87.5% (7/8)      | 87.5% (7/8)      | +0.0pp | [-25.0, +37.5]    |
| 2k          | 87.5% (7/8)      | 87.5% (7/8)      | +0.0pp | [-25.3, +37.5]    |
| 8k          | 75.0% (6/8)      | 87.5% (7/8)      | +12.5pp | [+0.0, +37.5]    |
| **32k**     | **75.0% (6/8)**  | **87.5% (7/8)**  | **+12.5pp** | **[+0.0, +37.5]** |

At both long-context buckets the 95% CI on the success-rate
delta has its lower bound at 0 — the data is consistent with a
true +12.5pp improvement up to +37.5pp but cannot rule out zero.
At n=8 per bucket that's what "directional, not significant"
looks like on exact-match scoring; binary outcomes at small n
are brutal to get p-values from. *If* the long-end lift is real,
this is exactly the shape Opus-tier pricing is supposed to
justify: 4.7 completing cases that 4.6 drops on long inputs.
Running with n≥50 per bucket, or switching to a graded scorer so
partial-credit signal shows up, would push this into publishable
territory.

Short-context is flat — both models already at 87.5% and no
evidence either way on what a 16th case would show. Any cost
story on short-workload traffic has to stand on the tokenizer
finding alone; there is no quality-side offset.

Two regressed cases on 4.7 — both variants of the seating-puzzle
CSP with distractor context. Worth a failure-mode writeup if the
pattern repeats on a larger run.

## Finding 2: the tokenizer changed. The price didn't.

The headline cost number above — `$4.72` vs `$6.84` — is a **45%
spend increase on the challenger despite list prices being
identical per token**. From the cache:

```
opus-4-6 total input tokens:  313,717  (mean 9,804/case)
opus-4-7 total input tokens:  453,957  (mean 14,186/case)
Extra tokens on 4-7:          140,240  (+44.7%)
```

For byte-identical prompts. This is not a long-context
artifact:

| Distractor | opus-4-6 mean tokens | opus-4-7 mean tokens | Ratio |
|------------|----------------------|----------------------|-------|
| 0k         | 60                   | 81                   | 1.37× |
| 2k         | 2,018                | 2,935                | 1.45× |
| 8k         | 7,506                | 10,869               | 1.45× |
| 32k        | 29,631               | 42,860               | 1.45× |

Per-prompt ratio range: **1.21× to 1.62×, mean 1.43×**. The
inflation is approximately uniform from short prompts to 32k-token
ones, which is consistent with a tokenizer-vocabulary change
(different BPE merges) rather than a fixed system-prompt insert —
a fixed insertion would produce a much higher ratio on the
60-token prompts than the 29k-token ones, and we don't see that.
The `usage.cache_creation_input_tokens` and
`usage.cache_read_input_tokens` fields are zero on every call,
ruling out prompt caching as a confound. Anthropic has not
published a tokenizer changelog for 4.7; treat the cause as
inferred from the ratio shape, not confirmed.

**Why this matters for anyone migrating:**

- **List-price parity is a mirage.** Opus 4.7 is listed at
  $15/Mtok input — the same as Opus 4.6. At 1.43× tokenization,
  the **effective rate on real prompts is ~$21.45/Mtok input**.
- **Enterprise contracts that renegotiate price-per-token
  without retesting tokens-per-prompt will be wrong.** If your
  annual committed spend is denominated in tokens and you
  default-upgrade your model, you will hit your cap ~30%
  sooner than last year's budget suggests.
- **`Cost-per-correct` is the right denominator.** On this
  suite: `$0.1815` (4.6) → `$0.2444` (4.7), a **+35% cost per
  fully correct answer** even accounting for 4.7's
  (non-significant) quality lift. The quality lift is not
  priced in; the token inflation is.

## Scale estimate

For a production pipeline at 10M daily input tokens on Opus 4.6,
naive upgrade to 4.7 moves you to ~14.3M tokens/day for
byte-identical prompts. At list ($15/Mtok input, 4.3M extra
tokens/day): **+$64.50/day input cost (~$23.5k/year) with zero
workload change**. Larger pipelines scale linearly — a 100M
tokens/day workload is ~$235k/year on the same assumption.

Your mileage will vary — the inflation ratio depends on what is
in your prompts (the 1.45× figure here is on English
corporate-filler distractor context; code-heavy or
multilingual workloads could differ substantially). Measure it
before you migrate, not after.

## What Rift caught that a casual eval would not have

The first run of this benchmark produced **"opus-4-7: 0/32, $0,
-100% drift"** — a catastrophic-looking regression. It was a
tooling bug: opus-4-7 deprecates `temperature`, `top_p`, and
`top_k` and rejects suites that pass them with a 400. The
runner correctly classified 400s as non-retryable, but the
reporter rendered the all-errored result as real drift. Two
fixes landed as a result:

1. A per-model `DEPRECATED_PARAMS` map in the Anthropic
   provider silently strips knobs the model no longer honors,
   preserving paired determinism (the dropped params were not
   being applied by the model anyway).
2. The report now carries an `Errors` column and a
   top-of-page warning when any model had API errors. A drift
   report on an error-tainted run is biased downward by
   construction; the tooling refuses to let it look clean.

## What Enterprise token-pricing changes

Two things change on Enterprise contracts, neither of which is
addressed by list-price parity:

1. **Tokenizer changes are not a line item in the contract.**
   Your negotiated rate-per-million-tokens stays the same; your
   tokens-per-prompt silently grows ~40%.
2. **`Cost-per-correct` folds quality and price into one number
   you can defend in a budget review** (latency-per-correct and
   tail-latency matter too, and aren't captured here). Rift's
   `--enterprise-multiplier` applies your negotiated rate
   uniformly; the delta in cost-per-correct is then directly
   comparable across models. In this case:

   ```bash
   rift compare --baseline opus-4-6 --challenger opus-4-7 \
       --suite context_rot_reasoning --context-rot \
       --subgroup distractor: --enterprise-multiplier 0.65
   ```

   At a 65%-of-list contract, the 4.6→4.7 cost-per-correct delta
   shrinks proportionally but stays positive — still negative on
   unit economics for this suite.

## What would change the conclusion

1. **n=32 is too small for a significance test on the quality
   side.** If the 8k/32k directional improvement holds at n≥50,
   it becomes publishable. The suite is intentionally small so
   the reproducibility bar is low; run it against your own data.
2. **This is one suite.** Reasoning with corporate-filler
   distractors is not the only shape of enterprise workload.
   Extraction, summarization, and code-gen tokenize differently
   and may show different inflation ratios.
3. **Enterprise contracts may include caching discounts this
   suite does not exercise.** A multi-turn workload with
   significant prompt overlap could recover some of the
   tokenizer inflation through prompt caching — but the base
   rate still moves.

## Reproduce

```bash
git clone https://github.com/shahcolate/rift && cd rift
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-ant-...
python benchmarks/run_context_rot.py --mode live \
    --models opus-4-6,opus-4-7 --baseline opus-4-6 \
    --concurrency 1 --output benchmarks/opus47_live.md
```

Raw report: [`opus47_live.md`](opus47_live.md). Total wall time
~20 minutes at concurrency=1, total spend ~$12.

---

_Rift is MIT-licensed. Run it against your own data before you
decide whether to upgrade. The numbers will change when the
models do._
