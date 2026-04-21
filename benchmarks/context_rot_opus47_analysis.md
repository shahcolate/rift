# Opus 4.7 costs 45% more than 4.6 for the same prompt

## Executive summary

I ran [Rift](https://github.com/shahcolate/rift) against
Anthropic's new Opus 4.7 on a context-rot reasoning suite,
paired against Opus 4.6 (same prompts, same scorer, real API
calls). Two findings:

1. **Quality is directionally better but not statistically
   significant** at n=32: +6.25 percentage points overall
   (26/32 → 28/32 correct), with the entire lift concentrated in
   the long-context regimes (+12.5pp at both 8k and 32k
   distractor tokens).
2. **The tokenizer changed.** For byte-identical prompts,
   Opus 4.7 emits **1.43× more input tokens** than 4.6 (range
   1.21×–1.62× across 32 paired prompts, uniform across all
   distractor levels). At list-price parity, this is a silent
   **~45% per-prompt cost increase** on migration.

The combined effect on `cost-per-correct-answer`: **+35% on
Opus 4.7** (`$0.244` vs `$0.181`). The quality lift, even if real,
does not pay for the tokenizer inflation on this suite.

The headline isn't "is Opus 4.7 better." The headline is "your
token bill goes up 30–45% even if you change nothing else."

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

### 4. Run your own paired drift report before signing off

Don't take this writeup's numbers as authoritative for your
workload. The 1.43× inflation here is on English corporate
filler. Code, JSON, multilingual, or domain-specific text
tokenizes differently and may show 1.1× or 1.7× — measure it.

```bash
pip install rift-eval
rift compare --baseline opus-4-6 --challenger opus-4-7 \
    --suite YOUR_PRODUCTION_SUITE
```

Look at three columns in the report: the accuracy delta, the
total-spend delta, and the **cost-per-correct delta**. The third
is the only one that integrates quality and price into a number
you can defend in a budget review.

### 5. If you must migrate now, negotiate

You have a real, measurable +30–45% bill increase on the same
workload. That is leverage. Either negotiate an offsetting
discount on the per-token rate, request a cap-protection clause
that locks the contract on `prompts/day` rather than
`tokens/day`, or stage the migration to a low-volume workload
first to characterize the inflation on your actual prompts.

---

## Finding 1: quality — directionally better, not significant

| Subgroup    | opus-4-6 | opus-4-7 | Δ      | 95% CI            |
|-------------|----------|----------|--------|-------------------|
| 0k          | 0.875    | 0.875    | +0.000 | [-0.250, +0.375]  |
| 2k          | 0.875    | 0.875    | +0.000 | [-0.253, +0.375]  |
| 8k          | 0.750    | 0.875    | +0.125 | [+0.000, +0.375]  |
| **32k**     | **0.750** | **0.875** | **+0.125** | **[+0.000, +0.375]** |

At both 8k and 32k distractor tokens, the 95% CI on the delta
has its lower bound at +0.000 — the data is consistent with a
true improvement up to +37.5pp but cannot rule out zero. *If*
the effect is real, it is concentrated where it matters
(long-context robustness) on a suite where the challenger had
no room to improve at the short-context end (both already at
87.5%). Running with n≥50 would push the p-value into
publishable territory.

Two regressed cases — both variants of the seating-puzzle CSP
with distractor context. Worth a failure-mode writeup if the
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
  $5/Mtok input — the same as Opus 4.6. At 1.43× tokenization,
  the **effective rate on real prompts is ~$7.15/Mtok input**.
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
byte-identical prompts. At list ($5/Mtok input, 4.3M extra
tokens/day): **+$21.50/day input cost (~$7.85k/year) with zero
workload change**. Larger pipelines scale linearly — a 100M
tokens/day workload is ~$78.5k/year on the same assumption.

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
