# Opus 4.7 costs 45% more than 4.6 for the same prompt

**TL;DR.** I ran Rift against Anthropic's new Opus 4.7 on a
context-rot reasoning suite paired against Opus 4.6. Two findings:
(1) quality is **directionally better** on long-context regimes
(+12.5pp at both 8k and 32k distractor tokens) but **not
statistically significant** at n=32. (2) Opus 4.7's tokenizer
produces **1.43× more input tokens** than 4.6 for byte-identical
prompts. At list-price parity, that's a silent **~45% cost
increase per prompt** on migration. Enterprise contracts that
renegotiate $/token without renegotiating $/prompt will be
mispriced.

**Benchmark:** `context_rot_reasoning` (8 base reasoning cases × 4
distractor regimes = 32 paired cases per model)
**Models:** `claude-opus-4-6`, `claude-opus-4-7`
**Mode:** `live` (real API calls, no synthesis)
**Tooling:** [Rift](https://github.com/shahcolate/rift) v0.2,
McNemar's exact test on paired binary outcomes, paired bootstrap
95% CI on deltas.
**Reproduce:**
```bash
python benchmarks/run_context_rot.py --mode live \
    --models opus-4-6,opus-4-7 --baseline opus-4-6 \
    --concurrency 1 --output benchmarks/opus47_live.md
```
Raw report: [`opus47_live.md`](opus47_live.md).

---

## Finding 1: quality — directional, not significant

| Model      | Mean   | Correct | Errors | Spend  | $/correct |
|------------|--------|---------|--------|--------|-----------|
| opus-4-6   | 0.812  | 26/32   | 0      | $4.72  | $0.1815   |
| **opus-4-7** | **0.875** | **28/32** | **0**  | **$6.84** | **$0.2444** |

**Δ = +6.25 pp, p = 0.69, 95% CI [-6.3, +21.9]** (McNemar's exact
test on 6 discordant pairs: 4 improved, 2 regressed). At n=32 this
is nowhere near statistically significant. It is, however,
directionally consistent in the one place that matters:

| Subgroup    | opus-4-6 | opus-4-7 | Δ      | 95% CI        |
|-------------|----------|----------|--------|---------------|
| 0k          | 0.875    | 0.875    | +0.000 | [-0.250, +0.375] |
| 2k          | 0.875    | 0.875    | +0.000 | [-0.253, +0.375] |
| 8k          | 0.750    | 0.875    | +0.125 | [+0.000, +0.375] |
| **32k**     | **0.750** | **0.875** | **+0.125** | **[+0.000, +0.375]** |

At both 8k and 32k distractor tokens, the 95% CI on the delta has
its lower bound at +0.000 — the data is consistent with a true
improvement up to +37.5pp but cannot rule out zero. Reading the
tea leaves: *if* the effect is real, it's concentrated where it
matters (long-context robustness) and shows up on a suite where the
challenger had no room to improve at the short-context end (both
already at 87.5%). Running with n≥50 would push the p-value into
publishable territory.

Two regressed cases — both variants of the seating-puzzle CSP with
distractor context. Worth a failure-mode writeup if the pattern
repeats on a larger run.

## Finding 2: the tokenizer changed. The price didn't.

The headline cost number above — $4.72 vs $6.84 — is a **45%
spend increase on challenger despite list prices being identical
per token**. Digging into the cache:

```
opus-4-6 total input tokens:  313,717  (mean 9,804/case)
opus-4-7 total input tokens:  453,957  (mean 14,186/case)
Extra tokens on 4-7:          140,240  (+44.7%)
```

For byte-identical prompts. This is not a long-context artifact:

| Distractor | opus-4-6 mean tokens | opus-4-7 mean tokens | Ratio |
|------------|----------------------|----------------------|-------|
| 0k         | 60                   | 81                   | 1.37× |
| 2k         | 2,018                | 2,935                | 1.45× |
| 8k         | 7,506                | 10,869               | 1.45× |
| 32k        | 29,631               | 42,860               | 1.45× |

Per-prompt ratio range: **1.21× to 1.62×, mean 1.43×**. The
inflation is uniform from short prompts to 32k-token ones, which
means it's a tokenizer-vocabulary change (different BPE merges),
not a system-prompt insert or a thinking-token leak. The
`usage.cache_creation_input_tokens` and
`usage.cache_read_input_tokens` fields are both zero on every
call, so it's not prompt caching either.

**Why this matters for anyone migrating:**

- **List-price parity is a mirage.** Opus 4.7 is listed at the
  same $15/Mtok input as Opus 4.6. At 1.43× tokenization, the
  effective rate on real prompts is $21.45/Mtok input.
- **Enterprise contracts that renegotiate $/token without
  retesting $/prompt will be wrong.** If your annual committed
  spend is denominated in tokens and you default-upgrade your
  model, you'll hit your cap ~30% sooner than last year's budget
  suggests.
- **Rift's `$/correct` column is the right denominator.** On this
  suite: $0.1815 (4.6) → $0.2444 (4.7), a **+35% cost per fully
  correct answer** even accounting for 4.7's (non-significant)
  quality lift. The quality lift isn't priced in; the token
  inflation is.

## Scale estimate

For a production pipeline at 10M daily input tokens on Opus 4.6,
naive upgrade to 4.7 moves you to ~14.3M tokens/day for
byte-identical prompts. At list: **+$195/day input cost
(~$71k/year) with zero workload change**.

Your mileage will vary — the inflation ratio depends on what's in
your prompts (the 1.45× figure here is on English corporate-filler
distractor context; code-heavy or multilingual workloads could
differ substantially). The point is to measure it before you
migrate, not after.

## What Rift caught that a casual eval wouldn't

The first run of this benchmark produced **"opus-4-7: 0/32, $0,
-100% drift"** — a catastrophic-looking regression. It was a
tooling bug: opus-4-7 deprecates `temperature`/`top_p`/`top_k` and
rejects suites that pass them with a 400. The runner correctly
classified 400s as non-retryable, but the reporter rendered the
all-errored result as real drift. Two fixes landed as a result:

1. A per-model `DEPRECATED_PARAMS` map in the Anthropic provider
   silently strips knobs the model no longer honors, preserving
   paired determinism (the dropped params weren't being applied
   by the model anyway).
2. The report now carries an `Errors` column + a top-of-page
   warning when any model had API errors. A drift report on an
   error-tainted run is biased downward by construction; the
   tooling refuses to let it look clean.

## What Enterprise token-pricing changes

Two things change on Enterprise contracts, neither of which is
addressed by list-price parity:

1. **Tokenizer changes aren't a line item in the contract.** Your
   negotiated `$/Mtok` stays the same; your `tokens/prompt`
   silently grows ~40%.
2. **`$/correct` is the only workload-normalized comparison.**
   Rift's `--enterprise-multiplier` applies your negotiated rate
   uniformly; the delta in `$/correct` is then directly
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
   side.** If the 8k/32k directional improvement holds at n≥50, it
   becomes publishable. The suite is intentionally small so the
   reproducibility bar is low; run it against your own data.
2. **This is one suite.** Reasoning with corporate-filler
   distractors isn't the only shape of enterprise workload.
   Extraction, summarization, and code-gen tokenize differently
   and may show different inflation ratios.
3. **Enterprise contracts may include caching discounts that
   this suite doesn't exercise.** A multi-turn workload with
   significant prompt overlap could recover some of the tokenizer
   inflation through prompt caching — but the base rate still
   moves.

---

_Rift is MIT-licensed. Run it against your own data before you
decide whether to upgrade. The numbers will change when the models
do._
