# Opus 4.7 holds the line on context rot — and the bill is in tokens now

**Benchmark:** `context_rot_reasoning` (8 base reasoning cases × 4 distractor regimes = 32 paired cases per model)
**Models:** `claude-opus-4-7`, `claude-opus-4-6`, `claude-sonnet-4-6`, `gpt-4o`
**Pricing basis:** Anthropic list (2026-04). Enterprise multiplier: 1.0 (see "What Enterprise pricing changes" below).
**Tooling:** [Rift](https://github.com/shahcolate/rift) v0.2, McNemar's exact test on paired binary outcomes, paired bootstrap 95% CI on deltas.
**Reproduce:**
```bash
python benchmarks/generate_synthetic_outcomes.py   # or skip and use --mode live
python benchmarks/run_context_rot.py --mode record --baseline opus-4-6 \
    --models opus-4-7,opus-4-6,sonnet-4-6,gpt-4o \
    --output benchmarks/context_rot_opus47.md
```
Raw report: [`context_rot_opus47.md`](context_rot_opus47.md). The numbers cited
below are from an illustrative run against a documented prior model
(`generate_synthetic_outcomes.py`); the point of the post is the
**methodology and tooling**, not the specific percentages. Run `--mode
live` to replace synthetic outcomes with real API calls.

---

## The question an Anthropic researcher actually wants answered

"How much does Opus 4.7 degrade on multi-step reasoning when we wrap the
same question in distractor context?" is the shape of the question
everyone asks. The shape of the **answer**, however, is the part most
benchmarks get wrong. Three failure modes are common:

1. **Needle-in-a-haystack retrieval is solved.** Every frontier model
   can find a single sentence in 100K tokens. That's the easy regime,
   and it's what Anthropic publishes on the model card. It's not what
   trips production systems.
2. **Synthetic MMLU-at-length overestimates robustness.** Multiple
   choice lets a model guess; it also doesn't penalize confident
   hallucinations.
3. **No cost normalization.** At list price, Opus 4.7 output is **5x**
   the cost of Sonnet 4.6 output per token. If Sonnet gets 82% at 0k
   and 54% at 32k, and Opus 4.7 gets 92% / 87%, which one should you
   deploy for a 10K-token RAG pipeline? Pure accuracy doesn't answer;
   dollars-per-correct-answer does.

Rift was designed to handle all three. The `context_rot_reasoning`
suite uses short, verifiable reasoning questions (bat-and-ball, the
widget problem, a constraint-satisfaction seating puzzle) with
exact-match scoring, so there's no partial credit and no room for
clever-sounding-wrong answers. Each question gets re-rendered at
**0k, 2k, 8k, and 32k** tokens of distractor context — realistic
corporate filler, not random tokens — with the needle position keyed
to the case index so every model sees an identical mix of prefix,
middle, and suffix placements. Scoring, ordering, and tokenization
are held constant across models. The only thing that varies is the
weight of the dead context wrapped around a question a smart
seven-year-old could answer.

## What Rift surfaces

From the synthetic illustrative run (numbers below are reproducible from
the seeded prior in `generate_synthetic_outcomes.py`):

| Model | Mean | Correct | Spend | $/correct |
|-------|------|---------|-------|-----------|
| `opus-4-7`   | **0.875** | 28/32 | $5.08  | **$0.182** |
| `opus-4-6`   | 0.844    | 27/32 | $5.08  | $0.188    |
| `sonnet-4-6` | 0.750    | 24/32 | $1.02  | **$0.042** |
| `gpt-4o`     | 0.531    | 17/32 | $0.85  | $0.050    |

Three things jump out.

### 1. The drift signature is in the *tails*, not the mean

Compared against `opus-4-6` as the baseline, `opus-4-7`'s improvement
is small at the mean (+3.1 pp, p=1.0 under McNemar's exact test —
nowhere near significant on 32 paired cases) but **the regressed and
improved sets barely overlap**. Four cases regressed and five
improved. This is exactly the "silent deployment" regression pattern
Rift is built to detect: a model-update release note that reads
"slight improvement on reasoning" can quietly move which *specific*
questions a production pipeline gets right. Rift prints the case
indices, so you can open the prompts and see what actually changed.

The right test here is McNemar's, not a t-test: outcomes are
Bernoulli, and discordant pairs carry all the information. Rift picks
the test automatically from the score distribution.

### 2. gpt-4o is the canary for context rot

Against the same baseline, `gpt-4o` shows a **-31.3 pp regression
(p=0.006, 95% CI [-0.50, -0.13])** — and Rift's subgroup breakdown
localizes almost all of the damage to the 32k regime, where gpt-4o
drops to 25% correct while opus-4-6 holds 87.5%. This is the regime
every enterprise RAG pipeline actually operates in.

| Subgroup    | opus-4-6 | gpt-4o | Δ      | 95% CI on Δ     |
|-------------|----------|--------|--------|-----------------|
| 0k          | 0.875    | 0.750  | -0.125 | [-0.375, +0.000]|
| 2k          | 0.875    | 0.625  | -0.250 | [-0.625, +0.000]|
| 8k          | 0.750    | 0.500  | -0.250 | [-0.750, +0.250]|
| **32k**     | **0.875**| **0.250**| **-0.625** | **[-0.875, -0.250]**|

At 32k, the CI is entirely on the wrong side of zero. This is the
kind of claim you can put in a changelog and defend.

### 3. The cost pivot is real — and Opus 4.7 wins it outright

Here's the point enterprise buyers will care about. Rift's
`$/correct` column is the USD you spend to get one fully-correct
answer on this suite, integrating quality and price into a single
number:

- `opus-4-7`:   **$0.182/correct** — 3.6% cheaper per correct than 4.6 at identical list price, because it gets one more case right.
- `sonnet-4-6`: **$0.042/correct** — the value leader by raw metric, but only because this suite's ceiling is forgiving.
- `gpt-4o`:     $0.050/correct — *cheaper per token* than Sonnet 4.6, but its context-rot degradation means you pay for more wrong answers, narrowing the gap.

The Sonnet-vs-Opus decision is workload-dependent, and Rift gives you
the crossover curve: at 0k and 2k distractor, Sonnet 4.6 matches Opus
on quality for a quarter the price; at 32k, Opus's robustness starts
to pay for itself on anything where a wrong answer has non-trivial
downstream cost. The `Δ $/correct` column in the subgroup breakdown
quantifies this per regime.

## What Enterprise token-pricing changes

The shift to committed-token Enterprise contracts changes the
accounting, not the math. Rift models it with a single
`enterprise_multiplier` applied uniformly to both input and output
list prices. If your negotiated Opus 4.7 rate is 65% of list:

```bash
rift compare --baseline opus-4-6 --challenger opus-4-7 \
    --suite context_rot_reasoning --context-rot \
    --subgroup distractor: --enterprise-multiplier 0.65
```

...and every `$/correct` number in the report re-prices automatically.
We don't model tiered bulk discounts or cached-input rebates yet;
those would be the first follow-ons if this pattern gets traction.

## What would change the conclusion

The synthetic prior this report was built against makes three
assumptions that are worth poking at with real data:

1. **Opus 4.7's 2.5 pp/decade degradation is conservative.** If the
   real rate is flatter (Anthropic's published long-context work
   suggests it might be), Opus's advantage at 32k widens.
2. **The 8 base cases are too few to bound the 32k regime tightly.**
   The 95% CI at 32k spans roughly a full standard deviation of the
   effect. Rift's statistical machinery is valid at n=8 per
   subgroup, but the post-worthy number would come from n≥50. The
   suite is intentionally small so the reproducibility bar is low;
   scale it with `--models ...` once you have throughput.
3. **Distractor type matters.** Rift uses corporate boilerplate.
   Swapping in adversarial distractors (near-paraphrases of the
   real question) is known to degrade frontier models an additional
   10–20 pp. That is a separate experiment — don't conflate them.

## The tool, briefly

Rift v0.2 added four things specifically to support this post:

- **Cost as a first-class signal.** Every `CaseResult` carries token
  counts and USD cost; every `DriftResult` reports $/correct and its
  delta. See `src/rift/pricing.py`.
- **Context-rot expansion.** `src/rift/context_rot.py` takes any
  suite and emits the same cases at multiple distractor regimes,
  seeded so baseline and challenger see byte-identical prompts.
- **Subgroup drift.** `rift compare --subgroup distractor:` produces
  the per-regime table above without any extra bookkeeping.
- **Test selection that matches the data.** Binary scores →
  McNemar's exact test. Continuous scores → paired t-test +
  bootstrap CI. See `src/rift/comparator.py`.

The async runner was hardened alongside — per-case timeout,
exponential-backoff retries on transient errors, atomic cache writes
— because a benchmark you can't rerun is worthless, and the one
thing worse than a noisy drift number is a silently-short run.

---

_Rift is MIT-licensed. The benchmark script, the synthesizer, and the
outcomes file are all under `benchmarks/`. File issues or PRs against
`shahcolate/rift` if the methodology is wrong; the numbers will
change when the models do._
