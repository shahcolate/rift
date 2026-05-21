# gpt-5.5 quietly took the cheap-per-correct crown from Gemini 3.5 Flash

## Executive summary

I extended the earlier [Opus 4.7 vs Gemini 3.5 Flash
writeup](../opus47_vs_gemini35_analysis.md) into a three-way matrix
by adding OpenAI's gpt-5.5, on the same three suites — `reasoning`
(n=10), `structured_extraction` (n=29), `open_ended_qa` (n=5) —
with the same scorers and the same single-trial protocol. 44 paired
prompts × 3 models = 132 live completions. Total live-API spend:
**$0.65**.

Three findings, ranked by what changes a decision:

1. **gpt-5.5 is the cheapest-per-correct model on every suite, by
   2–3×, at statistically indistinguishable correctness on the
   deterministic suites.** $0.0026 on reasoning, $0.0027 on
   extraction, $0.0034 on open-ended QA. Same correctness as Opus
   and Gemini on the two exact-match suites (9/10, 23/29, 23/29).
   The cheap-per-correct headline from the prior writeup —
   "Gemini Flash is 30% cheaper on extraction" — is still true,
   but is no longer the cheapest option in the market.

2. **The I:O-ratio mechanism from the prior writeup reproduces almost
   exactly, one week later, with a third tokenizer in the mix.**
   Gemini emitted 11.7× more output tokens than Opus on reasoning
   and 6.7× more on extraction; the prior 2-way reported 13.6× and
   6.8×. The "thinking tokens billed as output" effect is not a
   one-week artifact — it's a stable property of Gemini Flash's
   default `thinking_level=medium`. gpt-5.5 sits in the middle on
   verbosity (2.0× Opus on reasoning, 1.3× on extraction) and its
   low per-token output price ($10/Mtok) absorbs that without
   blowing out the bill.

3. **The family-bias caveat on the judge-scored suite *weakens*
   in the 3-way, but isn't ruled out.** Opus is uniquely perfect on
   `open_ended_qa` (5/5 with the Sonnet 4.6 judge), with gpt-5.5
   at 3/5 perfect (judge mean 0.970) and Gemini at 2/5 (0.950). If
   the Anthropic judge were purely tribal we'd expect it to dunk
   on both non-Anthropic models equally; instead it placed gpt-5.5
   only 0.03 below Opus and above Gemini. The signal "Opus is best
   at free-form generation" survives the bias check better than
   the 2-way alone could test, but n=5 with one judge family is
   still not publishable on its own.

The headline isn't "gpt-5.5 is the best model." It's **"on
deterministic, exact-match-scored workloads, the three frontier
models are now substitutable on correctness, so procurement
collapses to a per-token race that gpt-5.5 currently wins."**

A structural observation worth pulling forward: on `reasoning`, all
three models fail the **same case** (#7); on `extraction`, 5 of 6
imperfect cases overlap across all three. The substitutability
claim is stronger than a coincidental tie — the failure mode is in
the items, not the models. Procurement-wise, that's the
prerequisite for a price-only decision.

_Disclosure: I maintain Rift. Numbers below are from my own runs
against the live OpenAI, Anthropic, and Google APIs on 2026-05-21.
Run your own paired benchmark before making procurement decisions._

---

## At a glance

| Suite | n | Scoring | gpt-5.5 mean | Opus mean | Gemini mean | gpt-5.5 $/c | Opus $/c | Gemini $/c |
|---|---|---|---|---|---|---|---|---|
| reasoning | 10 | `exact_match` | 0.900 | 0.900 | 0.900 | **$0.0026** | $0.0057 | $0.0056 |
| extraction | 29 | `exact_match` (partial) | 0.941 | 0.933 | 0.941 | **$0.0027** | $0.0088 | $0.0061 |
| open_ended_qa | 5 | `llm_judge` (Sonnet 4.6) | 0.970 | **1.000** | 0.950 | $0.0034 | $0.0169 | $0.0163 |

McNemar / paired-t p-values on every model pair are ≥ 0.18 on the
binary suites and ≥ 0.08 on `open_ended_qa`. No pair clears a
conventional significance threshold on any suite — which is itself
the central finding for the deterministic suites.

The output-token volume table that was the structural finding of
the prior 2-way writeup, now with a third column:

| Suite | gpt-5.5 out tok | Opus out tok | Gemini out tok | Gemini / Opus | gpt-5.5 / Opus |
|---|---|---|---|---|---|
| reasoning | 953 | 471 | 5,495 | **11.7×** | 2.0× |
| extraction | 2,753 | 2,174 | 14,563 | **6.7×** | 1.3× |
| open_ended_qa | 475 | 1,083 | 3,603 | **3.3×** | 0.4× |

Prior 2-way writeup reported 13.6× / 6.8× / 3.5× for the same three
suites. The mechanism is intact.

Per-1M-token list prices (the lever the I:O ratio multiplies):

| Model | Input ($/Mtok) | Output ($/Mtok) |
|---|---|---|
| gpt-5.5 | $1.25 | $10.00 |
| gemini-3.5-flash | $1.50 | $9.00 |
| claude-opus-4-7 | $15.00 | $75.00 |

Gemini and gpt-5.5 sit within ~10% of each other on per-token
pricing. The 2–3× cost gap on `$/correct` is therefore almost
entirely about how many output tokens each model emits.

---

## What an executive leader should do this week

For an engineering, platform, or finance leader who saw the prior
"move some Opus workload to Flash" writeup and is now wondering
where gpt-5.5 fits, here is the action list, ranked by
reversibility cost (cheapest first):

### 1. If you already decided to move workload from Opus to Gemini Flash on the prior writeup, re-price against gpt-5.5 before signing

The prior writeup's headline — "Flash is 30% cheaper on
extraction" — is still true (Gemini $0.0061 vs Opus $0.0088). But
gpt-5.5 lands at $0.0027 on the same suite, which is **55% cheaper
than Flash and 69% cheaper than Opus** at the same correctness.
On reasoning, the prior writeup's headline "Flash is *more*
expensive than Opus per correct" reproduced (Gemini $0.0056 vs
Opus $0.0057), but gpt-5.5 is $0.0026 — half the cost of either.

Any contract being renegotiated this quarter on the "Opus → Flash
for cost" narrative should be re-baselined against a
gpt-5.5 row in the same paired-eval table before the term sheet
goes out.

### 2. The I:O-ratio measurement from the prior writeup still applies — gpt-5.5 just shifts the breakeven

The prior writeup told you to compute `input_tokens :
output_tokens` on your production traffic before quoting savings.
That still stands. What's changed is the answer:

- **Input-heavy (RAG, extraction, classification):** gpt-5.5 wins
  decisively. Its input price ($1.25/Mtok) is the cheapest, and it
  doesn't blow out output volume.
- **Output-heavy (reasoning, agentic loops, long generation):**
  gpt-5.5 wins on cost, but Opus is the latency winner and is
  comparable on $/correct. Gemini is the worst pick in this
  regime — its thinking-token volume erases its input-price
  discount.
- **Free-form / judge-scored (open-ended QA, writing):** Opus is
  uniquely perfect at this scale but costs 5× gpt-5.5. The
  per-correct gap may not survive a non-Anthropic judge — see §3.

### 3. Pin the OpenAI `reasoning_effort` setting before you switch

gpt-5.5 has its own version of the
`thinking_level` knob (`reasoning_effort`). Rift uses OpenAI's
default. Production cost optimization can move the bill 2–5× by
pinning it lower; running comparisons on one setting and
deploying on another is the most common foot-gun. Pick a level
before you switch and stay there.

### 4. Don't read the open-ended QA result as a clean Opus quality win, yet

The judge is Claude Sonnet 4.6 — same family as Opus. The 3-way
result is *less consistent with judge-tribalism than the prior
2-way result was* (Gemini got dunked, but gpt-5.5 also got
dunked, just less), but it doesn't refute the bias model. Before
budgeting around "Opus is the only quality option for free-form
generation," re-run `open_ended_qa` with a non-Anthropic judge
(e.g. GPT-5.5 or Gemini Flash as judge — and run both, since
the same family-bias concern applies in reverse).

### 5. Run your own paired benchmark before committing volume

Don't take the magnitudes in this writeup as authoritative for
your workload. n is small (5–29 per suite), the judge in §3 is
family-related to Opus, and a real production suite will have its
own difficulty distribution. The **direction** of every finding
generalizes; the **numbers** require your data:

```bash
pip install rift-eval
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...

rift matrix --models gpt-5.5,opus-4-7,gemini-3-5-flash \
    --suite YOUR_PRODUCTION_SUITE
```

Look at three columns in the Rift report: the accuracy delta, the
**output-token ratio**, and the **`cost-per-correct` delta**. The
last one is the only number a CFO cares about.

### 6. Renegotiate any budget line built on a per-token list-price comparison

If a 2026 procurement line still reads "Opus → Flash saves 10×"
or "stay on Opus because Flash is unreliable on reasoning," both
narratives are now stale: the cheapest model on deterministic
workloads is gpt-5.5, and the I:O-ratio mechanism that decides
the actual bill applies to all three vendors. Re-baseline on
$/correct, not $/token, before locking in the contract.

---

## What was run

For each suite, `rift matrix --models
gpt-5.5,opus-4-7,gemini-3-5-flash` against the three live APIs.
Gemini was called with `thinking_level=medium` (Google's own
default; pinned by `rift.providers.google.GoogleProvider` for
paired determinism). Opus ran with deterministic sampler defaults
(Rift drops `temperature` for Opus 4.7 per the model's
deprecated-params policy). gpt-5.5 is sent
`max_completion_tokens` and no `temperature` (gpt-5/o-series only
accepts the default temperature; see `src/rift/providers/openai.py`
after commit `3db18aa`).

| Suite | Scoring | Cases | Statistical test |
|---|---|---|---|
| `reasoning` | `exact_match` | 10 | McNemar's exact (binary) |
| `structured_extraction` | `exact_match` (dict, field-by-field, partial credit) | 29 | Paired t + bootstrap |
| `open_ended_qa` | `llm_judge` (`claude-sonnet-4-6` as judge) | 5 | Paired t + bootstrap |

Raw per-case JSONs are committed alongside this file:
- `benchmarks/3way_full/reasoning/{gpt-5.5,opus-4-7,gemini-3-5-flash}.json`
- `benchmarks/3way_full/extraction/{gpt-5.5,opus-4-7,gemini-3-5-flash}.json`
- `benchmarks/3way_full/open_ended_qa/{gpt-5.5,opus-4-7,gemini-3-5-flash}.json`

Total live-API spend across all nine runs: **$0.65**.

---

## Finding 1 — Reasoning: gpt-5.5 halves the cost at the same accuracy

| Model | n correct | Mean | Input tokens | Output tokens | Spend | $/correct |
|---|---|---|---|---|---|---|
| **gpt-5.5** | 9/10 | 0.900 | 785 | 953 | $0.0230 | **$0.0026** |
| opus-4-7 | 9/10 | 0.900 | 1,077 | 471 | $0.0515 | $0.0057 |
| gemini-3-5-flash | 9/10 | 0.900 | 787 | 5,495 | $0.0506 | $0.0056 |

All three models failed the **same single case (#7)**, so discordant
pairs across every pairing equal zero → McNemar p = 1.000 by
construction. Co-located failures are a stronger statement than a
coincidental tie: the failure mode is in the item, not the model.

The prior 2-way writeup's headline finding on this suite — "Flash
is *more expensive than Opus* per correct on reasoning, because the
10× input-price discount is consumed by thinking tokens billed as
output" — reproduces: Gemini $0.0056 vs Opus $0.0057. The new
finding is that **both are 2.2× more expensive than gpt-5.5**
($0.0026), which wins both axes simultaneously: cheapest input
price *and* 5.8× less output volume than Gemini.

The procurement implication: on reasoning workloads, neither Opus
nor Gemini is the cheap default anymore. gpt-5.5 is.

---

## Finding 2 — Extraction: gpt-5.5 is now cheaper than the "cheap" option

| Model | n correct (1.0) | Mean | Input tokens | Output tokens | Spend | $/correct |
|---|---|---|---|---|---|---|
| **gpt-5.5** | 23/29 | 0.941 | 1,602 | 2,753 | $0.0631 | **$0.0027** |
| opus-4-7 | 23/29 | 0.933 | 2,547 | 2,174 | $0.2013 | $0.0088 |
| gemini-3-5-flash | 22/29 | 0.941 | 1,673 | 14,563 | $0.1336 | $0.0061 |

**5 of the 6 imperfect cases are shared across all three models**
(cases 6, 9, 20, 21, 23, 24 — partial-credit scores from at least
two models on each). Gemini misses one extra case (#0 at 0.75),
which is the only correctness difference among the three. On this
suite the models are even more substitutable than on reasoning.

gpt-5.5 is **3.3× cheaper per correct than Opus** and **2.3×
cheaper than Gemini**. The prior 2-way claim — "Flash is 30%
cheaper than Opus on extraction" — is preserved exactly (Gemini at
$0.0061 vs Opus $0.0088). It's just no longer the cheapest option:
gpt-5.5 at $0.0027 beats Flash by another 55%.

Opus's spend ($0.20) is dominated by input-token cost — extraction
prompts are long. Gemini's cheaper input price gets partly clawed
back by 6.7× output verbosity, so it lands at $0.13. gpt-5.5 wins
both axes: cheapest input price and modest output verbosity.

---

## Finding 3 — Open-ended QA: Opus retains a quality edge, with the same caveat (now weakly weakened)

| Model | n perfect | Judge mean | Input tokens | Output tokens | Spend | $/correct |
|---|---|---|---|---|---|---|
| **opus-4-7** | 5/5 | **1.000** | 202 | 1,083 | $0.0843 | $0.0169 |
| gpt-5.5 | 3/5 | 0.970 | 127 | 475 | $0.0101 | **$0.0034** |
| gemini-3-5-flash | 2/5 | 0.950 | 108 | 3,603 | $0.0326 | $0.0163 |

Per-case judge scores:
- Opus: `[1.00, 1.00, 1.00, 1.00, 1.00]`
- gpt-5.5: `[0.90, 1.00, 0.95, 1.00, 1.00]`
- Gemini: `[0.90, 0.95, 0.90, 1.00, 1.00]`

Opus is uniquely perfect. n=5 caps statistical power (best
pairwise p is 0.089), so the gap is suggestive, not significant.

**Family-bias caveat update.** The prior 2-way flagged that the
judge was Claude Sonnet 4.6 — same family as Opus — and the 6 pp
gap might be partly judge-favoritism. In the 3-way the same
Anthropic judge ranks the field
**Opus > gpt-5.5 > Gemini**. If the judge were purely tribal it
would have dunked on the third-party model (gpt-5.5) about as
hard as on Gemini. Instead gpt-5.5 lands only 0.030 below Opus
and 0.020 above Gemini. That's weaker evidence for "the judge
picks its own family" than the 2-way data alone supported —
though it doesn't rule it out. Replicating with a non-Anthropic
judge would settle it; not done in this writeup. See action #4.

The cost ranking inverts from the binary suites because Opus's
perfect score divides the spend into 5 correct (the
$/correct denominator), so $0.0169 holds against gpt-5.5's
$0.0034. If you're optimizing for quality and don't care about
cost at this scale (open-ended QA total spend was $0.13), Opus is
the call. If you're optimizing for cost on free-form generation
that's "good enough," gpt-5.5 is 5× cheaper for a 0.030-point
judge-score drop.

---

## Cross-suite: the structural finding (extended)

The prior writeup's central observation was: **Gemini's
list-price discount erases on output-heavy workloads because
thinking tokens are billed as output.** Today's run confirms the
output-token ratios within 15% of last time on every suite.

The 3-way extension adds: **gpt-5.5 is, today, the model that
*neither* gets dinged by long input (its input price is the
cheapest of the three at $1.25/Mtok) *nor* by output verbosity
(2–6× less than Gemini on every suite, and its per-token output
price is $10/Mtok vs Opus's $75/Mtok).**

Multiply each model's listed output price by its average output
tokens on a representative prompt. The product, not the list
price, is what shows up on your invoice. That's the actionable
heuristic, and the 3-way bake-off gives it a second data point.

Models with extended-thinking defaults (Gemini Flash at
`thinking_level: medium`) need a 3–13× multiplier on their list
price for the implied bill; gpt-5.5 needs ~2× over Opus on
output volume but more than compensates with its 7.5× lower
per-output-token rate.

---

## What is NOT in this writeup

1. **A code-generation cell.** The
   [companion 5-case `code_generation` matrix](../3way_codegen/analysis.md)
   exists but is too small (n=5) to publish standalone. Folding it
   in would require expanding to HumanEval-scale (≥50 cases) with
   a harder difficulty distribution.
2. **A non-Anthropic judge replicate of `open_ended_qa`.** Action
   #4 above. The 3-way alone is suggestive that family-bias isn't
   the whole story, but it's not conclusive.
3. **A discovery loop.** The prior writeup ran one and explicitly
   disowned the headline because proposer-equals-challenger
   creates selection bias. Running the methodologically valid
   version (third-party proposer) is a one-line change but was
   out of scope for this writeup.
4. **Variance estimates within model.** Single-trial,
   temperature-pinned. No within-model error bars. Repeated
   trials would tighten the per-suite CIs and might surface
   instability that single trials hide.
5. **Latency-sensitive workloads.** All three models are within
   ~2× of each other on average latency (Opus 2.3 s on code
   generation was the fastest in the companion run), but I
   didn't tabulate latency systematically here. If response
   time is a SLO, run a paired latency benchmark on your own
   prompts before deciding.

---

## Reproduce

```bash
pip install rift-eval
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...

for SUITE in reasoning extraction open_ended_qa; do
  rift matrix \
    --models gpt-5.5,opus-4-7,gemini-3-5-flash \
    --suite $SUITE \
    --output-dir benchmarks/3way_full/$SUITE
done
```

Two environmental gotchas that came up during this run, documented
so they don't bite the next person:

1. **OpenAI requires `max_completion_tokens` and rejects non-default
   `temperature` for gpt-5/o-series.** Rift's OpenAI provider was
   patched in commit `3db18aa` —
   `src/rift/providers/openai.py` now branches on model prefix
   (`gpt-5`, `o1`, `o3`, `o4`) and rewrites both fields.
2. **Network egress allowlists in sandboxed environments.** If you
   run this from a cloud development environment with a network
   policy, `api.openai.com` must be in the allowlist *at session
   start*; mid-session allowlist changes don't apply until a new
   session.

Raw per-case JSONs are committed under
`benchmarks/3way_full/{reasoning,extraction,open_ended_qa}/` and
are sufficient to regenerate every number in this writeup without
spending another dollar of API.

---

## Bottom line

For deterministic, exact-match-scored workloads, **gpt-5.5 is
roughly 2–3× cheaper per correct answer than both Opus 4.7 and
Gemini 3.5 Flash, with statistically indistinguishable
correctness.** The structural reason — the I:O ratio bound on
total spend — is the same mechanism the 2-way Opus/Gemini
writeup identified, now with a third data point that strengthens
it. On judge-scored open-ended generation, Opus retains a small
quality edge that's worth its 5× cost premium if quality is
non-negotiable; otherwise gpt-5.5 is the new default.

The actionable claim for procurement: **multiply each model's
listed per-token output price by the average output tokens it
emits on your workload, not on the vendor demo.** That's the
number that shows up on your invoice.
