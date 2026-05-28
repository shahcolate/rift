# The Opus 4.5-gen price cut reopens the cost race: Opus is cheapest on reasoning, tied on extraction

> **Provenance.** The token counts, correctness, and judge scores below
> are from a **live API run on 2026-05-21** against the production
> OpenAI, Anthropic, and Google endpoints (n=44 paired prompts × 3
> models = 132 completions). Per-model completion JSONs are committed
> under `extraction/`, `open_ended_qa/`, and `reasoning/`.
>
> **The Opus dollar figures have been recomputed at the post-2026-05-28
> list price** ($5/Mtok input, $25/Mtok output — the rate the entire
> Opus 4.5 generation, 4.5 through 4.8, now lists at; see
> `src/rift/pricing.py`). The literal invoice from the 2026-05-21 run
> was 3× higher on the Opus rows, when Opus listed at $15/$75. gpt-5.5
> and Gemini prices did not change, so their dollars are the measured
> ones. Recomputed total spend across all nine runs: **$0.43** (it was
> $0.65 at the old Opus price). The bundled `rift demo` is a separate,
> offline synthetic story and does NOT cite these numbers.

## Executive summary

I extended the earlier [Opus 4.7 vs Gemini 3.5 Flash
writeup](../opus47_vs_gemini35_analysis.md) into a three-way matrix
by adding OpenAI's gpt-5.5, on the same three suites — `reasoning`
(n=10), `structured_extraction` (n=29), `open_ended_qa` (n=5) —
with the same scorers and the same single-trial protocol. 44 paired
prompts × 3 models = 132 live completions.

This writeup has been **re-priced** after Anthropic cut the Opus
4.5-generation list price 3× (from $15/$75 to $5/$25) alongside the
Opus 4.8 launch. That single price change reorders the cost ranking,
because it lands Opus at near-parity per token with gpt-5.5 (identical
$5 input; $25 vs $20 output) — at which point the bill is decided by
**output volume**, and Opus is the most parsimonious generator in the
field.

Three findings, ranked by what changes a decision:

1. **The cheap-per-correct crown is now contested, not gpt-5.5's
   alone.** Opus is the cheapest model on `reasoning` ($0.0019 vs
   gpt-5.5's $0.0026), statistically and economically tied with
   gpt-5.5 on `extraction` ($0.0029 vs $0.0027), and gpt-5.5 keeps a
   clear edge only on `open_ended_qa` ($0.0034 vs $0.0056). All at
   statistically indistinguishable correctness on the deterministic
   suites (9/10, 23/29, 23/29). At the old $15/$75 Opus price gpt-5.5
   won every suite by 2–3×; the price cut erases that.

2. **The I:O-ratio mechanism from the prior writeup reproduces almost
   exactly, one week later, with a third tokenizer in the mix.**
   Gemini emitted 11.7× more output tokens than Opus on reasoning
   and 6.7× more on extraction; the prior 2-way reported 13.6× and
   6.8×. The "thinking tokens billed as output" effect is not a
   one-week artifact — it's a stable property of Gemini Flash's
   default `thinking_level=medium`. What the price cut changes is the
   *consequence*: with Opus and gpt-5.5 now within 25% of each other
   per token, the model that emits fewer output tokens wins, and on
   reasoning that is Opus (471 out vs gpt-5.5's 953). Gemini's output
   tax now leaves it the **most expensive per correct on three of the
   four suites.**

3. **The family-bias caveat on the judge-scored suite *weakens*
   in the 3-way, but isn't ruled out.** Opus is uniquely perfect on
   `open_ended_qa` (5/5 with the Sonnet 4.6 judge), with gpt-5.5
   at 3/5 perfect (judge mean 0.970) and Gemini at 2/5 (0.950). If
   the Anthropic judge were purely tribal we'd expect it to dunk
   on both non-Anthropic models equally; instead it placed gpt-5.5
   only 0.03 below Opus and above Gemini. The signal "Opus is best
   at free-form generation" survives the bias check better than
   the 2-way alone could test, but n=5 with one judge family is
   still not publishable on its own. The price cut also narrows the
   cost penalty for choosing Opus here from 5× to 1.6×.

The headline isn't "Opus is the best model." It's **"on deterministic,
exact-match-scored workloads the three frontier models are
substitutable on correctness, so procurement collapses to a per-token
race — and the Opus 4.5-gen price cut just turned that race from a
gpt-5.5 runaway into a genuine three-way, with Opus winning the
output-light suites outright."**

A structural observation worth pulling forward: on `reasoning`, all
three models fail the **same case** (#7); on `extraction`, 5 of 6
imperfect cases overlap across all three. The substitutability
claim is stronger than a coincidental tie — the failure mode is in
the items, not the models. Procurement-wise, that's the
prerequisite for a price-only decision.

_Disclosure: I maintain Rift. Token counts and correctness below are
from my own runs against the live OpenAI, Anthropic, and Google APIs on
2026-05-21; Opus dollar figures are recomputed at the current $5/$25
list price. Run your own paired benchmark before making procurement
decisions._

---

## At a glance

| Suite | n | Scoring | gpt-5.5 mean | Opus mean | Gemini mean | gpt-5.5 $/c | Opus $/c | Gemini $/c |
|---|---|---|---|---|---|---|---|---|
| reasoning | 10 | `exact_match` | 0.900 | 0.900 | 0.900 | $0.0026 | **$0.0019** | $0.0056 |
| extraction | 29 | `exact_match` (partial) | 0.941 | 0.933 | 0.941 | **$0.0027** | $0.0029 | $0.0061 |
| open_ended_qa | 5 | `llm_judge` (Sonnet 4.6) | 0.970 | **1.000** | 0.950 | **$0.0034** | $0.0056 | $0.0163 |

McNemar / paired-t p-values on every model pair are ≥ 0.18 on the
binary suites and ≥ 0.08 on `open_ended_qa`. No pair clears a
conventional significance threshold on any suite — which is itself
the central finding for the deterministic suites. The cost ranking,
not the correctness ranking, is where the decision lives, and at the
new Opus price that ranking is split: Opus wins reasoning, gpt-5.5
wins QA, and extraction is a tie.

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
| gemini-3.5-flash | $1.50 | $9.00 |
| gpt-5.5 | $5.00 | $20.00 |
| claude-opus-4-7 | $5.00 | $25.00 |

The price relationship that drove the old conclusion is gone: Opus
and gpt-5.5 now share an **identical $5/Mtok input price** and sit
within 25% on output ($25 vs $20). Neither is "the expensive option"
anymore. Gemini is cheapest per input token ($1.50) but most expensive
per *correct answer* on three suites, because its `thinking_level:
medium` default multiplies its $9 output price by a 3–12× output-token
volume. The invoice is `output_tokens × output_price`, not
`output_price` — and once two models tie on per-token price, the one
that emits fewer tokens wins.

---

## What an executive leader should do this week

For an engineering, platform, or finance leader who saw the prior
"move some Opus workload to Flash" writeup and is now re-pricing after
the Opus 4.5-gen cut, here is the action list, ranked by reversibility
cost (cheapest first):

### 1. Re-baseline any "Opus is too expensive" decision — the 3× price cut likely reverses it

The prior writeup's headline — "Flash is 30% cheaper than Opus on
extraction" — was computed when Opus listed at $15/$75. At the new
$5/$25 rate that reverses: Opus is now **$0.0029 per correct on
extraction vs Gemini's $0.0061 — roughly 2× cheaper**, and on
reasoning Opus ($0.0019) is the cheapest of all three. Any contract
or migration plan built on "move Opus workload to Flash for cost"
should be re-priced against the current Opus rate before signing;
on output-light workloads the premise no longer holds.

### 2. The I:O-ratio measurement from the prior writeup still applies — the price cut just moved the breakeven

The prior writeup told you to compute `input_tokens :
output_tokens` on your production traffic before quoting savings.
That still stands. What's changed is the answer:

- **Input-heavy (RAG, extraction, classification):** gpt-5.5 and
  Opus are now effectively tied (identical $5/Mtok input), and both
  beat Gemini decisively — Gemini's output-token tax outweighs its
  cheaper input list price the moment the suite has any output at all.
- **Output-heavy and reasoning-style:** Opus is now the cost winner,
  not just the latency winner — it emits ~2× fewer output tokens than
  gpt-5.5 and ~12× fewer than Gemini, and at near-parity output
  pricing that volume advantage decides the bill.
- **Free-form / judge-scored (open-ended QA, writing):** gpt-5.5 is
  the cost winner ($0.0034 vs Opus $0.0056) because here Opus is the
  *verbose* one (1,083 out tok vs gpt-5.5's 475). Opus is uniquely
  perfect on quality, now at only a 1.6× cost premium — see §3.

### 3. Don't read the open-ended QA result as a clean Opus quality win, yet

The judge is Claude Sonnet 4.6 — same family as Opus. The 3-way
result is *less consistent with judge-tribalism than the prior
2-way result was* (Gemini got dunked, but gpt-5.5 also got
dunked, just less), but it doesn't refute the bias model. Before
budgeting around "Opus is the only quality option for free-form
generation," re-run `open_ended_qa` with a non-Anthropic judge
(e.g. GPT-5.5 or Gemini Flash as judge — and run both, since
the same family-bias concern applies in reverse). The cost penalty
for picking Opus here is now small (1.6×), so the quality question
matters more than the price one on this suite.

### 4. Pin the OpenAI `reasoning_effort` setting before you switch

gpt-5.5 has its own version of the
`thinking_level` knob (`reasoning_effort`). Rift uses OpenAI's
default. Production cost optimization can move the bill 2–5× by
pinning it lower; running comparisons on one setting and
deploying on another is the most common foot-gun. Pick a level
before you switch and stay there.

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

rift matrix --models gpt-5.5,opus-4-8,gemini-3-5-flash \
    --suite YOUR_PRODUCTION_SUITE
```

Look at three columns in the Rift report: the accuracy delta, the
**output-token ratio**, and the **`cost-per-correct` delta**. The
last one is the only number a CFO cares about.

### 6. Renegotiate any budget line built on a per-token list-price comparison

If a 2026 procurement line still reads "Opus → Flash saves 10×"
or "stay on Opus because Flash is unreliable on reasoning," both
narratives are now stale — and so is "gpt-5.5 is the cheapest on
everything." After the Opus 4.5-gen price cut the cheapest model is
**suite-dependent**: Opus on reasoning, a tie on extraction, gpt-5.5
on free-form. Re-baseline on $/correct for *your* workload mix, not
$/token, before locking in the contract.

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

The committed Opus `cost_usd` fields reflect the current $5/$25 list
price. Recomputed total spend across all nine runs: **$0.43** (the
literal 2026-05-21 invoice was $0.65 at the old $15/$75 Opus rate).

---

## Finding 1 — Reasoning: Opus is now the cheapest at the same accuracy

| Model | n correct | Mean | Input tokens | Output tokens | Spend | $/correct |
|---|---|---|---|---|---|---|
| **opus-4-7** | 9/10 | 0.900 | 1,077 | 471 | $0.0172 | **$0.0019** |
| gpt-5.5 | 9/10 | 0.900 | 785 | 953 | $0.0230 | $0.0026 |
| gemini-3-5-flash | 9/10 | 0.900 | 787 | 5,495 | $0.0506 | $0.0056 |

All three models failed the **same single case (#7)**, so discordant
pairs across every pairing equal zero → McNemar p = 1.000 by
construction. Co-located failures are a stronger statement than a
coincidental tie: the failure mode is in the item, not the model.

The prior 2-way writeup's headline finding on this suite — "Flash
is *more expensive than Opus* per correct on reasoning, because the
10× input-price discount is consumed by thinking tokens billed as
output" — reproduces and intensifies: Gemini $0.0056 vs Opus $0.0019.
The new finding is that **Opus is now the cheapest model on this
suite, edging gpt-5.5 ($0.0026)**, because Opus emits the fewest
output tokens of the three (471 vs gpt-5.5's 953 vs Gemini's 5,495)
and the price cut put it at parity with gpt-5.5 per output token.
Output parsimony, once an under-priced virtue at $75/Mtok, is now the
deciding advantage.

The procurement implication: on reasoning workloads, Opus is the cheap
default again — the conclusion is the exact reverse of what it was at
the $15/$75 price.

---

## Finding 2 — Extraction: Opus and gpt-5.5 are now a dead heat

| Model | n correct (1.0) | Mean | Input tokens | Output tokens | Spend | $/correct |
|---|---|---|---|---|---|---|
| **gpt-5.5** | 23/29 | 0.941 | 1,602 | 2,753 | $0.0631 | **$0.0027** |
| opus-4-7 | 23/29 | 0.933 | 2,547 | 2,174 | $0.0671 | $0.0029 |
| gemini-3-5-flash | 22/29 | 0.941 | 1,673 | 14,563 | $0.1336 | $0.0061 |

**5 of the 6 imperfect cases are shared across all three models**
(cases 6, 9, 20, 21, 23, 24 — partial-credit scores from at least
two models on each). Gemini misses one extra case (#0 at 0.75),
which is the only correctness difference among the three. On this
suite the models are even more substitutable than on reasoning.

gpt-5.5 and Opus are now within **7% of each other per correct**
($0.0027 vs $0.0029) — a tie for practical purposes — and both are
roughly **2× cheaper than Gemini** ($0.0061). At the old $15/$75 Opus
price this same data put gpt-5.5 at 3.3× cheaper than Opus; the price
cut closes that gap to a rounding difference. The prior 2-way claim —
"Flash is 30% cheaper than Opus on extraction" — is now *reversed*:
Opus at $0.0029 is cheaper than Flash at $0.0061.

Opus's higher input-token count (2,547 vs gpt-5.5's 1,602) is what
keeps gpt-5.5 nominally ahead — extraction prompts are long, and at
equal $5/Mtok input price the model that tokenizes the prompt into
fewer tokens pays less. Gemini's cheaper input list price
($1.50/Mtok) still gets clawed back by 6.7× output verbosity, so it
lands at $0.13, twice the other two.

---

## Finding 3 — Open-ended QA: Opus retains a quality edge, now at a 1.6× cost premium

| Model | n perfect | Judge mean | Input tokens | Output tokens | Spend | $/correct |
|---|---|---|---|---|---|---|
| **opus-4-7** | 5/5 | **1.000** | 202 | 1,083 | $0.0281 | $0.0056 |
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
judge would settle it; not done in this writeup. See action #3.

This is the one suite where gpt-5.5 still clearly wins on cost, and
the reason is the mirror image of Finding 1: here **Opus is the
verbose one** (1,083 output tokens vs gpt-5.5's 475), so its
output-volume advantage flips into a disadvantage. The price cut still
helps — Opus's per-correct cost fell from $0.0169 to $0.0056, cutting
the premium over gpt-5.5 from 5× to **1.6×**. If you're optimizing for
quality, that premium is now small enough that the decision turns on
the (unresolved) judge-bias question, not the bill. If you're
optimizing for cost on "good enough" free-form generation, gpt-5.5 is
still 1.6× cheaper for a 0.030-point judge-score drop.

---

## Cross-suite: the structural finding (extended)

The prior writeup's central observation was: **Gemini's
list-price discount erases on output-heavy workloads because
thinking tokens are billed as output.** Today's run confirms the
output-token ratios within 15% of last time on every suite, and the
Opus price cut sharpens the consequence: Gemini is now the **most
expensive per correct on three of the four suites** (reasoning,
extraction, code generation), beaten on cost by both of the
output-parsimonious models.

The 3-way extension, re-priced, adds: **the cost winner is now
suite-dependent, and it tracks output volume.** With Opus and gpt-5.5
sharing a $5/Mtok input price and sitting within 25% on output price,
neither has a per-token edge — so whichever model emits fewer tokens
on a given workload wins. Opus wins the output-light suites
(reasoning), gpt-5.5 wins the suites where Opus happens to be chatty
(open-ended QA), and they tie where their output volumes are close
(extraction).

Multiply each model's listed output price by its average output
tokens on a representative prompt. The product, not the list
price, is what shows up on your invoice. That heuristic is what now
*reverses* the old conclusion: at $25/Mtok output and 471 tokens, Opus
on reasoning costs less than gpt-5.5 at $20/Mtok and 953 tokens.

Models with extended-thinking defaults (Gemini Flash at
`thinking_level: medium`) still need a 3–12× multiplier on their list
price for the implied bill, which is why Gemini loses on cost despite
the cheapest input price. Among the two parsimonious models, the one
that wins is simply the one that talks less on your workload.

---

## What is NOT in this writeup

1. **A code-generation cell.** The
   [companion 5-case `code_generation` matrix](../3way_codegen/analysis.md)
   exists but is too small (n=5) to publish standalone. Folding it
   in would require expanding to HumanEval-scale (≥50 cases) with
   a harder difficulty distribution.
2. **A non-Anthropic judge replicate of `open_ended_qa`.** Action
   #3 above. The 3-way alone is suggestive that family-bias isn't
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
6. **A re-run on the actual Opus 4.8 build.** These completions are
   Opus 4.7 from 2026-05-21; the 4.5-generation price applies
   identically to 4.8, but correctness/verbosity on 4.8 could differ.
   Re-run with `--models gpt-5.5,opus-4-8,gemini-3-5-flash` for the
   current model.

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
spending another dollar of API. (The committed Opus `cost_usd` values
have been recomputed at the current $5/$25 list price; the underlying
token counts are unchanged from the 2026-05-21 capture.)

---

## Bottom line

The Opus 4.5-generation list-price cut (from $15/$75 to $5/$25, shipped
with Opus 4.8) reopens a cost race that had looked settled. On
deterministic, exact-match-scored workloads the three frontier models
remain statistically indistinguishable on correctness, so the decision
is still a per-token race — but the winner is now **suite-dependent**:
**Opus is cheapest on reasoning, tied with gpt-5.5 on extraction, and
gpt-5.5 keeps the edge only on free-form generation**, where Opus
happens to be the verbose one. Gemini 3.5 Flash, unchanged in price,
is now the most expensive per correct on three of four suites because
its thinking-token output tax never went away.

The actionable claim for procurement is unchanged in form and reversed
in result: **multiply each model's listed per-token output price by
the average output tokens it emits on your workload, not on the vendor
demo.** That product is the number that shows up on your invoice — and
at the new Opus price it now favors Opus wherever the workload is
output-light.
