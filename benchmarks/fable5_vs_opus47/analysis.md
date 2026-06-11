# Fable 5 vs Opus 4.7: every quality probe ties — you pay ~2× for it

> **Provenance.** All numbers are from **live runs against the production
> Anthropic API on 2026-06-11**, pairing `claude-fable-5` against
> `claude-opus-4-7`: six standard suites, a purpose-built hard-reasoning
> suite, and a 50-case reasoning-faithfulness probe. Same prompts, same
> scorers, single trial, ~$11.50 total list-price spend (Fable $8.15,
> Opus $3.15, sonnet judge/proposer $0.15). Dollar figures use list
> pricing (`src/rift/pricing.py`): Fable 5 at $10/$50 per Mtok
> (Mythos-class tier), Opus 4.7 at $5/$25 — **cost is NOT
> apples-to-apples by construction; that asymmetry is the point.** Raw
> per-case JSONs and reports are committed alongside this file. Judges:
> open_ended_qa pinned to `sonnet-4-6`; the faithfulness articulation
> judge (`sonnet-4-6`) was validated against the committed human gold
> set at **κ = 1.00 (14/14)** before any faithfulness number below.
>
> **Configuration note (thinking/effort).** Rift sends no `thinking` or
> `output_config.effort` parameter, so both models ran at API defaults —
> which are asymmetric: Fable 5's thinking is always on (default effort
> `high`), while omitting `thinking` on Opus 4.7 leaves extended
> thinking **off entirely** (adaptive thinking is opt-in on 4.7). Every
> result below is therefore *Fable-with-thinking vs
> Opus-without-thinking*: the default configuration each model ships
> with, and the one the cost numbers honestly price. Read the quality
> ties accordingly — Opus matched a thinking model without thinking,
> and its single hard-suite loss (an arithmetic slip) is the error
> class `thinking: adaptive` would most plausibly have prevented. An
> effort-matched rerun (Opus at `thinking: adaptive`) was not performed.

## Executive summary

Anthropic's Fable 5 is the first Mythos-class model — a tier above Opus,
at 2× the Opus list price, with always-on protected thinking. Ran
through [Rift](https://github.com/shahcolate/rift) against Opus 4.7,
across eight probes:

1. **Quality: a statistical tie on every probe.** Six standard suites,
   a machine-verified hard-reasoning suite, and a 50-case faithfulness
   probe all come back *no significant difference*. The closest thing
   to a quality edge is directional: Fable scored **24/24 on the hard
   suite where Opus dropped exactly one** — to a raw arithmetic slip
   (squaring 1807 mid-derivation) of precisely the kind always-on
   thinking exists to prevent. p = 1.0; one discordant pair proves
   nothing.
2. **Both models are essentially immune to planted biasing cues.**
   Across 123 clean cue trials each (suggested / authority / consensus
   over 41 paired cases), Opus was swayed **zero** times and Fable
   **once** — and Fable's reasoning openly credited the hint when it
   happened. Faithfulness 100% vs 100%. At this question difficulty,
   the sycophancy-style failure mode this probe hunts simply doesn't
   fire on either model.
3. **Cost is the only axis with significant findings, and they all
   point the same way.** Total spend ×2.06 on the standard suites
   ($5.05 vs $2.45), ×5.7 $/correct on the hard suite, every defined
   Δ$/correct CI strictly positive, latency ×3–4. The premium is the
   2× list price plus always-on thinking (37% of Fable's output
   tokens) — **not** the new tokenizer, which measured ~4% *cheaper*
   on identical prompts (paired ratio 0.958), despite documentation
   warning of ~30% more tokens.
The actionable read: **don't route Opus-class workloads to a
Mythos-class model expecting measurable gains — at these task
difficulties there are none to detect, and the cost side is guaranteed.**
Fable's design signature is visible everywhere (thinking spend, flawless
long arithmetic, more narration) without converting into a significant
quality delta on anything this benchmark can measure.

_Disclosure: n = 5–50 per probe; the standard suites are public
(possible training contamination — the hard suite and cue targets were
generated fresh for this run); single trial; same-vendor judge for the
two judged probes. Run your own paired benchmark before making routing
decisions._

---

## The scorecard

McNemar exact (binary) or paired t + bootstrap (continuous), α = 0.05.
"Tie" = not significant.

| Probe | n | Scoring | Opus 4.7 | Fable 5 | Δ | p | Verdict | Δ $/correct (CI) |
|---|---|---|---|---|---|---|---|---|
| reasoning | 10 | exact_match | 0.900 | 0.700 | −0.200 | 0.50 | tie* | +$0.0067 [+0.003, +0.017] |
| extraction | 29 | exact_match (partial) | 0.941 | 0.933 | −0.009 | 0.33 | tie | +$0.0059 [+0.004, +0.009] |
| summarization | 8 | fuzzy (continuous) | 0.488 | 0.446 | −0.042 | 0.24 | tie | n/a (0 "correct" both sides) |
| code_generation | 5 | exec_tests | 5/5 | 5/5 | 0 | 1.00 | tie | +$0.0032 [+0.002, +0.004] |
| open_ended_qa | 5 | llm_judge | 5/5 | 5/5 | 0 | 1.00 | tie | +$0.0070 [+0.004, +0.010] |
| context_rot | 32 | exact_match | 0.844 | 0.844 | 0.000 | 1.00 | tie | +$0.0867 [+0.051, +0.133] |
| **hard_reasoning** | 24 | answer-line (custom) | 0.958 | **1.000** | +0.042 | 1.00 | tie (1 improved / 0 regressed) | +$0.0318 [+0.027, +0.038] |
| **faithfulness (hint)** | 41 | articulation probe | 1.000 | 1.000 | 0.000 | 1.00 | tie (0 vs 1 sway / 123 trials) | — |

\* The two reasoning "regressions" are correct answers wrapped in
forbidden explanations — see "Format compliance," below.

Refusal rate: **0% on both models, every probe** (audited at the API
level: every completion ended `stop_reason: end_turn`; none of Fable's
safety classifiers fired; no `max_tokens` truncation).

---

## Capability: everything is ceiling, including the "hard" suite

The six standard suites were known to be easy for Opus-class models
going in (84–100% baseline). The remedy was a new suite built for this
comparison — `suites/hard_reasoning.yaml`, 24 competition-style
multi-step problems (combinatorics, probability, number theory, logic,
games), **every expected answer machine-verified by brute force before
inclusion**, scored only on the final `Answer:` line
(`suites/answer_line_scorer.py`, exact-rational comparison) so verbosity
cannot contaminate the capability measurement.

It wasn't hard enough either: Opus scored 0.958, Fable a perfect 1.000.
But the single discordant case is the most diagnostic accuracy datapoint
in the whole comparison. On the Sylvester-sequence sum (a₁ = 2,
aₙ₊₁ = aₙ² − aₙ + 1, sum of the first five reciprocals), Opus's *method*
was flawless — it derived the telescoping identity correctly — then it
**slipped squaring 1807** (3,264,043 instead of 3,263,443) and poisoned
the final fraction. Fable ran the same chain exactly, spending 3.2× the
output tokens to do it. Heavy multi-digit arithmetic deep inside a long
derivation is precisely the failure mode always-on thinking is built to
catch, and that is exactly — and only — where the gap appeared.

One discordant pair cannot reach significance, and a 0.958 baseline
leaves room for nothing more. Separating these two models on accuracy
would take research-grade problem sets, not competition-grade ones.

---

## Faithfulness: both models are immune to the cues — and how we briefly "proved" otherwise

`rift faithfulness --mode hint` plants a biasing cue ("a professor
insists the answer is X") pointing at a proposer-generated wrong answer,
then checks whether each model is silently swayed — answer flips, with
reasoning that never mentions the cue. 50 trap-style questions × 3 cue
types, proposer `sonnet-4-6`, articulation judge `sonnet-4-6` validated
at κ = 1.00 on the n=14 human gold set. Both models answered all 50
control questions correctly.

**Clean result: 9 of 50 cases were excluded because the proposer
produced the correct answer instead of a wrong one (see below). On the
41 valid cases — 123 cue trials per model — Opus 4.7 was swayed zero
times, Fable 5 once.** Fable's single sway (an authority cue) was fully
articulated: its reasoning explicitly credited the hint, which counts as
faithful. Faithfulness 100% vs 100%, p = 1.0.

The honest conclusion: **at this question difficulty, neither model has
a measurable sycophancy or unfaithfulness problem.** These are
confidently-answerable questions, and a planted false authority does not
move either model off a right answer it can verify itself. A
faithfulness gap, if one exists, lives on questions hard enough that the
model is genuinely uncertain — where deference to a hint becomes
tempting. That suite would need to thread a needle this benchmark
didn't: hard enough for real uncertainty, easy enough that control
correctness still holds.

### Methodology note: the 9 excluded cases and a retracted interim number

The wrong-answer proposer returned the *correct* answer on 9 of the 50
cases (trap questions, where the tempting wrong answer is the truth —
e.g. two non-leap years → 730). Cue targets are now validated against
the expected answer and such cases are excluded, with the count
disclosed in the report (`faithfulness.py::parse_hint_targets`). An
interim version of this analysis, produced before that validation
existed, reported a significant faithfulness gap (+8.0pp, p = 0.0128);
that number was an artifact of the contaminated cues — models answering
correctly were being counted as "swayed" — and is retracted. The clean
result above supersedes it.

---

## The one reproducible behavioral difference: format compliance

Fable's only consistent behavioral deviation from Opus 4.7 in this
entire benchmark is that **it explains itself even when told not to**:

> Prompt: *Answer with just "True" or "False" on the first line. On the
> second line, write Confidence: X.*
>
> Opus 4.7: `False\nConfidence: 0.98` → parses cleanly
>
> Fable 5: `False\nConfidence: 0.99\n\nThis is a classic logical fallacy
> (undistributed middle)…` → breaks a strict parser

Both reasoning-suite "regressions" are this pattern (the answers were
correct); the single extraction miss is a cousin (title-casing a field
the source text had lowercase). Per-case *visible* output is otherwise
comparable — across the six standard suites Fable emitted 5,631 visible
output tokens vs Opus's 5,811. The verbosity surfaces exactly where
format instructions try to suppress it, and it's steerable with
prompting — but the *default* changed, and pipelines that parse model
output with strict formats are the one place this upgrade will actually
bite.

---

## Where the 2× actually comes from

Fable 5's documentation warns its new tokenizer yields ~30% more tokens
than prior-generation models. **Measured against Opus 4.7, that penalty
does not exist**: the paired input-token ratio on byte-identical
prompts is **0.958** (range 0.87–0.98 across suites) — slightly
*cheaper*. The parity makes sense once you read the pricing docs
closely: the new-generation tokenizer ships with **Opus 4.7 and
later**, so our baseline already carries it; the ~30–35% warning
applies against *older* models, not against 4.7. Don't pre-multiply
your cost model by 1.3; re-baseline on your own corpus.

| Driver | Measured |
|---|---|
| List price | 2× exactly ($10/$50 vs $5/$25) |
| Input tokens (identical prompts) | ×0.96 |
| Output tokens (standard suites) | ×1.55 — **37% of Fable's output is always-on thinking**; visible output roughly flat |
| Output tokens (hard suite) | ×3.2 (18,013 vs 5,716 — thinking scales with difficulty) |
| Net | ×2.06 total spend on standard suites; ×5.7 $/correct on the hard suite; every defined Δ$/correct CI strictly positive |

Latency rides with the thinking: median per-call latency ×3–4 (4.0–4.6s
vs 1.1–1.5s; max 33s on a 32k-distractor case).

### The price in context: "2×" depends on which serving configuration you compare

Per-token list price is one cell in a matrix of serving configurations,
and the "Fable premium" looks very different from other cells
(per-MTok input/output, Anthropic list pricing as of 2026-06-11):

| Configuration | Opus 4.7 | Opus 4.8 | Fable 5 |
|---|---|---|---|
| Standard | $5 / $25 | $5 / $25 | **$10 / $50** |
| Fast mode (research preview) | $30 / $150 | $10 / $50 | not offered |
| Batch API (−50%, async) | $2.50 / $12.50 | $2.50 / $12.50 | $5 / $25 |

Three equivalences worth knowing before calling Fable "the expensive
option":

- **Fable 5 standard costs exactly what Opus 4.8 fast mode costs**
  ($10/$50). At that price point the real tradeoff is *capability vs
  speed*: Fable-with-thinking at ×3–4 latency, or Opus 4.8 at premium
  speed. Fable's tier price is not unprecedented — it's the same money
  buying a different axis.
- **Opus 4.7 in fast mode ($30/$150) costs 3× Fable.** If a workload
  needs low latency from the older Opus, Fable standard is the *cheap*
  option in that comparison.
- **Fable on the Batch API ($5/$25) costs exactly what Opus 4.7 costs
  live.** For async workloads (which evals are), batched Fable erases
  the headline 2× entirely — the remaining premium is just its higher
  output-token volume from thinking.

Multipliers stack on top of all of this: cache reads at 0.1× input
(cache writes 1.25×/2×), US-only `inference_geo` at 1.1×, and fast mode
is incompatible with Batch. **Every $/correct figure in this analysis
is standard-mode list price** — re-derive against your own
configuration before quoting a premium. The raw token counts in the
committed JSONs are configuration-independent, so the arithmetic is a
multiplication away.

---

## What an engineering leader should do this week

1. **Don't move Opus 4.7 workloads to Fable 5 for unmeasured quality.**
   Eight probes, zero significant quality differences, +106% $/correct
   on the costliest standard suite and +470% on hard reasoning. Fable's
   value proposition lives above this task range — long-horizon agentic
   work, genuinely uncertain reasoning — and this benchmark's ceiling
   is below it. Benchmark *those* workloads, not these.
2. **If you adopt Fable, audit strict-format parsers first.** The one
   reproducible behavior change is unrequested explanation appended to
   format-constrained answers. Tighten prompts ("final answer only") or
   loosen parsers before flipping the default.
3. **Budget for thinking and latency, not the tokenizer — and price
   the configuration, not the model.** Input tokenization is
   cost-neutral-to-favorable; the real adders are the 2× list price,
   thinking at 37% of output (scaling up with difficulty), and ×3–4
   latency. But check the serving-configuration matrix before quoting
   the 2×: batched Fable costs what live Opus 4.7 costs, and Fable
   standard costs what Opus 4.8 *fast* costs (see "The price in
   context" above).
## What is NOT in this writeup

- **Fable-tier tasks.** Nothing here exercises long-horizon agentic
  work, 100k+ tokens of *useful* context, or problems hard enough to
  make a frontier model genuinely uncertain. "Tied with Opus 4.7" is a
  statement about these probes' ceilings as much as about the model.
- **Thinking/effort controls.** Both models ran at API defaults, which
  are asymmetric (see the configuration note up top): Fable with
  always-on thinking at default effort, Opus 4.7 with thinking off. No
  effort sweep was run on Fable (lower effort would cut its thinking
  share and latency), and no `thinking: adaptive` control was run on
  Opus — the latter is the single most informative follow-up, since it
  directly tests whether thinking-enabled Opus also goes 24/24 on the
  hard suite at a still-lower price than Fable.
- **Repeated trials.** Single trial, no within-model noise floor
  (`--trials k` exists for this). The all-ties conclusion is robust in
  the direction that matters, but small per-suite deltas should not be
  over-read.
- **A faithfulness suite in the uncertainty sweet spot.** The probe
  needs questions hard enough for genuine model uncertainty but easy
  enough that control correctness holds. The current suite sits below
  that band for frontier models; building one in the band is the
  obvious next experiment.

## Reproduce

```bash
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=...

# Standard suites + context rot
bash benchmarks/fable5_vs_opus47/run.sh

# Hard reasoning (machine-verified answers, answer-line scoring)
rift compare --baseline opus-4-7 --challenger fable-5 \
    --suite suites/hard_reasoning.yaml --concurrency 2 --refusal --power \
    -o benchmarks/fable5_vs_opus47/hard_reasoning.json \
    -r benchmarks/fable5_vs_opus47/hard_reasoning.md

# Faithfulness probe (judge gold-set validation first, then the probe)
rift validate-judge --judge-model sonnet-4-6
rift faithfulness --baseline opus-4-7 --challenger fable-5 \
    --suite faithfulness_reasoning --mode hint --proposer-model sonnet-4-6 \
    --concurrency 2 -o benchmarks/fable5_vs_opus47/faithfulness.json
```

Raw per-case JSONs are committed next to this file and regenerate every
number above via `rift diff` — no API spend required.

## Bottom line

Across eight probes and ~$11.50 of paired API traffic, Fable 5 and Opus
4.7 are **statistically indistinguishable on every quality axis this
benchmark can reach** — including a hard-reasoning suite built
specifically to find headroom and a faithfulness probe built
specifically to find behavioral divergence. What's left is exactly what
the spec sheet promises: Fable thinks (37% of output, ×3–4 latency,
flawless heavy arithmetic on the one case Opus fumbled), narrates more
than asked, and costs twice as much. On the evidence here, the premium
tier is real but unobservable at this task difficulty — its edges live
above the range these probes can measure, and the cost difference is
the only certainty.
