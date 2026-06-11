# Fable 5 vs Opus 4.7: quality is a statistical tie — you pay ~2× for it

> **Provenance.** Numbers below are from a **live API run on 2026-06-11**
> against the production Anthropic API, pairing `claude-fable-5` against
> `claude-opus-4-7` on the same prompts, same scorers, single trial.
> Dollar figures are at list price (`src/rift/pricing.py`): Fable 5 at
> $10/$50 per Mtok (Mythos-class tier), Opus 4.7 at $5/$25 — so unlike
> the Opus-family comparisons, **cost here is NOT apples-to-apples by
> construction**; that asymmetry is the point. Raw per-case JSONs and
> per-suite drift reports are committed alongside this file. The
> open_ended_qa judge is pinned to `sonnet-4-6` (neither contestant).

## Executive summary

Anthropic's Fable 5 is the first Mythos-class model — a tier *above*
Opus, at 2× the Opus list price, with always-on (protected) thinking. So
I ran it through [Rift](https://github.com/shahcolate/rift) against Opus
4.7 on six suites. Three results, in increasing order of interest:

1. **Quality: statistically tied on all six suites.** Reasoning,
   extraction, summarization, code generation, open-ended QA, and
   long-context distractor reasoning all come back *no significant
   drift*. On the context-rot suite — the one that caught Opus 4.8
   regressing last month — Fable 5 and Opus 4.7 land on **identical
   accuracy (0.844)** with a perfectly balanced discordant split
   (1 regressed / 1 improved, p = 1.0).
2. **Cost: the only confidence intervals in this entire comparison that
   exclude zero are the cost ones.** Total spend $5.05 vs $2.45
   (**2.06×**); on the context-rot suite $/correct is $0.1715 vs $0.0847
   (**+102%**, CI [+$0.05, +$0.13]). Every suite with a defined cost
   CI (summarization's is undefined — zero "correct" on both sides) is
   strictly positive. For these workloads, Fable 5 buys nothing
   measurable and costs double.
3. **A useful tooling stress-test happened by accident: the API
   account ran out of credits mid-run.** Anthropic bills exhausted
   credit as an HTTP **400** ("credit balance too low"), not a 429 —
   so 19 of 32 Fable calls in the first context-rot pass hard-failed,
   and Rift's `compare` path scored each one 0 without disclosing it,
   producing a phantom −46.9pp "regression" at p = 0.000061. After a
   top-up, a clean re-run (cache replays the successes, only the
   errored cases re-fetch) erased the entire effect. Two fixes shipped
   from this (see below). **Billing ≠ drift.**

The headline isn't "Fable 5 is bad." On these suites the frontier-tier
model is *at ceiling parity* with Opus 4.7 — these tasks don't reach the
capability range Fable is priced for. The actionable takeaway: **don't
route Opus-class workloads to a Mythos-class model expecting free
gains; measure, because the cost side is guaranteed and the quality
side, on ordinary tasks, is not.**

_Disclosure: n is small per suite (5–32); the suites are public
(possible training contamination); single trial; same-vendor judge
(sonnet-4-6) for the one judged suite. Run your own paired benchmark
before making routing decisions._

---

## The scorecard

All tests McNemar exact (binary) or paired t + bootstrap (continuous),
α = 0.05. "Tie" = not significant.

| Suite | n | Scoring | Opus 4.7 | Fable 5 | Δ | p | Verdict | Δ $/correct (CI) |
|---|---|---|---|---|---|---|---|---|
| reasoning | 10 | exact_match | 0.900 | 0.700 | −0.200 | 0.50 | tie* | +$0.0067 [+0.003, +0.017] |
| extraction | 29 | exact_match (partial) | 0.941 | 0.933 | −0.009 | 0.33 | tie | +$0.0059 [+0.004, +0.009] |
| summarization | 8 | fuzzy (continuous) | 0.488 | 0.446 | −0.042 | 0.24 | tie | n/a (0 "correct" both sides) |
| code_generation | 5 | exec_tests | 5/5 | 5/5 | 0 | 1.00 | tie | +$0.0032 [+0.002, +0.004] |
| open_ended_qa | 5 | llm_judge | 5/5 | 5/5 | 0 | 1.00 | tie | +$0.0070 [+0.004, +0.010] |
| context_rot | 32 | exact_match | 0.844 | 0.844 | 0.000 | 1.00 | tie | +$0.0867 [+0.051, +0.133] |

\* The two reasoning "regressions" are not wrong answers — see next
section.

Refusal rate: **0% on both models, every suite** (audited at the API
level: all 178 completions ended `stop_reason: end_turn`; none of
Fable's safety classifiers fired, no `max_tokens` truncation).

## The real behavioral drift: format compliance, not capability

Both reasoning cases Fable "lost" look like this:

> Prompt: *Answer with just "True" or "False" on the first line. On the
> second line, write Confidence: X.*
>
> Opus 4.7: `False\nConfidence: 0.98` → scored 1.0
>
> Fable 5: `False\nConfidence: 0.99\n\nThis is a classic logical fallacy
> (undistributed middle). While all roses are flowers…` → scored 0.0

Fable answered **correctly**, then appended an explanation the prompt
explicitly forbade. The exact-match scorer (like any deployed parser
expecting a bare answer) breaks on the extra prose. The single
extraction regression is the same species: Fable title-cased a field
("Remote within EU" vs the source text's "remote within EU").

This matches Fable's documented behavioral profile (more user-facing
narration than Opus 4.7). Two practical notes:

- **If your pipeline parses model output with strict formats, this is
  the drift that will actually bite you** — not accuracy. It's
  steerable with prompting, but the *default* changed.
- Per-case **visible** output is otherwise comparable: across all 89
  paired completions Fable emitted 5,631 visible output tokens vs
  Opus's 5,811. The verbosity shows up exactly where format
  instructions try to suppress it.

## Where the 2× actually comes from (tokenizer myth-check)

Fable 5 ships a new tokenizer documented as yielding "~30% more tokens"
for the same content. **Measured on these prompts, that penalty does
not exist**: the paired input-token ratio (identical input text, Fable
tokens / Opus tokens) is **0.958 overall** — Fable's tokenizer is
actually ~4% *more* efficient on this short-English-prose workload
(range 0.87–0.98 across suites). Don't pre-multiply your cost model by
1.3; re-baseline on your own corpus.

The measured cost decomposition for the 2.06× total spend:

| Driver | Measured |
|---|---|
| List price | 2× exactly ($10/$50 vs $5/$25) |
| Input tokens (same prompts) | ×0.96 (slightly cheaper) |
| Output tokens | ×1.55 overall — **of which 37% is always-on thinking** (3,353 of 8,984 tokens; visible output is roughly flat) |
| Net | ×2.06 total spend; $/correct +102% on the costliest suite |

Latency rides along with the thinking: Fable's median per-call latency
is ~3–4× Opus 4.7's (4.0–4.6s vs 1.1–1.5s median; max 33s on a
32k-distractor case).

## When the credit balance dies mid-benchmark: what it looks like in a drift report

Mid-way through the first context-rot pass, the API account's credit
balance ran out. Anthropic surfaces that as **HTTP 400**, not 429 — so
the runner's retry logic (correctly) treated it as non-transient and
each affected call failed once, permanently. The first-pass report
looked like this (preserved verbatim from the run log — do not cite
these numbers):

| Metric | First pass (bad) | Clean re-run |
|---|---|---|
| Fable accuracy | 0.375 | 0.844 |
| Δ | **−46.9pp, p = 0.000061** | 0.0pp, p = 1.0 |
| Regressed/improved | 15 / 0 | 1 / 1 |
| distractor:0k subgroup | Fable 0/8 | 6/8 |
| distractor:8k subgroup | Fable 6/8 | 7/8 |

19 of 32 Fable calls returned the 400 (Opus had zero — its half of the
suite ran before the balance hit bottom; every Fable call succeeded on
replay after a top-up). Rift scored each errored case 0 and the
report's status line read "🔴 Regression Detected" with no mention of
errors anywhere in the markdown.

Two tells that should have been (and now are) automatic:

1. **Inverted difficulty gradient.** A real long-context regression
   degrades *with* distractor size. This "regression" was worst at 0k
   and mildest at 8k — because error incidence, not capability, drove
   the scores.
2. **Error counts.** `metadata.n_errors` was 19/32 on one side and 0 on
   the other. An asymmetry like that is an availability event, not a
   model comparison.

Fixes shipped in this PR:

- **Drift reports now disclose errored-case counts** with an explicit
  warning that errors are indistinguishable from wrong answers in the
  stats (`reporter.py`). The observatory already excluded errored pairs
  ("outage ≠ drift"); `compare` now at least refuses to let them pass
  silently.
- **The Anthropic provider now preserves 4xx response bodies** in the
  raised error (`providers/anthropic.py`). The cause here took real
  digging to establish because `raise_for_status()` discarded the
  API's explanation ("credit balance is too low") — with the body
  attached, the first errored case would have named it immediately.

Open question for a follow-up: whether `compare` should exclude
errored-on-either-side pairs from the paired test outright, as
`observe` does. (Arguably yes, with the exclusion count disclosed.)

## What an engineering leader should do this week

1. **Don't move Opus 4.7 workloads to Fable 5 for quality you haven't
   measured.** On six ordinary suites the quality delta is zero and the
   cost delta is +106%, with every cost CI excluding zero. Fable's
   value proposition is capability *above* this range (long-horizon
   agentic work, hard reasoning) — benchmark those workloads, not these.
2. **If you do adopt Fable 5, audit your parsers first.** The one
   reproducible behavior change is unrequested explanation appended to
   format-constrained answers. Tighten prompts ("final answer only") or
   loosen parsers before flipping the default.
3. **Budget for thinking, not for the tokenizer.** On short prompts the
   new tokenizer is cost-neutral-to-favorable; the real adders are the
   2× list price and the ~37%-of-output thinking spend, plus 3–4×
   latency.
4. **Gate your drift dashboards on error counts.** Anything that fails
   API calls on one side of a paired comparison — an outage, a rate
   limit, or (as here) a credit balance hitting zero mid-run —
   manufactures arbitrarily significant "regressions" if errors are
   scored as zeros silently. Note the failure mode: exhausted credit
   is a **400**, so a retry-on-429 policy won't save you.

## What is NOT in this writeup

- **Fable-tier tasks.** Nothing here exercises long-horizon agentic
  work, 100k+ token *useful* context, or frontier reasoning — the
  things a Mythos-class model is for. "Tied with Opus 4.7" on these
  suites is a statement about the suites' ceiling as much as the model.
- **Effort sweeps.** Fable ran at its default effort. Lower effort
  would cut the thinking share and latency; higher might separate
  quality on harder tasks. Single point measured.
- **Repeated trials.** Single trial, no within-model noise floor
  (`--trials k` exists for this). The all-ties conclusion is robust to
  noise in the direction that matters (nothing significant to begin
  with), but the small per-suite deltas should not be over-read.

## Reproduce

```bash
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=...

bash benchmarks/fable5_vs_opus47/run.sh
```

Raw per-case JSONs (`*.json`) are committed next to this file and are
sufficient to regenerate every number above with `rift diff` — no API
spend required. Check the error-count warning at the top of any
regenerated report before citing it.

## Bottom line

Fable 5, on the evidence here, is **a frontier-tier model that ordinary
eval suites cannot distinguish from Opus 4.7** — identical accuracy on
long-context distractor reasoning, ties everywhere else, zero refusals
— at 2.06× the spend, 3–4× the latency, and with one real behavioral
regression for pipeline builders: it explains itself even when told not
to. And the run's most reusable lesson wasn't about either model: when
the account's credit balance ran out mid-run, the resulting 400s scored
as zeros and read as a p = 0.000061 "regression" until the inverted
difficulty gradient gave it away. Score your billing and availability
events separately from your drift — Rift's reports now disclose them.
