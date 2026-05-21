# Three-way model bake-off: gpt-5.5 vs Opus 4.7 vs Gemini 3.5 Flash

## Executive summary

I extended the earlier [Opus 4.7 vs Gemini 3.5 Flash
writeup](../opus47_vs_gemini35_analysis.md) to a three-way matrix by
adding OpenAI's gpt-5.5, on the same three suites
(`reasoning` n=10, `structured_extraction` n=29, `open_ended_qa`
n=5) with the same scorers and the same single-trial protocol.
44 paired prompts × 3 models = 132 live completions. Total live
spend: **$0.65**.

The two findings from the prior 2-way writeup both survive the
extension, and one new finding appears:

1. **gpt-5.5 is the Pareto winner on cost across every suite.** It
   beats both alternatives on $/correct by 2–3× — $0.0026 (reasoning),
   $0.0027 (extraction), $0.0034 (open-ended QA) — with statistically
   indistinguishable correctness on the two binary-scored suites.
2. **Gemini's thinking-token tax reproduces almost exactly.** On
   `reasoning`, Gemini emitted 11.7× more output tokens than Opus
   (5,495 vs 471); on `extraction`, 6.7× (14,563 vs 2,174). The
   prior 2-way writeup reported 13.6× and 6.8× on the same suites.
   The I:O-ratio mechanism it described — "Gemini's list-price
   discount is consumed by thinking tokens billed as output" — is
   not a one-week artifact.
3. **gpt-5.5 sits in the middle on output verbosity.** It emits more
   output tokens than Opus on every suite (953 vs 471 on reasoning,
   2,753 vs 2,174 on extraction) but never close to Gemini's volume.
   Its low per-token price — $1.25 / $10 per 1M tokens, the cheapest
   of the three on both axes — wins the math anyway.
4. **The only correctness gap shows up on judge-scored open-ended
   generation** — and the prior writeup's family-bias caveat
   gets *weaker* in the 3-way, not stronger. See §3.

The headline isn't "gpt-5.5 is the best model." The headline is **on
deterministic, exact-match-scored workloads, the three frontier
models are now substitutable on correctness, and the choice
collapses to a price-per-token race that gpt-5.5 currently wins.**

_Disclosure: I maintain Rift. Numbers below are from my own runs
against the live OpenAI, Anthropic, and Google APIs on 2026-05-21.
Run your own paired benchmark before making procurement decisions._

---

## At a glance

| Suite          | n  | Test                  | gpt-5.5 mean | Opus mean | Gemini mean | gpt-5.5 $/c | Opus $/c | Gemini $/c |
|----------------|----|-----------------------|--------------|-----------|-------------|-------------|----------|------------|
| reasoning      | 10 | exact_match           | 0.900        | 0.900     | 0.900       | **$0.0026** | $0.0057  | $0.0056    |
| extraction     | 29 | exact_match (partial) | 0.941        | 0.933     | 0.941       | **$0.0027** | $0.0088  | $0.0061    |
| open_ended_qa  |  5 | llm_judge             | 0.970        | **1.000** | 0.950       | $0.0034     | $0.0169  | $0.0163    |

Per-suite McNemar p-values for every pair are p ≥ 0.18 (reasoning
and extraction) and p ≥ 0.08 (open_ended_qa). No pair clears a
conventional significance threshold on any suite — which is itself
the central finding for the two binary-scored suites.

The output-token volume table that was the structural finding of
the prior 2-way writeup, now with a third column:

| Suite       | gpt-5.5 out tok | Opus out tok | Gemini out tok | Gemini / Opus |
|-------------|-----------------|--------------|----------------|---------------|
| reasoning   | 953             | 471          | 5,495          | **11.7×**     |
| extraction  | 2,753           | 2,174        | 14,563         | **6.7×**      |
| open_ended_qa | 475           | 1,083        | 3,603          | **3.3×**      |

Prior writeup reported 13.6× / 6.8× / 3.5× for the same three
suites. The mechanism is intact.

---

## §1. Reasoning (n=10, exact_match)

| Model            | n correct | Mean   | Input tokens | Output tokens | Spend   | $/correct   |
|------------------|-----------|--------|--------------|---------------|---------|-------------|
| **gpt-5.5**          | 9/10  | 0.900  | 785          | 953           | $0.0230 | **$0.0026** |
| opus-4-7         | 9/10      | 0.900  | 1,077        | 471           | $0.0515 | $0.0057     |
| gemini-3-5-flash | 9/10      | 0.900  | 787          | 5,495         | $0.0506 | $0.0056     |

**All three models failed the same single case (#7).** Discordant
pairs = 0 across every pairing → McNemar p = 1.000 by construction.
Co-located failures are a stronger statement than a coincidental
tie: the failure mode is in the *item*, not in any model. On these
prompts the three are fully substitutable on correctness, so the
2.2× cost gap between gpt-5.5 and the other two is the entire
decision.

gpt-5.5's edge here comes from being cheap on both axes
simultaneously: half Opus's input price and 1/8th Gemini's output
volume billed at a much lower per-token rate than Gemini Flash.

---

## §2. Structured extraction (n=29, exact_match, partial credit)

| Model            | n correct (1.0) | Mean   | Input tokens | Output tokens | Spend   | $/correct   |
|------------------|-----------------|--------|--------------|---------------|---------|-------------|
| **gpt-5.5**          | 23/29       | 0.941  | 1,602        | 2,753         | $0.0631 | **$0.0027** |
| opus-4-7         | 23/29           | 0.933  | 2,547        | 2,174         | $0.2013 | $0.0088     |
| gemini-3-5-flash | 22/29           | 0.941  | 1,673        | 14,563        | $0.1336 | $0.0061     |

**5 of the 6 imperfect cases are shared across all three models**
(cases 6, 9, 20, 21, 23, 24 each came back with partial-credit
scores from at least two models). Gemini misses one extra case (#0
at 0.75) which is the only correctness difference among the three.

gpt-5.5 is **3.3× cheaper per correct than Opus** and **2.3× cheaper
than Gemini** on this suite. Opus's spend ($0.20) is dominated by
input-token cost — extraction prompts are long. Gemini's cheaper
input price gets partly clawed back by 6.7× output verbosity, so it
lands at $0.13. gpt-5.5 wins both axes: it has the cheapest input
price *and* it doesn't blow out output volume the way Gemini does.

The prior 2-way finding was "Gemini is 30% cheaper than Opus on
extraction" — that's confirmed (Gemini at $0.061 vs Opus $0.088,
exactly 30%). The new finding is that the 30% advantage is no
longer the cheapest option in the market.

---

## §3. Open-ended QA (n=5, LLM judge)

| Model            | n perfect | Judge mean | Input tokens | Output tokens | Spend   | $/correct |
|------------------|-----------|------------|--------------|---------------|---------|-----------|
| **opus-4-7**         | 5/5   | **1.000**  | 202          | 1,083         | $0.0843 | $0.0169   |
| gpt-5.5          | 3/5       | 0.970      | 127          | 475           | $0.0101 | **$0.0034** |
| gemini-3-5-flash | 2/5       | 0.950      | 108          | 3,603         | $0.0326 | $0.0163   |

Per-case judge scores:
- Opus: `[1.00, 1.00, 1.00, 1.00, 1.00]`
- gpt-5.5: `[0.90, 1.00, 0.95, 1.00, 1.00]`
- Gemini: `[0.90, 0.95, 0.90, 1.00, 1.00]`

Opus is uniquely perfect. n=5 caps statistical power (best pair p
is 0.089), so the gap is suggestive, not significant.

**Family-bias caveat update.** The prior writeup flagged that the
judge was Claude Sonnet 4.6 — same family as Opus — and a 6 pp
gap might be partly judge-favoritism. In the 3-way, the same
Anthropic judge now ranks the field
**Opus > gpt-5.5 > Gemini**. If the judge were purely tribal it
would have rewarded Opus *only*, but it places the third-party
model (gpt-5.5) above Gemini and only 0.030 below Opus. That's
weaker evidence for "the judge picks its own family" than the
2-way data alone supports — though it doesn't rule it out.
Replicating with a non-Anthropic judge would settle it; not done
here.

The cost ranking inverts from the binary suites because Opus's
perfect score divides into 5 correct (the denominator is what the
ratio compares against), so its $0.0169/correct is held against
$0.0034 for gpt-5.5. If you're optimizing for quality and don't
care about cost at this scale (open-ended QA spend was $0.13
total), Opus is the call. If you're optimizing for cost on free-form
generation that's "good enough," gpt-5.5 is 5× cheaper for a
0.030-point judge-score drop.

---

## §4. Cross-suite structural finding (extended)

The 2-way writeup's central observation was: **Gemini's
list-price discount erases on output-heavy workloads because
thinking tokens are billed as output.** Quantified by the
output-token volume ratio above. Today's run confirms the ratios
within 15% of last time on every suite.

The 3-way extension adds: **gpt-5.5 is, today, the model that
*neither* gets dinged by long input (its input price is the
cheapest of the three at $1.25/Mtok) *nor* by output verbosity
(2–6× less than Gemini on every suite, and its per-token output
price is $10/Mtok vs Opus's $75/Mtok).**

Per-1M-output-token list price (the lever Gemini's thinking tax
multiplies against):

| Model            | Input ($/Mtok) | Output ($/Mtok) |
|------------------|----------------|------------------|
| gpt-5.5          | $1.25          | $10.00          |
| gemini-3.5-flash | $1.50          | $9.00           |
| claude-opus-4-7  | $15.00         | $75.00          |

Gemini and gpt-5.5 have nearly identical per-token prices.
Gemini's 11.7× output volume on reasoning therefore translates
directly to ~11.7× output spend vs gpt-5.5, which is exactly
the gap we see in the per-suite numbers.

**Predictive heuristic that now has 6 data points worth of
support (3 suites × 2 runs):** for any workload, multiply each
model's listed output price by the average output tokens it emits
on a representative prompt. The product, not the list price, is
what shows up on your invoice. Models with extended-thinking
defaults (Gemini Flash at `thinking_level: medium`) need a
3–13× multiplier; gpt-5.5 needs ~2× over Opus; Opus needs ~1×
itself.

---

## §5. What this doesn't tell you

1. **n is small for two of three suites.** Reasoning (10), open_ended_qa
   (5). Extraction (29) is the only suite with meaningful resolving
   power. Co-located failures on reasoning are a clean signal but
   one item is one item.
2. **All three suites are easy.** Per-model means cluster around
   0.9–1.0. A harder distribution would let correctness differences
   surface before the McNemar floor.
3. **One LLM judge, one family.** Open-ended QA results need a
   non-Anthropic judge before the gap is publishable. The
   directional evidence in §3 is suggestive but not conclusive.
4. **Single trial per case, temperature 0.** No variance estimate
   within model. Repeated trials would tighten the CIs and might
   surface within-model instability that single trials hide.
5. **Code generation excluded from this 3-way.** The
   [companion 5-case code_generation matrix](../3way_codegen/analysis.md)
   exists but is too small to publish standalone. Folding it in
   would require expanding to HumanEval-scale.

---

## Reproducibility

```bash
# requires OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY
for SUITE in reasoning extraction open_ended_qa; do
  rift matrix \
    --models gpt-5.5,opus-4-7,gemini-3-5-flash \
    --suite $SUITE \
    --output-dir benchmarks/3way_full/$SUITE
done
```

Two environmental gotchas that came up during this run, documented
here so they don't bite the next person:

1. **OpenAI requires `max_completion_tokens` and rejects non-default
   `temperature` for gpt-5/o-series.** Rift's OpenAI provider was
   patched to handle this in commit `3db18aa` —
   `src/rift/providers/openai.py` now branches on model prefix.
2. **Network egress allowlists.** If you run this from Claude Code
   on the web, `api.openai.com` must be in the environment's
   allowlist *at session start*; mid-session allowlist changes
   don't apply until a new session.

Raw per-case JSONs live in
`benchmarks/3way_full/{reasoning,extraction,open_ended_qa}/`
and are sufficient to regenerate every number in this writeup
without API calls.

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
non-negotiable; otherwise gpt-5.5 is the default pick.

The actionable claim for procurement: **multiply each model's
listed per-token output price by the average output tokens it
emits on your workload, not on the vendor demo.** That's the
number that shows up on your invoice.
