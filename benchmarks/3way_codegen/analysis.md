# 3-Way Code-Generation Matrix: gpt-5.5 vs opus-4-7 vs gemini-3-5-flash

## Command

```bash
rift matrix --models gpt-5.5,opus-4-7,gemini-3-5-flash \
            --suite code_generation \
            --output-dir benchmarks/3way_codegen
```

Suite: `suites/code_generation.yaml` (5 cases — fizzbuzz, is_palindrome,
flatten, two_sum, merge_sorted). Scoring: `exec_tests` (run model output
against expected unit tests, score 1.0 iff every assertion passes).
`temperature: 0`. Single trial per case, no replays.

## Run environment caveat

`api.openai.com` is not in this remote execution environment's network
allowlist. All 5 `gpt-5.5` calls returned `403 Forbidden` from the egress
proxy, so the gpt-5.5 row of the matrix reflects no API contact — not a
model regression. The matrix is therefore really a 2-way comparison
between Opus 4.7 and Gemini 3.5 Flash, plus a null row for gpt-5.5.

## Results

| Model | n correct | Mean | Spend | $/correct | In tok | Out tok | Avg latency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| opus-4-7         | 5/5 | 1.0000 | $0.0430 | $0.0086 | 499 | 473  | 2.2 s |
| gemini-3-5-flash | 5/5 | 1.0000 | $0.0283 | $0.0057 | 307 | 3092 | 3.5 s |
| gpt-5.5          | 0/5 | 0.0000 | $0.0000 | ∞       | —   | —    | — (403) |

### Pairwise drift

```
                  challenger
                  gemini    gpt-5.5*    opus-4-7
baseline gemini      —      Δ -1.000     Δ +0.000   (p=1.000)
         gpt-5.5*  Δ +1.000    —         Δ +1.000   (artefact, see caveat)
         opus-4-7  Δ +0.000  Δ -1.000      —        (p=1.000)
```

\* gpt-5.5 row/column is an artefact of the 403s — ignore for model
comparison.

The opus-4-7 ↔ gemini-3-5-flash cell is `Δ +0.000, p=1.000`: on these 5
tasks the two models are indistinguishable on correctness. The p-values
of 0.062 against gpt-5.5 are McNemar's exact on 5 fully discordant pairs
(2·0.5⁵ = 0.0625) — they reflect the floor of what 5 cases can resolve,
not real evidence about gpt-5.5.

## Cost & token observations

- Gemini is ~34% cheaper per correct ($0.0057 vs $0.0086) on this suite
  despite producing 6.5× more output tokens (3092 vs 473). The cost
  advantage is Flash's much lower per-token price; the bulk of those
  extra output tokens is likely internal thinking content (the actual
  emitted function bodies are comparable in size to Opus's, see the
  `output` fields in `gemini-3-5-flash.json`).
- Opus is faster on average (2.2 s vs 3.5 s per case) — consistent
  with Gemini Flash spending its time on the longer output trace.
- The input-token gap (499 Opus vs 307 Gemini) is tokenizer-driven, not
  prompt content — both providers received the identical suite text.

## How to fully reproduce

1. Export `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, and `OPENAI_API_KEY`.
2. Run the command at the top of this file.
3. To replay without keys, the per-case completions are cached under
   `.rift/cache/` keyed on `(model, model_params, input_hash)` — but
   that cache is not committed. To produce a publishable, key-free
   reproduction, port this run to the `benchmarks/run_context_rot.py`
   pattern with `--mode record` and commit the outcomes file.

## Files in this directory

- `gpt-5.5.json` — 5 cases, all with `HTTPStatusError: 403 Forbidden`
  (host not in network allowlist). No usable output.
- `opus-4-7.json` — 5 cases, all `score=1.0`, total spend $0.0430.
- `gemini-3-5-flash.json` — 5 cases, all `score=1.0`, total spend
  $0.0283.
- `analysis.md` — this file.

## Bottom line

With 5 cases this suite has ~0 statistical resolving power — both Opus
and Gemini Flash ace it. A meaningful 3-way head-to-head needs (a) a
network path to OpenAI from the runner, and (b) a substantially larger
or harder code-generation suite (HumanEval-style, ≥50 cases) before any
pairwise cell can clear the McNemar floor.
