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
`temperature: 0` is requested for all models; the OpenAI provider drops
it for gpt-5 family since those endpoints reject any non-default
temperature. Single trial per case, no replays.

## Results

| Model | n correct | Mean | Spend | $/correct | In tok | Out tok | Avg latency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-5.5          | 5/5 | 1.000 | $0.0100 | $0.0020 | 323 | 421  | 2.7 s |
| gemini-3-5-flash | 5/5 | 1.000 | $0.0283 | $0.0057 | 307 | 3092 | 3.9 s |
| opus-4-7         | 5/5 | 1.000 | $0.0427 | $0.0085 | 499 | 469  | 2.3 s |

### Pairwise drift

```
                       challenger
                       gemini-flash    gpt-5.5         opus-4-7
baseline gemini-flash      —           Δ +0.000        Δ +0.000
                                       p=1.000         p=1.000
                                       Δ$/c -$0.0037   Δ$/c +$0.0029

         gpt-5.5         Δ +0.000        —             Δ +0.000
                         p=1.000                       p=1.000
                         Δ$/c +$0.0037                 Δ$/c +$0.0065

         opus-4-7        Δ +0.000      Δ +0.000          —
                         p=1.000       p=1.000
                         Δ$/c -$0.0029 Δ$/c -$0.0065
```

All three models tie on correctness (5/5 each, every pairwise McNemar
p=1.000). The signal is entirely on the cost axis: `Δ$/correct`
collapses to a simple ranking.

## Cost & token observations

**gpt-5.5 is 2.9× cheaper than Gemini Flash and 4.3× cheaper than Opus
4.7 on this suite**, despite Opus being slightly faster wall-clock
(2.3 s vs 2.7 s avg). The advantage comes from gpt-5.5's terse output —
421 total output tokens across 5 cases, vs Gemini's 3092 (mostly
internal "thinking" trace) and Opus's 469.

Per-case cost ranking is consistent across all 5 cases — gpt-5.5
wins every cell:

| Case | gpt-5.5 | gemini-flash | opus-4-7 |
| --- | --- | --- | --- |
| fizzbuzz (basic,loops)         | $0.0024 | $0.0053 | $0.0113 |
| is_palindrome (basic,strings)  | $0.0016 | $0.0048 | $0.0056 |
| flatten (recursion)            | $0.0018 | $0.0083 | $0.0066 |
| two_sum (algorithms,hash_map)  | $0.0017 | $0.0047 | $0.0072 |
| merge_sorted (sorting)         | $0.0025 | $0.0053 | $0.0120 |

Latency notes:
- Opus 4.7 is fastest on average (2.3 s) — consistent with its short
  outputs and no extended-thinking trace on this suite.
- Gemini Flash is slowest (3.9 s avg), dragged by its long internal
  thinking output (avg 618 output tokens per case vs gpt-5.5's 84 and
  Opus's 94).
- gpt-5.5 sits in the middle on latency despite producing the
  cheapest outputs.

The 499 vs 323 vs 307 input-token spread is tokenizer-driven, not
prompt-content — every model received the same suite text.

## Why this 3-way ran in two passes

The first matrix run had two failure modes worth recording in case they
trip up future runs:

1. **Network allowlist (environment-level).** The Claude Code on the
   web container had `api.openai.com` blocked at the egress proxy, so
   gpt-5.5 returned `403 Forbidden` with `Host not in allowlist`.
   Allowlist edits only take effect on session start, so the fix
   required either a new web session or running the matrix locally.
   We chose local.
2. **OpenAI provider request shape (rift-level).** Once OpenAI was
   reachable, gpt-5.5 returned `400 Bad Request` because the provider
   sent `max_tokens` (renamed to `max_completion_tokens` for gpt-5/
   o-series) and forwarded the suite's `temperature: 0` (gpt-5/o-series
   only accept the default temperature). Fixed in commit `3db18aa`:
   `src/rift/providers/openai.py` now branches on model prefix
   (`gpt-5`, `o1`, `o3`, `o4`) and rewrites both fields.

## How to fully reproduce

1. Export `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, and `OPENAI_API_KEY`.
2. From the repo root: `pip install -e ".[dev]"`.
3. Run the command at the top of this file.
4. To replay without keys, per-case completions are cached under
   `.rift/cache/` keyed on `(model, model_params, input_hash)` — but
   that cache is not committed. For a publishable, key-free
   reproduction, port this run to the `benchmarks/run_context_rot.py`
   pattern with `--mode record` and commit the outcomes file.

## Files in this directory

- `gpt-5.5.json` — 5 cases, all `score=1.0`, total spend $0.0100.
- `opus-4-7.json` — 5 cases, all `score=1.0`, total spend $0.0427.
- `gemini-3-5-flash.json` — 5 cases, all `score=1.0`, total spend
  $0.0283.
- `analysis.md` — this file.

## Bottom line

On a 5-case toy code-generation suite, all three frontier models are
indistinguishable on correctness (5/5 each, McNemar p=1.000 across
every pair). The actionable signal is price: **gpt-5.5 is roughly 3×
cheaper than Gemini 3.5 Flash and 4× cheaper than Opus 4.7 per correct
output**, with Opus the latency winner.

These results don't generalize — the suite is small and the tasks are
easy enough that no model would be expected to fail. A serious 3-way
head-to-head would expand to HumanEval-scale (≥50 cases) with a
harder difficulty distribution so the correctness axis actually
resolves before the McNemar floor (`2 · 0.5^k` for k discordant
pairs).
