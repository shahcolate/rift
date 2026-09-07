# Fable 5.1 launch study (not yet run)

`run.sh` is the reproducible driver for the two launch-week questions:

| Pairing | Baseline | Challenger | Question |
|---|---|---|---|
| `upgrade/` | `fable-5` | `fable-5-1` | Same tier, same price: did the point release regress anything? |
| `tier/` | `opus-5` | `fable-5-1` | Is the Mythos-class tier worth 2× — with both sides thinking by default this time |

Nothing under this directory is a result until the JSONs and `.md`
reports exist beside this file. When they do, write `analysis.md` in the
form of [`../fable5_vs_opus47/analysis.md`](../fable5_vs_opus47/analysis.md):
provenance block first (date, exact model ids echoed by the API, judge +
its κ, total spend, configuration disclosure), then the scorecard, then
the cost-in-context section.

Estimated cost before running (keyless):

```bash
rift estimate --model fable-5 --model fable-5-1 --model opus-5 \
  --suite reasoning --suite extraction --suite summarization \
  --suite code_generation --suite open_ended_qa --suite hard_reasoning \
  --calibrate-from benchmarks/fable5_vs_opus47/hard_reasoning.json \
  --calibrate-from benchmarks/fable5_vs_opus47/reasoning.json
```

≈ $20 at list price for the full script; `SKIP_CONTEXT_ROT=1 ./run.sh` ≈ $9.
The `fable-5-1` completions are cached after the first pairing, so the
second pairing pays only for the `opus-5` side.

What would make this publishable rather than just interesting:

- a **significant** result on `hard_reasoning` or `context_rot` in either
  direction, with the `$/correct` CI (the Fable 5 study's finding was
  "every quality probe ties; only cost is significant");
- a non-zero **API-level refusal** count on either side (`stop_reason=refusal`,
  disclosed by the drift report since this release) — over-refusal drift
  between Fable 5 and 5.1 is exactly the kind of behavior change nobody
  else measures;
- a **fingerprint** on the `fable-5-1` side that differs from the one the
  Observatory panel records the following Monday — a launch-week rollout.
