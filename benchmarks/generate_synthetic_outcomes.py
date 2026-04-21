"""Generate a deterministic, literature-informed outcomes file.

**This script does not call any LLM.** It emits a seeded, reproducible
approximation of what a context-rot run *might* look like, derived
from a documented prior model. It exists so reviewers can:

1. Reproduce the published report byte-for-byte without API keys.
2. Audit the assumptions behind every number (they are in this file).
3. Replace the synthetic outcomes with real ones by running
   ``python benchmarks/run_context_rot.py --mode live``.

The prior model has three parameters per model:

* ``base_accuracy`` — probability of correctness at zero distractor
  context, averaged over the eight base cases.
* ``degradation_per_log_decade`` — percentage points lost per 10x
  increase in distractor length. 0k is treated as 1k for the log.
* ``position_sensitivity`` — extra penalty when the needle is in the
  middle of the context (lost-in-the-middle effect).

These are *intentionally* conservative and roughly consistent with
public context-length benchmarks as of Q1 2026 (RULER, HELM-long,
Needle-in-a-Haystack variants). They should not be interpreted as
Anthropic's internal numbers, and they are not a claim about any
specific model. The whole point of Rift is that you run it yourself.
"""

from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path

import yaml


MODELS = {
    # model_id: (base_accuracy, pp_per_log_decade, middle_position_penalty)
    "claude-opus-4-7":          (0.92, 2.5, 0.02),
    "claude-opus-4-6":          (0.88, 4.0, 0.04),
    "claude-sonnet-4-6":        (0.82, 6.5, 0.06),
    "gpt-4o":                   (0.78, 8.0, 0.08),
}


# Rough per-case difficulty multiplier (1.0 = average). Informed by
# which cases tend to trip frontier models: bat-and-ball and the
# widget problem are intuition traps with known high-error rates.
CASE_DIFFICULTY = {
    0: 1.00,   # apple discount — arithmetic
    1: 1.10,   # meeting trains — rate+distance
    2: 0.95,   # syllogism
    3: 0.95,   # affirming consequent
    4: 0.85,   # geometric sequence — easy
    5: 1.15,   # seating puzzle — harder CSP
    6: 1.05,   # bat and ball — intuition trap
    7: 1.05,   # widgets — intuition trap
}


EXPECTED = ["8.40", "10:48 AM", "False", "False", "B", "Bob", "5", "5"]
# Plausible wrong answers per case — what a model that doesn't read
# carefully tends to emit. These make the recording feel like a real
# run even under synthesis.
WRONG_FALLBACKS = {
    0: "10.50",          # forgot the discount
    1: "10:36 AM",       # arithmetic slip
    2: "True",           # classic syllogism error
    3: "True",           # classic affirming-consequent error
    4: "A",              # 54*2 off-by-one
    5: "Eve",
    6: "10",             # bat and ball lore
    7: "100",            # widget lore
}


LEVEL_TOKENS = {"0k": 0, "2k": 2_000, "8k": 8_000, "32k": 32_000}


def _seeded(key: str) -> random.Random:
    return random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))


def _p_correct(model: str, origin: int, level: str, position: str) -> float:
    base, per_decade, mid_pen = MODELS[model]
    # Difficulty correction: harder cases lower base probability.
    p = base / CASE_DIFFICULTY[origin]
    # Context-length penalty, log-scaled. Treat 0k as 1k floor so log
    # is defined; the 0k regime still sees no penalty because the
    # decades-from-1k is 0.
    tokens = LEVEL_TOKENS[level]
    decades = math.log10(max(tokens, 1_000) / 1_000)
    p -= (per_decade / 100.0) * decades
    # Position penalty applies only when there's enough context to
    # meaningfully lose the needle.
    if position == "middle" and tokens >= 8_000:
        p -= mid_pen
    return max(0.0, min(1.0, p))


def _position_for(origin: int) -> str:
    return ("prefix", "middle", "suffix")[origin % 3]


def _token_counts(level: str, answer: str) -> tuple[int, int]:
    """Synthesize plausible usage numbers.

    Input tokens = distractor level + ~50 for the question.
    Output tokens = len(answer) / 4 + short scaffold.
    """
    inp = LEVEL_TOKENS[level] + 60
    out = max(3, len(answer) // 4 + 5)
    return inp, out


def generate() -> dict:
    outcomes: dict = {}
    for model in MODELS:
        outcomes[model] = {}
        for origin in range(len(EXPECTED)):
            pos = _position_for(origin)
            for level in LEVEL_TOKENS:
                rng = _seeded(f"{model}|{origin}|{level}")
                p = _p_correct(model, origin, level, pos)
                is_correct = rng.random() < p
                answer = EXPECTED[origin] if is_correct else WRONG_FALLBACKS[origin]
                inp_tok, out_tok = _token_counts(level, answer)
                key = f"origin:{origin}|distractor:{level}"
                outcomes[model][key] = {
                    "output": answer,
                    "input_tokens": inp_tok,
                    "output_tokens": out_tok,
                    "latency_ms": round(200 + LEVEL_TOKENS[level] * 0.02, 1),
                }
    return outcomes


def main():
    out = Path(__file__).parent / "context_rot_outcomes.yaml"
    outcomes = generate()
    preamble = [
        "# Auto-generated by benchmarks/generate_synthetic_outcomes.py.",
        "# These are SYNTHESIZED outcomes from a documented prior model,",
        "# NOT real API responses. Re-run benchmarks/run_context_rot.py",
        "# with --mode live to replace with real data.",
        "",
    ]
    body = yaml.safe_dump(outcomes, sort_keys=True)
    out.write_text("\n".join(preamble) + body)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
