"""Synthetic task corpus for RiftLM.

Every training example is a single line of the form::

    <task> <arg> = <answer>\n

with four tasks a ~400K-parameter character model can actually learn:

- ``cpy abc = abc``     copy a lowercase string
- ``rev abc = cba``     reverse it
- ``srt 3142 = 1234``   sort a digit string
- ``max 3172 = 7``      largest digit

The prompt Rift sends the model is everything up to and including the
``"= "`` separator; the expected completion is the answer (generation
stops at the newline). Answers are deterministic functions of the
input, so ``exact_match`` scoring works and accuracy per task is a
real, interpretable number.

Train/eval split is by content hash of the full line
(:func:`is_eval_line`): the training sampler rejects eval-split lines
and the committed suite (``rift lm suite``) draws only from them, so
held-out means held out without any file coordination between the two.
"""

from __future__ import annotations

import hashlib

import numpy as np

# Index 0 is newline on purpose: it doubles as the pad token, and the
# loss mask (see batch()) stops one position after the first newline so
# padding never contributes gradient.
VOCAB = "\n abcdefghijklmnopqrstuvwxyz0123456789="
STOI = {ch: i for i, ch in enumerate(VOCAB)}
ITOS = {i: ch for i, ch in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)
NEWLINE_ID = 0

TASKS = ("cpy", "rev", "srt", "max")

# Longest possible line is "rev abcdef = fedcba\n" (20 chars); the model's
# context window must cover it minus the shifted-off last char.
MAX_LINE_LEN = 20

_LETTERS = "abcdefghijklmnopqrstuvwxyz"
_DIGITS = "0123456789"

# Fraction (percent) of the example universe reserved for evaluation.
EVAL_PCT = 5


def encode(text: str) -> list[int]:
    """Map text to token ids. Unknown characters become spaces after
    lowercasing, so an out-of-vocab prompt degrades instead of crashing."""
    out = []
    for ch in text.lower():
        out.append(STOI.get(ch, STOI[" "]))
    return out


def decode(ids: list[int]) -> str:
    return "".join(ITOS[int(i)] for i in ids)


def is_eval_line(line: str) -> bool:
    """True when ``line`` belongs to the held-out evaluation split.

    Content-hash based so the split is a property of the example itself:
    the training sampler and the suite generator agree by construction,
    with no shared state or file to keep in sync.
    """
    h = int.from_bytes(hashlib.sha256(line.encode()).digest()[:8], "big")
    return h % 100 < EVAL_PCT


def _answer(task: str, arg: str) -> str:
    if task == "cpy":
        return arg
    if task == "rev":
        return arg[::-1]
    if task == "srt":
        return "".join(sorted(arg))
    if task == "max":
        return max(arg)
    raise ValueError(f"unknown task: {task}")


def make_example(task: str, rng: np.random.Generator) -> tuple[str, str, str]:
    """One random example: returns ``(prompt, answer, full_line)``."""
    n = int(rng.integers(3, 7))  # arg length 3..6
    alphabet = _LETTERS if task in ("cpy", "rev") else _DIGITS
    arg = "".join(alphabet[i] for i in rng.integers(0, len(alphabet), size=n))
    answer = _answer(task, arg)
    prompt = f"{task} {arg} = "
    return prompt, answer, prompt + answer + "\n"


def sample_line(mix: dict[str, float], rng: np.random.Generator) -> str:
    """Sample one *training* line from the task mix, skipping eval-split lines."""
    tasks = [t for t in TASKS if mix.get(t, 0.0) > 0]
    probs = np.array([mix[t] for t in tasks], dtype=np.float64)
    probs /= probs.sum()
    while True:
        task = tasks[int(rng.choice(len(tasks), p=probs))]
        _, _, line = make_example(task, rng)
        if not is_eval_line(line):
            return line


def batch(
    mix: dict[str, float],
    batch_size: int,
    block_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a training batch.

    Returns ``(x, y, mask)`` each of shape ``(batch_size, block_size)``:
    ``x`` is the line right-padded with newline, ``y`` is ``x`` shifted
    left by one, and ``mask`` is 1.0 over real targets (the line body
    plus its terminating newline) and 0.0 over padding.
    """
    x = np.zeros((batch_size, block_size), dtype=np.int64)
    y = np.zeros((batch_size, block_size), dtype=np.int64)
    mask = np.zeros((batch_size, block_size), dtype=np.float64)
    for b in range(batch_size):
        ids = encode(sample_line(mix, rng))  # ends with the newline token
        ids = ids[: block_size + 1]
        x[b, : len(ids) - 1] = ids[:-1]
        y[b, : len(ids) - 1] = ids[1:]
        mask[b, : len(ids) - 1] = 1.0
    return x, y, mask


def gen_eval_cases(
    per_task: int, seed: int = 1234
) -> list[dict[str, object]]:
    """Deterministically draw held-out cases for the committed suite.

    Only eval-split lines are eligible (the exact complement of what the
    training sampler will accept), deduplicated, ``per_task`` cases per
    task, each tagged ``task:<name>`` so ``rift compare --subgroup task:``
    can pinpoint which capability drifted.
    """
    rng = np.random.default_rng(seed)
    cases: list[dict[str, object]] = []
    for task in TASKS:
        seen: set[str] = set()
        while len(seen) < per_task:
            prompt, answer, line = make_example(task, rng)
            if not is_eval_line(line) or line in seen:
                continue
            seen.add(line)
            cases.append(
                {"input": prompt, "expected": answer, "tags": [f"task:{task}"]}
            )
    return cases
