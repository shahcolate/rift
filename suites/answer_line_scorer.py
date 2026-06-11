"""Final-answer-line scorer for the hard_reasoning suite.

Parses the LAST line of the output that starts with ``Answer:`` and
compares only that value against ``expected`` — reasoning above the
answer line is encouraged, not penalized. This deliberately decouples
*capability* from *format compliance*: the Fable 5 vs Opus 4.7 standard
suites showed Fable appending explanations to bare-answer prompts, and
exact-match scoring would re-measure that verbosity habit instead of
problem-solving.

Comparison: exact-rational equality via ``fractions.Fraction`` when both
sides parse as numbers (so ``53/5`` == ``10.6`` and ``7`` == ``7.0``),
else case-insensitive string equality after stripping markdown
decoration and trailing punctuation.
"""

import re
from fractions import Fraction

_ANSWER_RE = re.compile(r"^\s*\**`?answer`?\**\s*[:=]\s*(.+?)\s*$", re.IGNORECASE)


def _extract(output: str) -> str | None:
    last = None
    for line in output.splitlines():
        m = _ANSWER_RE.match(line)
        if m:
            last = m.group(1)
    return last


def _clean(value: str) -> str:
    value = value.strip().strip("`*_$ ")
    value = re.sub(r"[.。]\s*$", "", value)
    # "14 stones", "7 hours", "$8.40" -> keep the bare token when the
    # remainder is a unit word; comparison falls back to string equality
    # if this guess is wrong.
    value = value.strip()
    return value


def _as_fraction(value: str) -> Fraction | None:
    value = value.replace(",", "").replace(" ", "")
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None


def score(output: str, expected) -> float:
    raw = _extract(output or "")
    if raw is None:
        return 0.0
    got = _clean(raw)
    want = _clean(str(expected))

    f_got, f_want = _as_fraction(got), _as_fraction(want)
    if f_got is not None and f_want is not None:
        return 1.0 if f_got == f_want else 0.0

    if f_want is not None:
        # expected is numeric but answer has extra words ("14 stones"):
        # try the first numeric-looking token.
        m = re.search(r"-?\d+(?:/\d+|\.\d+)?", got)
        if m:
            f_tok = _as_fraction(m.group(0))
            if f_tok is not None:
                return 1.0 if f_tok == f_want else 0.0
        return 0.0

    return 1.0 if got.lower() == want.lower() else 0.0
