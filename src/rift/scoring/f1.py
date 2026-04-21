"""F1 semantic scoring — token-overlap harmonic mean.

A lightweight alternative to embedding-based similarity. Treats a
BLEU-style unigram precision and a ROUGE-style unigram recall as the
two complementary signals a summary-style metric usually trades off,
then combines them with the standard F1 harmonic mean:

    BLEU  = |output ∩ expected| / |output|        (precision-like)
    ROUGE = |output ∩ expected| / |expected|      (recall-like)
    F1    = 2 * BLEU * ROUGE / (BLEU + ROUGE)

Counts are clipped per token (multiset intersection), so repeating a
correct word doesn't inflate the score. Tokenization is lowercase
Unicode word-splitting — no stemming, no stopword removal, no
external dependencies.

Use when you want more forgiveness than ``exact_match`` (paraphrases
and word-order changes survive) but don't want to pay for an
embedding model. For open-ended generation where meaning can drift
even when tokens overlap, prefer an embedding or ``llm_judge``
scorer once those land.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class F1Scorer:
    """Scores token-level F1 between output and expected strings."""

    def score(self, output: str, expected: Any) -> float:
        expected_str = str(expected)
        output_tokens = _tokenize(output)
        expected_tokens = _tokenize(expected_str)

        if not output_tokens or not expected_tokens:
            return 0.0

        overlap = sum(
            (Counter(output_tokens) & Counter(expected_tokens)).values()
        )
        if overlap == 0:
            return 0.0

        precision = overlap / len(output_tokens)
        recall = overlap / len(expected_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        return round(f1, 4)
