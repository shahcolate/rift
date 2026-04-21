"""Fuzzy string-similarity scoring.

This is a lexical similarity scorer based on ``difflib.SequenceMatcher``. It
compares character sequences, not meaning. Two strings that express the same
idea with different wording will score low; two unrelated strings that share
surface tokens will score higher than they should.

If you need meaning-level comparison, swap this for an embedding-based scorer.
"""

from difflib import SequenceMatcher
from typing import Any


class FuzzyMatchScorer:
    """Scores output by lexical (character-sequence) similarity to expected."""

    def score(self, output: str, expected: Any) -> float:
        expected_str = str(expected).strip().lower()
        output_str = output.strip().lower()

        if not expected_str or not output_str:
            return 0.0

        ratio = SequenceMatcher(None, output_str, expected_str).ratio()
        return round(ratio, 4)
