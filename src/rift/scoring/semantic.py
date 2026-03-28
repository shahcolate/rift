"""Semantic similarity scoring."""

from difflib import SequenceMatcher
from typing import Any


class SemanticScorer:
    """Scores output by semantic similarity to expected.

    Uses sequence matching as a baseline. Can be extended with
    embedding-based similarity when an embedding provider is configured.
    """

    def score(self, output: str, expected: Any) -> float:
        expected_str = str(expected).strip().lower()
        output_str = output.strip().lower()

        if not expected_str or not output_str:
            return 0.0

        # Sequence-based similarity as baseline
        ratio = SequenceMatcher(None, output_str, expected_str).ratio()
        return round(ratio, 4)
