"""Scoring functions for eval comparison."""

from __future__ import annotations

from typing import Any, Protocol


class Scorer(Protocol):
    """Protocol for scoring functions."""

    def score(self, output: str, expected: Any) -> float:
        """Score a model output against expected. Returns 0.0-1.0."""
        ...


def get_scorer(name: str) -> Scorer:
    """Get a scorer by name."""
    from .exact_match import ExactMatchScorer
    from .fuzzy_match import FuzzyMatchScorer

    scorers: dict[str, Scorer] = {
        "exact_match": ExactMatchScorer(),
        "fuzzy_match": FuzzyMatchScorer(),
    }
    if name not in scorers:
        raise ValueError(f"Unknown scorer: {name}. Available: {list(scorers.keys())}")
    return scorers[name]
