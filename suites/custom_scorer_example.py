"""Example custom scorer for ``suites/custom_scorer_example.yaml``.

A custom scorer is any ``score(output, expected) -> float`` (0.0-1.0). Selected
from a suite via ``scoring: custom`` +
``custom_scorer: ./custom_scorer_example.py:keyword_overlap``. You may also
export an async ``ascore(output, expected, context=None)`` or a class
implementing the Scorer protocol. See the README "Custom scoring functions".
"""

from __future__ import annotations


def keyword_overlap(output: str, expected) -> float:
    """Fraction of the expected keywords (space-separated) present in output.

    A simple, dependency-free example: score = |expected_words ∩ output_words|
    / |expected_words|, case-insensitive. Returns 0.0 when expected is empty.
    """
    expected_words = {w.lower() for w in str(expected).split()}
    if not expected_words:
        return 0.0
    out_words = {w.lower().strip(".,!?;:") for w in output.split()}
    hits = sum(1 for w in expected_words if w in out_words)
    return hits / len(expected_words)
