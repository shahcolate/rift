"""Scoring functions for eval comparison."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Scorer(Protocol):
    """Protocol for scoring functions."""

    def score(self, output: str, expected: Any) -> float:
        """Score a model output against expected. Returns 0.0-1.0."""
        ...


@runtime_checkable
class AsyncScorer(Protocol):
    """Protocol for scorers that need network access.

    The runner checks for ``ascore`` and awaits it when present;
    plain :class:`Scorer` implementations are called synchronously.
    Async scorers may receive the original prompt as ``context`` so
    they can grade in context (an LLM judge needs this).
    """

    async def ascore(self, output: str, expected: Any,
                     context: str | None = None) -> float: ...


def get_scorer(name: str, **kwargs) -> Scorer:
    """Get a scorer by name.

    ``kwargs`` are forwarded to the scorer constructor. ``llm_judge``
    accepts ``judge_model``, ``cache_dir``, ``provider_factory``,
    ``judge_params``; the other built-ins ignore kwargs.
    """
    from .exact_match import ExactMatchScorer
    from .fuzzy_match import FuzzyMatchScorer
    from .llm_judge import LLMJudgeScorer

    if name == "exact_match":
        return ExactMatchScorer()
    if name == "fuzzy_match":
        return FuzzyMatchScorer()
    if name == "llm_judge":
        return LLMJudgeScorer(**kwargs)
    raise ValueError(
        f"Unknown scorer: {name}. Available: "
        f"['exact_match', 'fuzzy_match', 'llm_judge']"
    )
