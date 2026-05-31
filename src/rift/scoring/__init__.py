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
    ``judge_params``; ``semantic`` accepts ``embedding_model``,
    ``cache_dir``, ``embedder_factory``, ``threshold``; ``custom``
    accepts ``custom_scorer`` (a ``target:callable`` spec) and an
    optional ``base_dir`` for resolving a relative file path; the other
    built-ins ignore kwargs.
    """
    from .exact_match import ExactMatchScorer
    from .exec_tests import ExecTestsScorer
    from .fuzzy_match import FuzzyMatchScorer
    from .llm_judge import LLMJudgeScorer
    from .semantic import SemanticScorer

    if name == "exact_match":
        return ExactMatchScorer()
    if name == "fuzzy_match":
        return FuzzyMatchScorer()
    if name == "exec_tests":
        return ExecTestsScorer(**kwargs)
    if name == "llm_judge":
        return LLMJudgeScorer(**kwargs)
    if name == "semantic":
        return SemanticScorer(**kwargs)
    if name == "custom":
        from .custom import load_custom_scorer

        spec = kwargs.get("custom_scorer")
        if not spec:
            raise ValueError(
                "custom scoring requires a 'custom_scorer' spec "
                "(e.g. 'mypkg.scorers:score' or './scorer.py:score')"
            )
        return load_custom_scorer(spec, base_dir=kwargs.get("base_dir"))
    raise ValueError(
        f"Unknown scorer: {name}. Available: "
        f"['exact_match', 'fuzzy_match', 'exec_tests', 'llm_judge', "
        f"'semantic', 'custom']"
    )
