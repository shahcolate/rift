"""run_suite must surface a missing API key, not bury it per-case."""

from __future__ import annotations

import asyncio

import pytest

from rift.config import EvalCase, ModelConfig, SuiteConfig
from rift.providers import MissingAPIKeyError
from rift.runner import run_suite


def _suite() -> SuiteConfig:
    return SuiteConfig(
        name="needs_key",
        scoring="exact_match",
        cases=[EvalCase(input="What is 2+2?", expected="4")],
    )


def test_missing_key_propagates_not_swallowed(tmp_path, monkeypatch):
    # No key set, uncached case -> the provider is constructed lazily and
    # raises MissingAPIKeyError. It must propagate (clean ClickException),
    # not be caught and turned into CaseResult(error=..., score=0.0).
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    suite = _suite()
    cfg = ModelConfig(provider="anthropic", model="claude-opus-4-8")
    with pytest.raises(MissingAPIKeyError, match="ANTHROPIC_API_KEY"):
        asyncio.run(
            run_suite(suite, cfg, cache_dir=str(tmp_path), show_progress=False)
        )
