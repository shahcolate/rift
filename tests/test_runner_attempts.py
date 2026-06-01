"""A failed case must record the *real* attempt count, not always MAX_RETRIES.

A non-transient error (e.g. a 4xx) fails on the first attempt; reporting it as
MAX_RETRIES corrupts the audit trail and overstates retry effort.
"""

from __future__ import annotations

import asyncio

from rift.config import EvalCase, ModelConfig, SuiteConfig
from rift.providers import BaseProvider
from rift.runner import MAX_RETRIES, run_suite


class _AlwaysFailProvider(BaseProvider):
    """Raises a non-transient error on every completion call."""

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, prompt: str, **params):  # type: ignore[override]
        self.calls += 1
        raise ValueError("bad request")  # non-transient => no retries

    async def close(self) -> None:
        pass


def _suite() -> SuiteConfig:
    return SuiteConfig(
        name="fails",
        scoring="exact_match",
        cases=[EvalCase(input="What is 2+2?", expected="4")],
    )


def test_non_transient_error_records_one_attempt(tmp_path, monkeypatch):
    provider = _AlwaysFailProvider()
    monkeypatch.setattr("rift.runner._get_provider", lambda cfg: provider)

    cfg = ModelConfig(provider="anthropic", model="claude-opus-4-8")
    result = asyncio.run(
        run_suite(_suite(), cfg, cache_dir=str(tmp_path), show_progress=False)
    )

    case = result.cases[0]
    assert case.error is not None
    assert "bad request" in case.error
    # The crux: a non-transient failure tried exactly once.
    assert case.attempts == 1
    assert case.attempts != MAX_RETRIES
    assert provider.calls == 1
