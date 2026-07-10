"""Regression tests for the core-engine audit fixes.

Pins: pricing must not bill named submodels at family rates, derived
suites must carry their grader configuration, a failing scorer must
error one case (not abort the run), and confidence-stripping must only
touch the trailing line.
"""

from __future__ import annotations

import click
import pytest

from rift.config import SuiteConfig
from rift.context_rot import expand_suite
from rift.pricing import PRICING, lookup
from rift.runner import run_suite
from rift.scoring.exact_match import _strip_confidence
from rift.sycophancy import build_pushback_suite


# ---------------------------------------------------------------------------
# pricing.lookup family-prefix fallback
# ---------------------------------------------------------------------------

def test_dated_variants_inherit_family_price():
    assert lookup("claude-opus-4-7-20260315") == PRICING["claude-opus-4-7"]
    assert lookup("gpt-4o-2024-08-06") == PRICING["gpt-4o"]


def test_named_submodels_do_not_inherit_family_price():
    # gpt-4o-mini at gpt-4o rates is ~16x over — an unknown model must
    # price as unknown (0 / None), never confidently wrong.
    assert lookup("gpt-4o-mini") is None
    assert lookup("o1-mini") is None
    assert lookup("o3-mini") is None


def test_exact_keys_still_hit():
    assert lookup("gpt-4o") == PRICING["gpt-4o"]
    assert lookup("o1") == PRICING["o1"]


# ---------------------------------------------------------------------------
# Derived suites carry grader configuration
# ---------------------------------------------------------------------------

def _judge_suite() -> SuiteConfig:
    return SuiteConfig(
        name="graded",
        scoring="llm_judge",
        judge_model="claude-sonnet-4-6",
        prompts={},
        cases=[{"input": "why is the sky blue?", "expected": "rayleigh"}],
    )


def test_pushback_suite_carries_judge_model():
    from rift.runner import CaseResult, RunResult

    original = _judge_suite()
    run = RunResult(
        model="m", suite_name="graded", scoring_method="llm_judge",
        cases=[CaseResult(
            case_index=0, input_text="why is the sky blue?",
            expected="rayleigh", output="scattering", score=1.0,
            latency_ms=1.0, input_tokens=1, output_tokens=1,
        )],
    )
    derived = build_pushback_suite(original, run)
    assert derived.judge_model == original.judge_model
    assert derived.scoring == original.scoring


def test_context_rot_expansion_carries_judge_model():
    original = _judge_suite()
    expanded = expand_suite(original)
    assert expanded.judge_model == original.judge_model


def test_context_rot_expansion_carries_embedding_model():
    suite = SuiteConfig(
        name="sem", scoring="semantic", embedding_model="text-embedding-3-large",
        cases=[{"input": "a", "expected": "b"}],
    )
    assert expand_suite(suite).embedding_model == "text-embedding-3-large"


# ---------------------------------------------------------------------------
# Scorer failures: per-case error, not an aborted run
# ---------------------------------------------------------------------------

class _StubProvider:
    def __init__(self, model="stub"):
        self.model = model

    async def complete(self, prompt, **params):
        from rift.providers import Completion

        return Completion(
            model=self.model, input_text=prompt, output_text="out",
            latency_ms=1.0, input_tokens=1, output_tokens=1,
            raw_response={},
        )

    async def close(self):
        pass


class _ExplodingScorer:
    calls = 0

    def score(self, output, expected):
        _ExplodingScorer.calls += 1
        if _ExplodingScorer.calls == 2:
            raise RuntimeError("judge fell over")
        return 1.0


async def test_scorer_failure_errors_one_case_and_run_continues(tmp_path, monkeypatch):
    _ExplodingScorer.calls = 0
    monkeypatch.setattr("rift.runner._get_provider", lambda cfg: _StubProvider())
    monkeypatch.setattr("rift.runner.get_scorer", lambda name, **kw: _ExplodingScorer())

    from rift.config import ModelConfig

    suite = SuiteConfig(name="s", scoring="exact_match", cases=[
        {"input": "a", "expected": "out"},
        {"input": "b", "expected": "out"},
        {"input": "c", "expected": "out"},
    ])
    result = await run_suite(
        suite, ModelConfig(provider="local", model="stub"),
        cache_dir=str(tmp_path), show_progress=False, concurrency=1,
    )
    errored = [c for c in result.cases if c.error]
    scored = [c for c in result.cases if not c.error]
    assert len(errored) == 1
    assert "scoring" in errored[0].error
    assert len(scored) == 2  # the run survived the mid-flight scorer failure


class _MissingKeyScorer:
    def score(self, output, expected):
        raise click.ClickException("OPENAI_API_KEY missing for judge")


async def test_scorer_click_exception_stays_fatal(tmp_path, monkeypatch):
    monkeypatch.setattr("rift.runner._get_provider", lambda cfg: _StubProvider())
    monkeypatch.setattr("rift.runner.get_scorer", lambda name, **kw: _MissingKeyScorer())

    from rift.config import ModelConfig

    suite = SuiteConfig(name="s", scoring="exact_match",
                        cases=[{"input": "a", "expected": "out"}])
    with pytest.raises(click.ClickException, match="OPENAI_API_KEY"):
        await run_suite(
            suite, ModelConfig(provider="local", model="stub"),
            cache_dir=str(tmp_path), show_progress=False,
        )


# ---------------------------------------------------------------------------
# Confidence strip: trailing line only
# ---------------------------------------------------------------------------

def test_strip_confidence_trailing_line():
    assert _strip_confidence("Paris\nConfidence: 0.9") == "Paris"
    assert _strip_confidence("Paris\nI am 85% sure") == "Paris"


def test_strip_confidence_leaves_mid_answer_lines_alone():
    # A confidence-shaped line that is NOT trailing is part of the answer.
    text = "I am 90% sure\nParis"
    assert _strip_confidence(text) == text


def test_strip_confidence_mid_body_probability_untouched():
    text = "There's a 50% chance of rain.\nAnswer: yes"
    assert _strip_confidence(text) == text


def test_strip_confidence_lone_confidence_line_kept():
    # If the whole output is the confidence tag, stripping would leave
    # nothing — keep the original.
    assert _strip_confidence("Confidence: 0.5") == "Confidence: 0.5"
