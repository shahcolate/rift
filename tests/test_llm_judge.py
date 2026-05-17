"""Tests for the LLM-as-judge scorer.

Uses a stub :class:`BaseProvider` so no real network call is
exercised. Tests cover: response parsing (well-formed, fenced,
malformed, out-of-range), cache hit/miss behavior, rubric vs.
reference dispatch, and the runner-side detection of async scorers.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable

import pytest

from rift.providers import BaseProvider, Completion
from rift.scoring import get_scorer
from rift.scoring.llm_judge import (
    LLMJudgeScorer,
    _build_judge_prompt,
    _format_target_block,
    _parse_judge_response,
)


# ---------------------------------------------------------------------------
# Stub provider: returns a scripted response per call.
# ---------------------------------------------------------------------------


@dataclass
class _Recorded:
    prompt: str
    kwargs: dict


class StubProvider(BaseProvider):
    """A BaseProvider whose `complete` returns whatever you script.

    ``responder`` receives the prompt and returns the
    ``output_text`` string the judge will see. Every call is logged
    on ``calls`` for assertion.
    """

    def __init__(self, responder: Callable[[str], str], model: str = "stub-1") -> None:
        self.responder = responder
        self.model = model
        self.calls: list[_Recorded] = []
        self.closed = False

    async def complete(self, prompt: str, **kwargs) -> Completion:
        self.calls.append(_Recorded(prompt=prompt, kwargs=kwargs))
        out = self.responder(prompt)
        return Completion(
            model=self.model,
            input_text=prompt,
            output_text=out,
            latency_ms=1.0,
            input_tokens=10,
            output_tokens=5,
            raw_response={},
        )

    async def close(self) -> None:
        self.closed = True


def _stub_factory(responder: Callable[[str], str]) -> Callable[[str], BaseProvider]:
    """Return a provider_factory that yields a fresh StubProvider per model."""
    def factory(model: str) -> BaseProvider:
        return StubProvider(responder, model=model)
    return factory


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_clean_json(self):
        s, r = _parse_judge_response('{"score": 0.85, "reasoning": "good"}')
        assert s == 0.85
        assert r == "good"

    def test_markdown_fenced(self):
        s, r = _parse_judge_response(
            '```json\n{"score": 1.0, "reasoning": "perfect"}\n```'
        )
        assert s == 1.0
        assert r == "perfect"

    def test_with_surrounding_prose(self):
        s, _ = _parse_judge_response(
            'Here is my grade: {"score": 0.4, "reasoning": "partial"} hope that helps.'
        )
        assert s == 0.4

    def test_out_of_range_clipped(self):
        # Some judges return 5 on a 1-5 scale despite the prompt.
        s, _ = _parse_judge_response('{"score": 5, "reasoning": "ok"}')
        assert s == 1.0
        s, _ = _parse_judge_response('{"score": -0.5, "reasoning": "neg"}')
        assert s == 0.0

    def test_non_numeric_score(self):
        s, r = _parse_judge_response('{"score": "high", "reasoning": "x"}')
        assert s == 0.0
        assert "non-numeric" in r

    def test_unparseable(self):
        s, r = _parse_judge_response("Hmm, I would rate this an 8 out of 10.")
        assert s == 0.0
        assert "unparseable" in r

    def test_missing_reasoning_ok(self):
        s, r = _parse_judge_response('{"score": 0.5}')
        assert s == 0.5
        assert r == ""


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


class TestPromptAssembly:
    def test_string_reference(self):
        block = _format_target_block("Paris")
        assert "Reference answer" in block
        assert "Paris" in block

    def test_rubric_dict(self):
        block = _format_target_block({"rubric": "Mentions both parties to the contract."})
        assert "Grading rubric" in block
        assert "both parties" in block

    def test_dict_without_rubric(self):
        block = _format_target_block({"name": "Alice", "age": 30})
        assert "Reference answer (JSON)" in block
        # JSON should be rendered.
        assert "Alice" in block

    def test_full_prompt_includes_all_three(self):
        prompt = _build_judge_prompt(
            question="What's the capital of France?",
            output="Paris is the capital.",
            expected="Paris",
        )
        assert "capital of France" in prompt
        assert "Paris is the capital" in prompt
        assert "Reference answer" in prompt


# ---------------------------------------------------------------------------
# Scoring + caching
# ---------------------------------------------------------------------------


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) \
        if asyncio.get_event_loop_policy().get_event_loop().is_running() \
        else asyncio.run(coro)


class TestAscore:
    def test_basic_score(self, tmp_path):
        provider = StubProvider(
            lambda _: '{"score": 0.9, "reasoning": "essentially correct"}'
        )
        scorer = LLMJudgeScorer(
            judge_model="judge-1",
            provider_factory=lambda _m: provider,
            cache_dir=tmp_path,
        )
        sc = asyncio.run(scorer.ascore("Paris is the capital.", "Paris",
                                       context="What is the capital?"))
        assert sc == 0.9
        assert len(provider.calls) == 1
        # The judge prompt should have included the context and the output.
        assert "What is the capital?" in provider.calls[0].prompt
        assert "Paris is the capital" in provider.calls[0].prompt

    def test_cache_hit_skips_call(self, tmp_path):
        responses = iter([
            '{"score": 0.7, "reasoning": "first"}',
            '{"score": 0.0, "reasoning": "should not be called"}',
        ])
        provider = StubProvider(lambda _: next(responses))
        scorer = LLMJudgeScorer(
            judge_model="judge-1",
            provider_factory=lambda _m: provider,
            cache_dir=tmp_path,
        )
        s1 = asyncio.run(scorer.ascore("answer", "expected", context="q"))
        s2 = asyncio.run(scorer.ascore("answer", "expected", context="q"))
        assert s1 == 0.7
        assert s2 == 0.7
        # Only one network call.
        assert len(provider.calls) == 1

    def test_cache_persists_across_instances(self, tmp_path):
        responses = iter([
            '{"score": 0.42, "reasoning": "first"}',
            '{"score": 0.0, "reasoning": "should be cached"}',
        ])
        provider1 = StubProvider(lambda _: next(responses))
        scorer1 = LLMJudgeScorer(
            judge_model="judge-1",
            provider_factory=lambda _m: provider1,
            cache_dir=tmp_path,
        )
        asyncio.run(scorer1.ascore("a", "b", context="q"))

        provider2 = StubProvider(lambda _: next(responses))
        scorer2 = LLMJudgeScorer(
            judge_model="judge-1",
            provider_factory=lambda _m: provider2,
            cache_dir=tmp_path,
        )
        s2 = asyncio.run(scorer2.ascore("a", "b", context="q"))
        assert s2 == 0.42
        # provider2 should not have been called — cache hit.
        assert len(provider2.calls) == 0

    def test_different_judge_does_not_collide(self, tmp_path):
        responses_a = iter(['{"score": 0.1, "reasoning": "a"}'])
        responses_b = iter(['{"score": 0.9, "reasoning": "b"}'])
        scorer_a = LLMJudgeScorer(
            judge_model="judge-A",
            provider_factory=lambda _m: StubProvider(lambda _: next(responses_a)),
            cache_dir=tmp_path,
        )
        scorer_b = LLMJudgeScorer(
            judge_model="judge-B",
            provider_factory=lambda _m: StubProvider(lambda _: next(responses_b)),
            cache_dir=tmp_path,
        )
        sa = asyncio.run(scorer_a.ascore("x", "y", context="q"))
        sb = asyncio.run(scorer_b.ascore("x", "y", context="q"))
        assert sa == 0.1
        assert sb == 0.9

    def test_rubric_dispatch(self, tmp_path):
        seen = []

        def responder(prompt):
            seen.append(prompt)
            return '{"score": 1.0, "reasoning": "rubric satisfied"}'

        provider = StubProvider(responder)
        scorer = LLMJudgeScorer(
            judge_model="j",
            provider_factory=lambda _m: provider,
            cache_dir=tmp_path,
        )
        asyncio.run(scorer.ascore(
            "Some response.",
            {"rubric": "Mentions the year 1969."},
            context="When did Apollo 11 land?",
        ))
        assert "Grading rubric" in seen[0]
        assert "1969" in seen[0]

    def test_malformed_judge_response_returns_zero(self, tmp_path):
        provider = StubProvider(lambda _: "i think this is pretty good actually")
        scorer = LLMJudgeScorer(
            judge_model="j",
            provider_factory=lambda _m: provider,
            cache_dir=tmp_path,
        )
        sc = asyncio.run(scorer.ascore("answer", "expected", context="q"))
        assert sc == 0.0

    def test_sync_score_in_event_loop_raises(self, tmp_path):
        provider = StubProvider(lambda _: '{"score": 1.0, "reasoning": "x"}')
        scorer = LLMJudgeScorer(
            judge_model="j",
            provider_factory=lambda _m: provider,
            cache_dir=tmp_path,
        )

        async def go():
            with pytest.raises(RuntimeError):
                scorer.score("a", "b")

        asyncio.run(go())

    def test_close_releases_provider(self, tmp_path):
        provider = StubProvider(lambda _: '{"score": 0.5, "reasoning": ""}')
        scorer = LLMJudgeScorer(
            judge_model="j",
            provider_factory=lambda _m: provider,
            cache_dir=tmp_path,
        )
        asyncio.run(scorer.ascore("a", "b", context="q"))
        asyncio.run(scorer.close())
        assert provider.closed is True


# ---------------------------------------------------------------------------
# get_scorer dispatch
# ---------------------------------------------------------------------------


class TestGetScorer:
    def test_returns_judge_for_llm_judge(self, tmp_path):
        provider = StubProvider(lambda _: '{"score": 1.0, "reasoning": ""}')
        scorer = get_scorer(
            "llm_judge",
            judge_model="j",
            provider_factory=lambda _m: provider,
            cache_dir=tmp_path,
        )
        assert isinstance(scorer, LLMJudgeScorer)
        assert scorer.judge_model == "j"

    def test_ignores_kwargs_for_exact_match(self):
        # Exact match shouldn't crash when called with judge kwargs.
        scorer = get_scorer("exact_match")
        assert scorer.score("Paris", "Paris") == 1.0

    def test_unknown_scorer_raises(self):
        with pytest.raises(ValueError):
            get_scorer("nonsense")

    def test_judge_model_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RIFT_JUDGE_MODEL", "env-judge-99")
        scorer = get_scorer(
            "llm_judge",
            provider_factory=lambda _m: StubProvider(lambda _: ""),
            cache_dir=tmp_path,
        )
        assert scorer.judge_model == "env-judge-99"


# ---------------------------------------------------------------------------
# Runner integration
# ---------------------------------------------------------------------------


class TestRunnerIntegration:
    """End-to-end: a suite scored by the stub LLM judge through the runner."""

    def test_run_suite_dispatches_to_async_scorer(self, tmp_path, monkeypatch):
        from rift.config import EvalCase, ModelConfig, SuiteConfig
        from rift import runner as runner_mod
        from rift import scoring as scoring_mod

        # 1) Stub model provider — returns the case input as the answer.
        model_provider = StubProvider(lambda prompt: f"answer for: {prompt[:30]}")

        def fake_get_provider(_cfg):
            return model_provider
        monkeypatch.setattr(runner_mod, "_get_provider", fake_get_provider)

        # 2) Stub judge provider — score derived from the case index
        # found in the prompt, so we don't depend on completion order.
        scores_by_q = {"Q1": 0.9, "Q2": 0.5, "Q3": 0.1}

        def judge_responder(prompt):
            for q, sc in scores_by_q.items():
                if q in prompt:
                    return f'{{"score": {sc}, "reasoning": "ok"}}'
            return '{"score": 0.0, "reasoning": "no match"}'

        judge_provider = StubProvider(judge_responder)

        # 3) Patch get_scorer to inject the stub judge factory.
        real_get_scorer = scoring_mod.get_scorer

        def patched_get_scorer(name, **kwargs):
            if name == "llm_judge":
                kwargs.setdefault("provider_factory", lambda _m: judge_provider)
                kwargs.setdefault("cache_dir", str(tmp_path))
            return real_get_scorer(name, **kwargs)

        monkeypatch.setattr(runner_mod, "get_scorer", patched_get_scorer)

        suite = SuiteConfig(
            name="t",
            scoring="llm_judge",
            judge_model="judge-x",
            cases=[
                EvalCase(input="Q1", expected="A1"),
                EvalCase(input="Q2", expected="A2"),
                EvalCase(input="Q3", expected="A3"),
            ],
        )
        model_cfg = ModelConfig(provider="anthropic", model="test-model")

        result = asyncio.run(runner_mod.run_suite(
            suite, model_cfg, concurrency=1, cache_dir=str(tmp_path)
        ))

        assert [c.score for c in result.cases] == [0.9, 0.5, 0.1]
        assert result.metadata["judge_model"] == "judge-x"
        # Judge was called once per case.
        assert len(judge_provider.calls) == 3
