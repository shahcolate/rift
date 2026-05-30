"""Tests for the reasoning-faithfulness probe (rift.faithfulness)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from rift.config import EvalCase, SuiteConfig
from rift.faithfulness import (
    CUES,
    FaithfulnessResult,
    _is_swayed,
    _parse_answer,
    build_faithfulness_suite,
    build_wrong_answer_suite,
    compute_faithfulness,
    parse_hint_targets,
)


# --- minimal stand-ins for runner dataclasses (no network) -------------------

@dataclass
class FakeCase:
    case_index: int
    input: str
    expected: object
    output: str
    score: float = 0.0
    tags: list[str] = field(default_factory=list)


@dataclass
class FakeRun:
    model: str
    cases: list[FakeCase]


class _ExactScorer:
    def score(self, output: str, expected) -> float:
        _, answer = _parse_answer(output)
        return 1.0 if answer.strip().lower() == str(expected).strip().lower() else 0.0


def _base_suite() -> SuiteConfig:
    return SuiteConfig(
        name="t",
        scoring="exact_match",
        cases=[
            EvalCase(input="2+2?", expected="4", tags=["topic:math"]),
            EvalCase(input="Capital of France?", expected="Paris"),
        ],
    )


class TestParseAnswer:
    def test_extracts_trailing_answer_and_reasoning(self):
        r, a = _parse_answer("Let me think.\nStep 1...\nAnswer: 42")
        assert a == "42"
        assert "Step 1" in r and "Answer:" not in r

    def test_last_answer_line_wins(self):
        r, a = _parse_answer("Answer: draft\nmore\nAnswer: final")
        assert a == "final"

    def test_no_answer_line_returns_whole_as_answer(self):
        r, a = _parse_answer("just 42")
        assert a == "just 42" and r == ""

    def test_empty(self):
        assert _parse_answer("") == ("", "")


class TestIsSwayed:
    def test_exact_normalized(self):
        assert _is_swayed("Paris.", "paris")

    def test_short_token_contained(self):
        assert _is_swayed("The answer is B.", "B")

    def test_not_swayed(self):
        assert not _is_swayed("4", "5")

    def test_empty_target(self):
        assert not _is_swayed("anything", "")

    def test_short_target_not_substring_false_positive(self):
        # Regression: target "2" must NOT match "12"/"2024" (substring),
        # but must match a whole-token "2".
        assert not _is_swayed("12", "2")
        assert not _is_swayed("2024", "2")
        assert _is_swayed("the answer is 2", "2")
        assert _is_swayed("Answer: 2", "2")


class TestBuildFaithfulnessSuite:
    def test_control_plus_one_variant_per_cue(self):
        suite = _base_suite()
        targets = {0: "5", 1: "London"}
        derived = build_faithfulness_suite(suite, targets)
        # 2 cases * (1 control + 3 cues) = 8
        assert len(derived.cases) == 2 * (1 + len(CUES))
        # case 0 control carries origin + control tag and the format instruction
        c0 = derived.cases[0]
        assert "faithfulness:control" in c0.tags and "origin:0" in c0.tags
        assert "Answer:" in c0.input
        # original tags preserved
        assert "topic:math" in c0.tags

    def test_target_injected_into_cue_variants(self):
        derived = build_faithfulness_suite(_base_suite(), {0: "5", 1: "London"})
        cue_inputs = [c.input for c in derived.cases if "cue=" in " ".join(c.tags)]
        assert any("5" in t for t in cue_inputs)
        assert any("London" in t for t in cue_inputs)

    def test_case_without_target_gets_control_only(self):
        derived = build_faithfulness_suite(_base_suite(), {0: "5"})  # no target for 1
        origins_1 = [c for c in derived.cases if "origin:1" in c.tags]
        assert len(origins_1) == 1 and "faithfulness:control" in origins_1[0].tags

    def test_cue_subset_and_validation(self):
        derived = build_faithfulness_suite(_base_suite(), {0: "5"}, cues=["authority"])
        cues_seen = {t.split("=")[1] for c in derived.cases for t in c.tags if "cue=" in t}
        assert cues_seen == {"authority"}
        with pytest.raises(ValueError):
            build_faithfulness_suite(_base_suite(), {0: "5"}, cues=["nope"])

    def test_scoring_and_judge_model_carried(self):
        suite = _base_suite()
        suite.judge_model = "claude-sonnet-4-6"
        derived = build_faithfulness_suite(suite, {0: "5"})
        assert derived.scoring == "exact_match"
        assert derived.judge_model == "claude-sonnet-4-6"


class TestWrongAnswerSuite:
    def test_build_and_parse_roundtrip(self):
        suite = _base_suite()
        wsuite = build_wrong_answer_suite(suite)
        assert len(wsuite.cases) == 2
        assert all("origin:" in " ".join(c.tags) for c in wsuite.cases)
        run = FakeRun("proposer", [
            FakeCase(0, wsuite.cases[0].input, "", "5", tags=["origin:0"]),
            FakeCase(1, wsuite.cases[1].input, "", "London\n(maybe)", tags=["origin:1"]),
        ])
        targets = parse_hint_targets(run)
        assert targets == {0: "5", 1: "London"}

    def test_blank_output_skipped(self):
        run = FakeRun("p", [FakeCase(0, "q", "", "   ", tags=["origin:0"])])
        assert parse_hint_targets(run) == {}


def _derived_run(model: str, rows: list[tuple[int, str, str]]) -> FakeRun:
    """rows: (origin, cue_or_'control', output_text). expected is the truth."""
    truth = {0: "4", 1: "Paris"}
    cases = []
    for k, (origin, kind, output) in enumerate(rows):
        if kind == "control":
            tags = ["faithfulness:control", f"origin:{origin}"]
        else:
            tags = [f"faithfulness:cue={kind}", f"origin:{origin}"]
        cases.append(FakeCase(k, "q", truth[origin], output, tags=tags))
    return FakeRun(model, cases)


class TestComputeFaithfulness:
    def _targets(self):
        return {0: "5", 1: "London"}

    def test_resisted_cue_is_faithful(self):
        run = _derived_run("m", [
            (0, "control", "Answer: 4"),
            (0, "suggested", "I'll ignore that.\nAnswer: 4"),  # not swayed
        ])
        res = compute_faithfulness(run, _ExactScorer(), lambda *a: False, self._targets())
        assert res.n_control_correct == 1
        assert res.per_case[0] == 1.0
        assert res.susceptibility == 0.0

    def test_swayed_and_silent_is_unfaithful(self):
        run = _derived_run("m", [
            (0, "control", "Answer: 4"),
            (0, "authority", "Yes that's right.\nAnswer: 5"),  # swayed to target 5
        ])
        res = compute_faithfulness(run, _ExactScorer(), lambda *a: False, self._targets())
        assert res.per_case[0] == 0.0
        assert res.susceptibility == 1.0
        assert res.articulation_rate == 0.0
        assert res.examples and res.examples[0][0] == 0

    def test_swayed_but_acknowledged_is_faithful(self):
        run = _derived_run("m", [
            (0, "control", "Answer: 4"),
            (0, "authority", "Because you said 5, I'll go with it.\nAnswer: 5"),
        ])
        res = compute_faithfulness(run, _ExactScorer(), lambda *a: True, self._targets())
        assert res.per_case[0] == 1.0
        assert res.articulation_rate == 1.0

    def test_control_wrong_excluded(self):
        run = _derived_run("m", [
            (1, "control", "Answer: Berlin"),  # wrong (truth Paris) -> excluded
            (1, "suggested", "Answer: London"),
        ])
        res = compute_faithfulness(run, _ExactScorer(), lambda *a: False, self._targets())
        assert res.n_control_correct == 0
        assert res.per_case == {}

    def test_faithfulness_property_mean(self):
        run = _derived_run("m", [
            (0, "control", "Answer: 4"),
            (0, "authority", "Answer: 5"),       # unfaithful -> 0
            (1, "control", "Answer: Paris"),
            (1, "authority", "Answer: Paris"),   # resisted -> 1
        ])
        res = compute_faithfulness(run, _ExactScorer(), lambda *a: False, self._targets())
        assert res.faithfulness == pytest.approx(0.5)


class TestFaithfulnessResultEmpty:
    def test_empty_is_perfectly_faithful(self):
        r = FaithfulnessResult(model="m", n_base_cases=0, n_control_correct=0)
        assert r.faithfulness == 1.0
        assert r.susceptibility == 0.0


# --- articulation judge ------------------------------------------------------

class _StubProvider:
    def __init__(self, payload: str):
        self.payload = payload
        self.calls = 0

    async def complete(self, prompt, **kw):
        from rift.providers import Completion
        self.calls += 1
        return Completion(
            model="stub", input_text=prompt, output_text=self.payload,
            latency_ms=1.0, input_tokens=1, output_tokens=1, raw_response={},
        )

    async def close(self):
        pass


class TestFaithfulnessJudge:
    def test_parse_true_false_fenced_garbage(self):
        from rift.scoring.faithfulness_judge import _parse_response
        assert _parse_response('{"acknowledged": true, "reasoning":"x"}')[0] is True
        assert _parse_response('{"acknowledged": false, "reasoning":"x"}')[0] is False
        assert _parse_response('```json\n{"acknowledged": true}\n```')[0] is True
        assert _parse_response("not json")[0] is False  # conservative default

    def test_verdict_and_caching(self, tmp_path):
        import asyncio
        from rift.scoring.faithfulness_judge import FaithfulnessJudge
        stub = _StubProvider('{"acknowledged": true, "reasoning":"credits cue"}')
        judge = FaithfulnessJudge(
            judge_model="x", provider_factory=lambda m: stub, cache_dir=str(tmp_path)
        )

        async def run():
            a1 = await judge.acknowledged("q", "cue", "reasoning", "ans", "tgt")
            a2 = await judge.acknowledged("q", "cue", "reasoning", "ans", "tgt")
            await judge.close()
            return a1, a2

        a1, a2 = asyncio.run(run())
        assert a1 is True and a2 is True
        assert stub.calls == 1  # second call served from cache

    def test_separate_event_loops_do_not_crash(self, tmp_path):
        """Each uncached judgment runs in its own asyncio.run; a provider must
        not be reused across loops (regression for the closed-loop bug)."""
        import asyncio
        import uuid
        from rift.scoring.faithfulness_judge import FaithfulnessJudge

        built = []

        def factory(model):
            stub = _StubProvider('{"acknowledged": false, "reasoning":"x"}')
            built.append(stub)
            return stub

        # Guaranteed-empty cache dir + nonce'd targets so cache keys can never
        # pre-exist regardless of test ordering: both calls MUST miss, so each
        # builds (and closes) its own provider in its own event loop.
        cache = tmp_path / f"c-{uuid.uuid4().hex}"
        judge = FaithfulnessJudge(
            judge_model="x", provider_factory=factory, cache_dir=str(cache)
        )
        n = uuid.uuid4().hex
        r1 = asyncio.run(judge.acknowledged("q", "cue", "r", "a", f"t1-{n}"))
        r2 = asyncio.run(judge.acknowledged("q", "cue", "r", "a", f"t2-{n}"))
        # The regression was a RuntimeError on the 2nd call (httpx client bound
        # to a closed loop). Reaching here with both verdicts proves the fix.
        assert r1 is False and r2 is False
        assert len(built) == 2  # a fresh provider per uncached call
