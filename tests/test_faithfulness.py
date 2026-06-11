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
    # Field names mirror the real runner.CaseResult — a fixture that
    # diverges (e.g. `input` instead of `input_text`) masks attribute
    # bugs in code paths the fakes exercise.
    case_index: int
    input_text: str
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

    def test_judge_receives_control_question_text(self):
        # Regression: compute_faithfulness crashed with AttributeError on
        # the first live run that produced a swayed case — it read
        # ``control.input`` (suite EvalCase field) off a CaseResult,
        # whose field is ``input_text``. Assert the judge gets the
        # actual question string, not an attribute error.
        run = _derived_run("m", [
            (0, "control", "Answer: 4"),
            (0, "authority", "Answer: 5"),  # swayed -> judge is consulted
        ])
        seen = {}

        def ack(question, cue_text, reasoning, answer, target):
            seen["q"] = question
            return False

        compute_faithfulness(run, _ExactScorer(), ack, self._targets())
        assert seen["q"] == "q"  # the control CaseResult's input_text

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


# --- Phase 2: CoT-dependence ------------------------------------------------

from rift.faithfulness import (  # noqa: E402
    COT_PERTURBATIONS,
    CotFaithfulnessResult,
    _alternative_answer,
    _inject_mistake,
    _truncate_reasoning,
    build_control_suite,
    build_cot_perturbation_suite,
    compute_cot_faithfulness,
)


class TestBuildControlSuite:
    def test_one_control_per_case_with_tags(self):
        suite = _base_suite()
        ctrl = build_control_suite(suite)
        assert len(ctrl.cases) == 2
        for i, c in enumerate(ctrl.cases):
            assert "faithfulness:control" in c.tags
            assert f"origin:{i}" in c.tags
            assert "Answer:" in c.input
        assert ctrl.scoring == "exact_match"


class TestTruncateReasoning:
    def test_keeps_first_half_of_lines(self):
        r = "step one\nstep two\nstep three\nstep four"
        out = _truncate_reasoning(r)
        assert "step one" in out and "step four" not in out

    def test_single_line_halved_by_sentence(self):
        r = "First we note A. Then we conclude B."
        out = _truncate_reasoning(r)
        assert "First we note A" in out and "conclude B" not in out

    def test_empty(self):
        assert _truncate_reasoning("   ") == ""


def _cot_control_run(model, rows):
    """rows: (origin, output). expected truth from _base_suite (4 / Paris)."""
    truth = {0: "4", 1: "Paris"}
    cases = [
        FakeCase(k, "q", truth[o], out,
                 tags=["faithfulness:control", f"origin:{o}"])
        for k, (o, out) in enumerate(rows)
    ]
    return FakeRun(model, cases)


class TestBuildCotPerturbationSuite:
    def test_variants_only_for_control_correct_with_reasoning(self):
        suite = _base_suite()
        run = _cot_control_run("m", [
            (0, "First, 2+2.\nThat is 4.\nAnswer: 4"),     # correct + reasoning
            (1, "Answer: Berlin"),                          # wrong control -> excluded
        ])
        pert, control_answers = build_cot_perturbation_suite(
            suite, run, _ExactScorer()
        )
        assert control_answers[0] == "4"
        # origin 1 wrong -> no perturbations; origin 0 -> 2 perturbations
        kinds = {_origin_and_kind(c) for c in pert.cases}
        assert (0, "early") in kinds and (0, "mistake") in kinds
        assert not any(o == 1 for (o, _k) in kinds)

    def test_correct_but_no_reasoning_gets_no_variants(self):
        suite = _base_suite()
        run = _cot_control_run("m", [(0, "Answer: 4")])  # correct, no CoT
        pert, control_answers = build_cot_perturbation_suite(
            suite, run, _ExactScorer()
        )
        assert control_answers[0] == "4"
        assert len(pert.cases) == 0  # nothing to perturb

    def test_perturbation_subset_and_validation(self):
        suite = _base_suite()
        run = _cot_control_run("m", [(0, "reasoning here\nAnswer: 4")])
        pert, _ = build_cot_perturbation_suite(
            suite, run, _ExactScorer(), perturbations=["early"]
        )
        kinds = {k for (_o, k) in (_origin_and_kind(c) for c in pert.cases)}
        assert kinds == {"early"}
        with pytest.raises(ValueError):
            build_cot_perturbation_suite(suite, run, _ExactScorer(),
                                         perturbations=["bogus"])


def _origin_and_kind(case):
    o = next((int(t.split(":")[1]) for t in case.tags if t.startswith("origin:")), None)
    k = next((t.split("=")[1] for t in case.tags if t.startswith("faithfulness:cot=")), None)
    return (o, k)


def _cot_perturbed_run(model, rows):
    """rows: (origin, kind, output)."""
    cases = [
        FakeCase(k, "q", "x", out,
                 tags=[f"faithfulness:cot={kind}", f"origin:{o}"])
        for k, (o, kind, out) in enumerate(rows)
    ]
    return FakeRun(model, cases)


class TestComputeCotFaithfulness:
    def test_flip_is_faithful(self):
        # Answer changed from control "4" -> "5": the CoT drove it (faithful).
        run = _cot_perturbed_run("m", [(0, "early", "Answer: 5")])
        res = compute_cot_faithfulness(run, {0: "4"})
        assert res.per_case[0] == 1.0
        assert res.flip_rate == 1.0

    def test_unchanged_is_unfaithful(self):
        # Answer stayed "4" despite corrupted reasoning: post-hoc (unfaithful).
        run = _cot_perturbed_run("m", [(0, "mistake", "Answer: 4")])
        res = compute_cot_faithfulness(run, {0: "4"})
        assert res.per_case[0] == 0.0
        assert res.flip_rate == 0.0
        assert res.examples and res.examples[0][0] == 0

    def test_per_case_mean_over_perturbations(self):
        run = _cot_perturbed_run("m", [
            (0, "early", "Answer: 5"),    # flip
            (0, "mistake", "Answer: 4"),  # no flip
        ])
        res = compute_cot_faithfulness(run, {0: "4"})
        assert res.per_case[0] == pytest.approx(0.5)

    def test_cases_outside_control_answers_ignored(self):
        run = _cot_perturbed_run("m", [(9, "early", "Answer: z")])
        res = compute_cot_faithfulness(run, {0: "4"})
        assert res.per_case == {}

    def test_empty_is_perfectly_faithful_default(self):
        r = CotFaithfulnessResult(model="m", n_base_cases=0, n_control_correct=0)
        assert r.faithfulness == 1.0


class TestAlternativeAnswer:
    def test_integer_mutates_to_different_int(self):
        alt = _alternative_answer("4")
        assert alt is not None and alt != "4"
        assert "." not in alt  # rendered as int

    def test_float_mutates(self):
        alt = _alternative_answer("3.5")
        assert alt is not None and alt != "3.5"

    def test_boolean_flips(self):
        assert _alternative_answer("yes") == "no"
        assert _alternative_answer("True") == "false"

    def test_freeform_has_no_safe_mutation(self):
        assert _alternative_answer("Paris") is None


class TestInjectMistake:
    def test_appends_and_preserves_original(self):
        out = _inject_mistake("2+2=4", "4")
        assert out.startswith("2+2=4")
        assert len(out) > len("2+2=4")

    def test_numeric_injects_specific_alternative(self):
        # The corruption must name a concrete different number, not a
        # content-free "it's different" nudge — otherwise a faithful model has
        # nothing definite to adopt and the flip signal is confounded.
        out = _inject_mistake("six times seven", "42")
        assert "43" in out
        assert "42" in out  # references the original it's overturning

    def test_freeform_forbids_original_answer(self):
        out = _inject_mistake("the capital of France", "Paris")
        assert "Paris" in out
        assert "not" in out.lower()


def test_cot_perturbations_constant():
    assert COT_PERTURBATIONS == ("early", "mistake")


class TestScoreAnswer:
    def test_scores_parsed_answer_not_raw_output(self):
        # Regression: a correct step-by-step output must score 1.0 even though
        # the raw text != expected. exact_match on the whole string would fail.
        from rift.faithfulness import _score_answer
        from rift.scoring import get_scorer
        scorer = get_scorer("exact_match")
        out = "Let me reason.\nStep two.\nAnswer: 4"
        assert _score_answer(scorer, out, "4") == 1.0
        assert scorer.score(out, "4") == 0.0  # proves the raw output would fail

    def test_wrong_parsed_answer_scores_zero(self):
        from rift.faithfulness import _score_answer
        from rift.scoring import get_scorer
        scorer = get_scorer("exact_match")
        assert _score_answer(scorer, "reasoning\nAnswer: 5", "4") == 0.0


class TestScoreAnswerDictExpected:
    def test_multiline_json_after_answer_still_scores(self):
        # Regression for the review finding: dict-expected (structured
        # extraction) suites emit multi-line JSON; _parse_answer keeps only the
        # first line, so _score_answer must fall back to scoring the whole
        # output (the scorer's JSON extraction scans the full text).
        from rift.faithfulness import _score_answer
        from rift.scoring import get_scorer
        scorer = get_scorer("exact_match")
        expected = {"invoice": "4521", "total": "1240"}
        out = (
            "Let me extract the fields.\n"
            'Answer:\n{\n  "invoice": "4521",\n  "total": "1240"\n}'
        )
        assert _score_answer(scorer, out, expected) == 1.0

    def test_scalar_still_prefers_parsed_answer(self):
        from rift.faithfulness import _score_answer
        from rift.scoring import get_scorer
        scorer = get_scorer("exact_match")
        assert _score_answer(scorer, "reasoning\nAnswer: 4", "4") == 1.0


class TestCotExamplesDedup:
    def test_one_example_per_origin(self):
        # Both perturbations of the same case fail to flip -> only ONE example.
        run = _cot_perturbed_run("m", [
            (0, "early", "Answer: 4"),
            (0, "mistake", "Answer: 4"),
        ])
        res = compute_cot_faithfulness(run, {0: "4"})
        assert [e[0] for e in res.examples] == [0]  # not [0, 0]
