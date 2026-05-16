"""Tests for sycophancy pushback-suite construction and flip-rate analysis."""

from dataclasses import dataclass

import pytest

from rift.config import EvalCase, SuiteConfig
from rift.sycophancy import (
    build_pushback_suite,
    compute_sycophancy,
)


@dataclass
class _FakeCase:
    case_index: int
    output: str
    score: float


@dataclass
class _FakeRun:
    cases: list


def _suite(*pairs):
    return SuiteConfig(
        name="t",
        scoring="exact_match",
        cases=[EvalCase(input=inp, expected=exp) for inp, exp in pairs],
    )


class TestBuildPushbackSuite:
    def test_includes_original_answer_and_challenge(self):
        suite = _suite(("What is 2+2?", "4"))
        run = _FakeRun(cases=[_FakeCase(0, "4", 1.0)])
        pb = build_pushback_suite(suite, run)
        assert len(pb.cases) == 1
        # Original answer should appear in the new prompt.
        assert "4" in pb.cases[0].input
        # And some kind of challenge phrasing.
        challenge_markers = ["sure", "don't think", "professor", "reconsider"]
        assert any(m in pb.cases[0].input.lower() for m in challenge_markers)
        # Tag is preserved + sycophancy tag added.
        assert "sycophancy:pushback" in pb.cases[0].tags

    def test_true_false_picks_inverse(self):
        suite = _suite(("Is the sky blue?", "True"))
        run = _FakeRun(cases=[_FakeCase(0, "True", 1.0)])
        pb = build_pushback_suite(suite, run)
        # Wrong-hint should be the inverse.
        assert "False" in pb.cases[0].input

    def test_multiple_choice_picks_different_letter(self):
        suite = _suite(("Pick: A) x B) y C) z D) w", "B"))
        run = _FakeRun(cases=[_FakeCase(0, "B", 1.0)])
        pb = build_pushback_suite(suite, run)
        # Wrong-hint should be a different letter, not B.
        # (Specifically "A" per _pick_wrong_hint mapping.)
        text = pb.cases[0].input
        # Use word-boundary check to avoid matching "B" elsewhere.
        assert "A" in text


class TestComputeSycophancy:
    def test_flip_rate_only_counts_originally_correct(self):
        orig = _FakeRun(cases=[
            _FakeCase(0, "4", 1.0),     # correct
            _FakeCase(1, "wrong", 0.0), # wrong
            _FakeCase(2, "5", 1.0),     # correct
            _FakeCase(3, "5", 1.0),     # correct
        ])
        # Pushback: case 0 flips to wrong, 2 holds, 3 flips. Case 1
        # remains wrong (not counted in flip-rate denominator).
        push = _FakeRun(cases=[
            _FakeCase(0, "no idea", 0.0),
            _FakeCase(1, "still wrong", 0.0),
            _FakeCase(2, "5", 1.0),
            _FakeCase(3, "wrong", 0.0),
        ])
        a = compute_sycophancy(orig, push)
        assert a.n_originally_correct == 3
        assert a.n_flipped_to_wrong == 2
        assert a.flip_rate == pytest.approx(2 / 3, abs=1e-3)
        assert a.flipped_cases == [0, 3]

    def test_recovery_rate(self):
        orig = _FakeRun(cases=[
            _FakeCase(0, "wrong", 0.0),
            _FakeCase(1, "wrong", 0.0),
        ])
        push = _FakeRun(cases=[
            _FakeCase(0, "right", 1.0),
            _FakeCase(1, "wrong", 0.0),
        ])
        a = compute_sycophancy(orig, push)
        assert a.n_originally_wrong == 2
        assert a.n_flipped_to_right == 1
        assert a.recovery_rate == 0.5

    def test_length_mismatch_errors(self):
        with pytest.raises(ValueError):
            compute_sycophancy(
                _FakeRun(cases=[_FakeCase(0, "x", 1.0)]),
                _FakeRun(cases=[]),
            )
