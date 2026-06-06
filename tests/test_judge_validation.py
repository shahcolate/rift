"""Cohen's kappa + articulation-judge validation harness."""

from __future__ import annotations

import asyncio

from click.testing import CliRunner

from rift.cli import main
from rift.comparator import cohens_kappa
from rift.judge_validation import (
    GOLD_ARTICULATION,
    JudgeValidationResult,
    validate_judge,
)


class TestCohensKappa:
    def test_perfect_agreement(self):
        assert cohens_kappa([True, False, True], [True, False, True]) == 1.0

    def test_total_disagreement_is_negative(self):
        assert cohens_kappa([True, False, True, False],
                            [False, True, False, True]) < 0

    def test_chance_level_near_zero(self):
        # Judge ignores input and always says True; gold is balanced. Expected
        # agreement equals observed -> kappa ~ 0.
        gold = [True, False] * 10
        judged = [True] * 20
        assert abs(cohens_kappa(gold, judged)) < 1e-6

    def test_constant_raters_perfect_when_equal(self):
        assert cohens_kappa([True, True], [True, True]) == 1.0

    def test_constant_raters_zero_when_unequal(self):
        assert cohens_kappa([True, True], [False, False]) == 0.0

    def test_length_mismatch_raises(self):
        import pytest
        with pytest.raises(ValueError):
            cohens_kappa([True], [True, False])

    def test_empty_is_one(self):
        assert cohens_kappa([], []) == 1.0


class TestGoldSet:
    def test_balanced(self):
        acks = [e.acknowledged for e in GOLD_ARTICULATION]
        # Roughly balanced so kappa is meaningful (not flattered by imbalance).
        pos = sum(acks)
        assert 0.3 * len(acks) <= pos <= 0.7 * len(acks)

    def test_all_fields_present(self):
        for e in GOLD_ARTICULATION:
            assert e.cue and e.reasoning and e.answer and e.target
            assert isinstance(e.acknowledged, bool)


class TestValidateJudge:
    def test_perfect_judge_kappa_one(self):
        async def perfect(question, cue, reasoning, answer, target):
            # Cheat: look the case up by its reasoning text.
            for e in GOLD_ARTICULATION:
                if e.reasoning == reasoning:
                    return e.acknowledged
            return False

        r = asyncio.run(validate_judge(perfect, "stub-judge"))
        assert isinstance(r, JudgeValidationResult)
        assert r.kappa == 1.0
        assert r.accuracy == 1.0
        assert r.fp == 0 and r.fn == 0
        assert r.kappa_magnitude == "almost-perfect"

    def test_inverted_judge_negative_kappa(self):
        async def inverted(question, cue, reasoning, answer, target):
            for e in GOLD_ARTICULATION:
                if e.reasoning == reasoning:
                    return not e.acknowledged
            return False

        r = asyncio.run(validate_judge(inverted, "stub-judge"))
        assert r.kappa < 0
        assert len(r.mismatches) == r.n

    def test_constant_judge_zero_kappa(self):
        async def always_true(*a):
            return True

        r = asyncio.run(validate_judge(always_true, "stub-judge"))
        assert abs(r.kappa) < 1e-6
        # Every gold-negative becomes a false positive.
        assert r.fp == sum(1 for e in GOLD_ARTICULATION if not e.acknowledged)


class TestValidateJudgeCLI:
    def test_listed_in_help(self):
        result = CliRunner().invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "validate-judge" in result.output
