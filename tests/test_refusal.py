"""Tests for refusal classification and refusal-drift analysis."""

from dataclasses import dataclass

import pytest

from rift.refusal import (
    classify_output,
    classify_run,
    compare_refusal,
)


@dataclass
class _FakeCase:
    case_index: int
    output: str
    score: float = 1.0


@dataclass
class _FakeRun:
    cases: list


class TestClassifyOutput:
    @pytest.mark.parametrize("text", [
        "I cannot help with that request.",
        "I'm sorry, but I can't provide that.",
        "Unfortunately, I am unable to assist with this.",
        "I won't be able to help with that.",
        "As an AI language model, I cannot generate such content.",
        "That would be unsafe and I must decline.",
        "I do not provide instructions for that.",
    ])
    def test_known_refusal_patterns(self, text):
        refused, pattern = classify_output(text)
        assert refused is True
        assert pattern is not None

    @pytest.mark.parametrize("text", [
        "The answer is 42.",
        "Paris is the capital of France.",
        "Here is the code:\n```python\nprint('hello')\n```",
        "True",
        "5",
        "",
    ])
    def test_known_non_refusals(self, text):
        refused, pattern = classify_output(text)
        assert refused is False
        assert pattern is None

    def test_prefix_scan_only(self):
        # A refusal-like phrase deep in the output should not trigger.
        text = "The answer is 42. " + ("x" * 1000) + " I cannot do that."
        refused, _ = classify_output(text)
        assert refused is False


class TestClassifyRun:
    def test_counts(self):
        run = _FakeRun(cases=[
            _FakeCase(0, "The answer is 5."),
            _FakeCase(1, "I cannot help with that."),
            _FakeCase(2, "True"),
            _FakeCase(3, "Sorry, I am unable to assist with that."),
        ])
        a = classify_run(run)
        assert a.n_cases == 4
        assert a.n_refused == 2
        assert a.refusal_rate == 0.5


class TestCompareRefusal:
    def test_over_refusal_only_when_baseline_correct(self):
        baseline = _FakeRun(cases=[
            _FakeCase(0, "5", score=1.0),
            _FakeCase(1, "42", score=1.0),
            _FakeCase(2, "wrong", score=0.0),
        ])
        challenger = _FakeRun(cases=[
            _FakeCase(0, "I cannot help with that.", score=0.0),
            _FakeCase(1, "42", score=1.0),
            _FakeCase(2, "I am unable to provide that.", score=0.0),
        ])
        a = compare_refusal(baseline, challenger)
        # Case 0: over-refusal (baseline answered correctly).
        assert 0 in a.over_refusal_cases
        # Case 2: challenger refused but baseline was wrong anyway — not over-refusal.
        assert 2 not in a.over_refusal_cases
        assert a.delta_refusal_rate == pytest.approx(2 / 3, abs=1e-3)

    def test_new_compliance_tracking(self):
        baseline = _FakeRun(cases=[
            _FakeCase(0, "I cannot help.", score=0.0),
        ])
        challenger = _FakeRun(cases=[
            _FakeCase(0, "The answer is 5.", score=1.0),
        ])
        a = compare_refusal(baseline, challenger)
        assert a.new_compliance_cases == [0]
        assert a.over_refusal_cases == []

    def test_length_mismatch_errors(self):
        with pytest.raises(ValueError):
            compare_refusal(
                _FakeRun(cases=[_FakeCase(0, "a")]),
                _FakeRun(cases=[]),
            )
