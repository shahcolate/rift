"""Tests for confidence parsing and calibration drift."""

from dataclasses import dataclass

import pytest

from rift.calibration import (
    compare_calibration,
    compute_calibration,
    parse_confidence,
)


@dataclass
class _FakeCase:
    case_index: int
    output: str
    score: float


@dataclass
class _FakeRun:
    cases: list


class TestParseConfidence:
    @pytest.mark.parametrize("text,expected", [
        ("Confidence: 0.85", 0.85),
        ("confidence: 85%", 0.85),
        ("Confidence = 0.5", 0.5),
        ("I am 90% sure the answer is X", 0.90),
        ("I'm 0.7 confident", 0.7),
        ("Confidence: 85", 0.85),       # bare > 1 → percent
        ("p: 0.95", 0.95),
    ])
    def test_known_formats(self, text, expected):
        c = parse_confidence(text)
        assert c is not None
        assert c == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize("text", [
        "",
        "The answer is 5.",
        "I think it's 42 but I'm not sure.",
        None,
    ])
    def test_unparseable(self, text):
        assert parse_confidence(text) is None

    def test_clamped(self):
        # 150% should clamp to 1.0
        assert parse_confidence("Confidence: 150%") == 1.0


class TestComputeCalibration:
    def test_perfect_calibration(self):
        # All confidence=1.0, all correct → Brier 0.
        run = _FakeRun(cases=[
            _FakeCase(i, "answer\nConfidence: 1.0", score=1.0) for i in range(10)
        ])
        c = compute_calibration(run)
        assert c.n_parsed == 10
        assert c.brier == 0.0
        assert c.ece == 0.0

    def test_overconfident(self):
        # All claim 100%, only 50% actually correct.
        cases = []
        for i in range(10):
            score = 1.0 if i % 2 == 0 else 0.0
            cases.append(_FakeCase(i, "ans\nConfidence: 1.0", score=score))
        c = compute_calibration(_FakeRun(cases=cases))
        assert c.overconfidence > 0.4
        assert c.brier > 0

    def test_unparsed_excluded(self):
        cases = [
            _FakeCase(0, "Confidence: 0.8", score=1.0),
            _FakeCase(1, "no conf here", score=0.0),
            _FakeCase(2, "Confidence: 0.8", score=1.0),
        ]
        c = compute_calibration(_FakeRun(cases=cases))
        assert c.n_parsed == 2
        assert c.n_unparsed == 1


class TestCompareCalibration:
    def test_improvement_negative_delta(self):
        # Baseline overconfident, challenger well-calibrated.
        b_cases = [_FakeCase(i, "Confidence: 0.9", score=1.0 if i < 5 else 0.0)
                   for i in range(10)]
        c_cases = [_FakeCase(i, "Confidence: 0.5", score=1.0 if i < 5 else 0.0)
                   for i in range(10)]
        comp = compare_calibration(_FakeRun(b_cases), _FakeRun(c_cases))
        # Challenger should have lower Brier — improvement is negative.
        assert comp.delta_brier < 0
