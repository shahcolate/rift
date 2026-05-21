"""Tests for the exec_tests scorer used by code_generation suites."""

from rift.scoring import get_scorer
from rift.scoring.exec_tests import ExecTestsScorer


FIZZBUZZ_SPEC = {
    "function": "fizzbuzz",
    "tests": [
        {"args": [5], "expected": ["1", "2", "Fizz", "4", "Buzz"]},
        {"args": [1], "expected": ["1"]},
    ],
}


CORRECT_FIZZBUZZ = """
def fizzbuzz(n):
    out = []
    for i in range(1, n + 1):
        if i % 15 == 0: out.append("FizzBuzz")
        elif i % 3 == 0: out.append("Fizz")
        elif i % 5 == 0: out.append("Buzz")
        else: out.append(str(i))
    return out
"""


def test_correct_function_scores_one() -> None:
    s = ExecTestsScorer()
    assert s.score(CORRECT_FIZZBUZZ, FIZZBUZZ_SPEC) == 1.0


def test_fenced_code_is_extracted() -> None:
    fenced = "```python\n" + CORRECT_FIZZBUZZ + "\n```"
    assert ExecTestsScorer().score(fenced, FIZZBUZZ_SPEC) == 1.0


def test_missing_function_scores_zero() -> None:
    code = "def something_else(n): return []"
    assert ExecTestsScorer().score(code, FIZZBUZZ_SPEC) == 0.0


def test_syntax_error_scores_zero() -> None:
    assert ExecTestsScorer().score("def fizzbuzz(n) return", FIZZBUZZ_SPEC) == 0.0


def test_runtime_exception_scores_zero() -> None:
    code = "def fizzbuzz(n):\n    raise RuntimeError('nope')"
    assert ExecTestsScorer().score(code, FIZZBUZZ_SPEC) == 0.0


def test_partial_pass_returns_fraction() -> None:
    # Drops the Buzz rule -> fails on n=5 (case 0), passes on n=1 (case 1).
    code = """
def fizzbuzz(n):
    return [("Fizz" if i % 3 == 0 else str(i)) for i in range(1, n+1)]
"""
    score = ExecTestsScorer().score(code, FIZZBUZZ_SPEC)
    assert score == 0.5


def test_timeout_scores_zero() -> None:
    code = "def fizzbuzz(n):\n    while True: pass"
    # Tight timeout so the test stays snappy.
    assert ExecTestsScorer(timeout_s=0.5).score(code, FIZZBUZZ_SPEC) == 0.0


def test_malformed_expected_scores_zero() -> None:
    s = ExecTestsScorer()
    assert s.score(CORRECT_FIZZBUZZ, "function") == 0.0
    assert s.score(CORRECT_FIZZBUZZ, {"function": "fizzbuzz"}) == 0.0
    assert s.score(CORRECT_FIZZBUZZ, {"function": "fizzbuzz", "tests": []}) == 0.0


def test_registered_in_get_scorer() -> None:
    assert isinstance(get_scorer("exec_tests"), ExecTestsScorer)
