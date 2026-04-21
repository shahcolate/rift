"""Tests for context-rot suite expansion."""

from rift.config import EvalCase, SuiteConfig
from rift.context_rot import DEFAULT_LEVELS, DistractorLevel, expand_suite


def _mini_suite() -> SuiteConfig:
    return SuiteConfig(
        name="mini",
        description="test",
        scoring="exact_match",
        cases=[
            EvalCase(input="What is 1+1?", expected="2"),
            EvalCase(input="What is 2+2?", expected="4"),
        ],
    )


class TestExpandSuite:
    def test_expansion_count(self):
        suite = _mini_suite()
        expanded = expand_suite(suite)
        assert len(expanded.cases) == len(suite.cases) * len(DEFAULT_LEVELS)

    def test_zero_level_preserves_input(self):
        suite = _mini_suite()
        expanded = expand_suite(suite, levels=(DistractorLevel("0k", 0),))
        # With zero distractor tokens, inputs must round-trip unchanged.
        assert expanded.cases[0].input == "What is 1+1?"
        assert expanded.cases[1].input == "What is 2+2?"

    def test_tags_carry_level_origin_position(self):
        suite = _mini_suite()
        expanded = expand_suite(suite, levels=(DistractorLevel("8k", 8000),))
        tags = expanded.cases[0].tags
        assert any(t == "distractor:8k" for t in tags)
        assert any(t == "origin:0" for t in tags)
        assert any(t.startswith("position:") for t in tags)

    def test_determinism(self):
        suite = _mini_suite()
        a = expand_suite(suite)
        b = expand_suite(suite)
        for ca, cb in zip(a.cases, b.cases):
            assert ca.input == cb.input

    def test_distractor_grows_prompt(self):
        suite = _mini_suite()
        expanded = expand_suite(
            suite,
            levels=(
                DistractorLevel("0k", 0),
                DistractorLevel("8k", 8000),
            ),
        )
        # Case 0 at 0k, case 1 at 8k (level varies fastest).
        assert len(expanded.cases[0].input) < len(expanded.cases[1].input)

    def test_expected_preserved(self):
        suite = _mini_suite()
        expanded = expand_suite(suite)
        # Every expansion of case 0 must keep "2" as expected answer.
        origin_0 = [c for c in expanded.cases if "origin:0" in c.tags]
        assert all(c.expected == "2" for c in origin_0)
