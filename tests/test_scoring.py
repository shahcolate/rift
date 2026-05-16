"""Tests for scoring functions."""

from rift.scoring.exact_match import ExactMatchScorer
from rift.scoring.fuzzy_match import FuzzyMatchScorer


class TestExactMatchScorer:
    def setup_method(self):
        self.scorer = ExactMatchScorer()

    def test_string_exact_match(self):
        assert self.scorer.score("hello", "hello") == 1.0

    def test_string_mismatch(self):
        assert self.scorer.score("hello", "world") == 0.0

    def test_string_whitespace_tolerance(self):
        assert self.scorer.score("  hello  ", "hello") == 1.0

    def test_dict_full_match(self):
        output = '{"name": "Alice", "age": "30"}'
        expected = {"name": "Alice", "age": "30"}
        assert self.scorer.score(output, expected) == 1.0

    def test_dict_partial_match(self):
        output = '{"name": "Alice", "age": "25"}'
        expected = {"name": "Alice", "age": "30"}
        assert self.scorer.score(output, expected) == 0.5

    def test_dict_no_match(self):
        output = '{"name": "Bob", "age": "25"}'
        expected = {"name": "Alice", "age": "30"}
        assert self.scorer.score(output, expected) == 0.0

    def test_dict_json_in_markdown(self):
        output = '```json\n{"name": "Alice"}\n```'
        expected = {"name": "Alice"}
        assert self.scorer.score(output, expected) == 1.0

    def test_dict_json_with_surrounding_text(self):
        output = 'Here is the result: {"name": "Alice"} hope that helps'
        expected = {"name": "Alice"}
        assert self.scorer.score(output, expected) == 1.0

    def test_dict_invalid_json(self):
        output = "not json at all"
        expected = {"name": "Alice"}
        assert self.scorer.score(output, expected) == 0.0


class TestExactMatchConfidenceTolerance:
    """The exact-match scorer should ignore a trailing confidence line."""

    def setup_method(self):
        self.scorer = ExactMatchScorer()

    def test_strips_trailing_confidence_colon(self):
        assert self.scorer.score("8.40\nConfidence: 0.9", "8.40") == 1.0

    def test_strips_trailing_confidence_percent(self):
        assert self.scorer.score("True\nConfidence: 85%", "True") == 1.0

    def test_strips_trailing_im_sure_form(self):
        assert self.scorer.score(
            "Bob\nI am 90% sure", "Bob"
        ) == 1.0

    def test_strips_trailing_p_form(self):
        assert self.scorer.score("5\np: 0.7", "5") == 1.0

    def test_does_not_strip_midline_confidence(self):
        # A confidence-shaped token inside the answer should NOT be
        # stripped — only a trailing standalone line is.
        assert self.scorer.score(
            "I am 50% confident in this answer is 5", "5"
        ) == 0.0

    def test_unchanged_when_no_confidence(self):
        assert self.scorer.score("42", "42") == 1.0
        assert self.scorer.score("wrong", "42") == 0.0

    def test_wrong_answer_with_confidence_still_zero(self):
        assert self.scorer.score("9\nConfidence: 0.99", "5") == 0.0


class TestFuzzyMatchScorer:
    def setup_method(self):
        self.scorer = FuzzyMatchScorer()

    def test_identical_strings(self):
        assert self.scorer.score("hello world", "hello world") == 1.0

    def test_similar_strings(self):
        score = self.scorer.score(
            "the cat sat on the mat", "the cat is sitting on the mat"
        )
        assert score > 0.7

    def test_dissimilar_strings(self):
        score = self.scorer.score(
            "quantum computing breakthrough", "best chocolate cake recipe"
        )
        assert score < 0.3

    def test_empty_strings(self):
        assert self.scorer.score("", "hello") == 0.0
        assert self.scorer.score("hello", "") == 0.0
