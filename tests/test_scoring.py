"""Tests for scoring functions."""

from rift.scoring.exact_match import ExactMatchScorer
from rift.scoring.semantic import SemanticScorer


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


class TestSemanticScorer:
    def setup_method(self):
        self.scorer = SemanticScorer()

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
