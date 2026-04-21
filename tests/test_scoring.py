"""Tests for scoring functions."""

import pytest

from rift.scoring import get_scorer
from rift.scoring.exact_match import ExactMatchScorer
from rift.scoring.f1 import F1Scorer
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


class TestF1Scorer:
    def setup_method(self):
        self.scorer = F1Scorer()

    def test_identical_strings(self):
        assert self.scorer.score("the cat sat on the mat", "the cat sat on the mat") == 1.0

    def test_completely_disjoint(self):
        assert self.scorer.score("alpha beta gamma", "delta epsilon zeta") == 0.0

    def test_paraphrase_gets_partial_credit(self):
        # output 5 tokens, expected 7 tokens, overlap 4 ({the, cat, on, mat})
        # precision 4/5, recall 4/7 → F1 ≈ 0.6667
        score = self.scorer.score(
            "the cat sat on mat",
            "the cat is sitting on the mat",
        )
        assert score == pytest.approx(0.6667, abs=1e-3)

    def test_word_order_invariant(self):
        # Same tokens, different order → F1 = 1.0 (bag-of-words)
        assert self.scorer.score("alpha beta gamma", "gamma alpha beta") == 1.0

    def test_case_insensitive(self):
        assert self.scorer.score("Hello World", "hello world") == 1.0

    def test_punctuation_stripped(self):
        assert self.scorer.score("hello, world!", "hello world") == 1.0

    def test_repeated_tokens_clipped(self):
        # output repeats "cat" — clipped intersection is still 1
        # precision = 1/3, recall = 1/1 → F1 = 0.5
        assert self.scorer.score("cat cat cat", "cat") == 0.5

    def test_empty_inputs(self):
        assert self.scorer.score("", "hello") == 0.0
        assert self.scorer.score("hello", "") == 0.0

    def test_non_string_expected(self):
        # str() conversion: numbers, etc.
        assert self.scorer.score("42", 42) == 1.0

    def test_score_range(self):
        score = self.scorer.score("the quick brown fox", "a quick red fox")
        assert 0.0 <= score <= 1.0


class TestGetScorer:
    def test_returns_f1_scorer(self):
        scorer = get_scorer("f1")
        assert isinstance(scorer, F1Scorer)

    def test_unknown_scorer_raises(self):
        with pytest.raises(ValueError, match="Unknown scorer"):
            get_scorer("nonexistent")
