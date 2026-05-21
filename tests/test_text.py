"""Unit tests for the shared text-utility module."""

from __future__ import annotations

import pytest

from rift import _text


# ---------------------------------------------------------------------------
# parse_json_array_response
# ---------------------------------------------------------------------------


class TestParseJsonArrayResponse:
    def test_strips_triple_backtick_fences(self):
        text = '```json\n[{"input": "hi"}]\n```'
        out = _text.parse_json_array_response(text)
        assert out == [{"input": "hi"}]

    def test_strips_plain_backtick_fences(self):
        text = '```\n[{"input": "hi"}]\n```'
        out = _text.parse_json_array_response(text)
        assert out == [{"input": "hi"}]

    def test_extracts_array_from_surrounding_prose(self):
        text = 'Sure, here is the output:\n[{"input": "hi"}]\nLet me know.'
        out = _text.parse_json_array_response(text)
        assert out == [{"input": "hi"}]

    def test_returns_empty_on_non_json(self):
        assert _text.parse_json_array_response("totally not json") == []
        assert _text.parse_json_array_response("") == []
        assert _text.parse_json_array_response("{just an object}") == []

    def test_returns_empty_when_top_level_not_array(self):
        assert _text.parse_json_array_response('{"input": "hi"}') == []

    def test_drops_items_missing_required_keys(self):
        text = '[{"input": "a"}, {"other": "b"}, {"input": "c"}]'
        out = _text.parse_json_array_response(text)
        assert out == [{"input": "a"}, {"input": "c"}]

    def test_drops_items_with_empty_required_string(self):
        text = '[{"input": ""}, {"input": "   "}, {"input": "ok"}]'
        out = _text.parse_json_array_response(text)
        assert out == [{"input": "ok"}]

    def test_drops_items_with_nonstring_required_key(self):
        text = '[{"input": 42}, {"input": null}, {"input": "ok"}]'
        out = _text.parse_json_array_response(text)
        assert out == [{"input": "ok"}]

    def test_multi_required_str_keys(self):
        text = (
            '[{"input": "a", "expected": "x"},'
            ' {"input": "b"},'
            ' {"input": "c", "expected": ""},'
            ' {"input": "d", "expected": "z"}]'
        )
        out = _text.parse_json_array_response(
            text, required_str_keys=("input", "expected"),
        )
        assert out == [
            {"input": "a", "expected": "x"},
            {"input": "d", "expected": "z"},
        ]

    def test_required_keys_presence_only_accepts_non_string_values(self):
        # Regression test for the original parse_proposer_response
        # behaviour: ``expected`` only had to be present, not a
        # string. Extraction-style suites have dict / number /
        # list / bool ``expected`` values and the proposer mimics
        # the seed format. The new ``required_keys`` kwarg
        # preserves that.
        text = (
            '[{"input": "a", "expected": {"k": "v"}},'  # dict
            ' {"input": "b", "expected": 42},'           # number
            ' {"input": "c", "expected": [1, 2, 3]},'    # list
            ' {"input": "d", "expected": true},'          # bool
            ' {"input": "e"},'                            # missing → drop
            ' {"input": "f", "expected": null}]'          # null → drop
        )
        out = _text.parse_json_array_response(
            text,
            required_str_keys=("input",),
            required_keys=("expected",),
        )
        assert out == [
            {"input": "a", "expected": {"k": "v"}},
            {"input": "b", "expected": 42},
            {"input": "c", "expected": [1, 2, 3]},
            {"input": "d", "expected": True},
        ]

    def test_truncates_oversize_input_then_parses_array(self):
        # Bury a valid array inside the leading edge of a
        # >_MAX_RESPONSE_CHARS blob. The parser must truncate first
        # and still extract the array via the regex fallback against
        # the truncated text. Prose padding > the truncation budget
        # guarantees the truncation branch fires.
        valid_array = '\n[{"input": "before-truncation"}]\n'
        prose = "x" * (_text._MAX_RESPONSE_CHARS + 500)
        text = valid_array + prose
        assert len(text) > _text._MAX_RESPONSE_CHARS
        out = _text.parse_json_array_response(text)
        assert out == [{"input": "before-truncation"}]

    def test_non_dict_items_in_array_skipped(self):
        text = '[{"input": "a"}, "stringy", 42, null, {"input": "b"}]'
        out = _text.parse_json_array_response(text)
        assert out == [{"input": "a"}, {"input": "b"}]


# ---------------------------------------------------------------------------
# jaccard_5gram
# ---------------------------------------------------------------------------


class TestJaccard5gram:
    def test_identical_strings_score_one(self):
        assert _text.jaccard_5gram("the quick brown fox", "the quick brown fox") == 1.0

    def test_disjoint_strings_score_zero(self):
        assert _text.jaccard_5gram("abcdefgh", "wxyz1234") == 0.0

    def test_partial_overlap_in_middle_range(self):
        a = "the quick brown fox jumps over"
        b = "the quick brown cat jumps over"
        score = _text.jaccard_5gram(a, b)
        assert 0.3 < score < 0.95  # overlap on shared n-grams but not identical

    def test_short_strings_fall_back_to_exact_equality(self):
        # Below the 5-gram window, the function returns 1.0 only on
        # case-insensitive exact match — otherwise 0.0.
        assert _text.jaccard_5gram("hi", "hi") == 1.0
        assert _text.jaccard_5gram("Hi", "hi  ") == 1.0
        assert _text.jaccard_5gram("hi", "no") == 0.0

    def test_one_side_short_other_long_is_zero(self):
        # The short-string fallback only kicks in if BOTH are short.
        # Asymmetric: short vs. long → 0.0 (the long has 5-grams,
        # the short doesn't, no intersection possible).
        assert _text.jaccard_5gram("hi", "hello world") == 0.0


# ---------------------------------------------------------------------------
# default_provider_factory
# ---------------------------------------------------------------------------


class TestDefaultProviderFactory:
    @pytest.mark.parametrize(
        "model_id,expected_provider_name",
        [
            ("claude-opus-4-7", "AnthropicProvider"),
            ("gpt-5.5", "OpenAIProvider"),
            ("gemini-3.5-flash", "GoogleProvider"),
        ],
    )
    def test_dispatches_to_right_provider(
        self, model_id, expected_provider_name, monkeypatch
    ):
        # We don't want to instantiate real provider clients (they
        # require API keys). Patch the three provider classes with
        # markers and check dispatch.
        called = {}

        class _Marker:
            def __init__(self, **kw):
                called["kwargs"] = kw

        # Patch each provider class to the same marker so we can tell
        # *which* one would have been called by inspecting __module__.
        for name in ("AnthropicProvider", "OpenAIProvider", "GoogleProvider"):
            monkeypatch.setattr(_text, name, _Marker)
        # All three end up as _Marker; check we got an instance back
        # without raising.
        result = _text.default_provider_factory(model_id)
        assert isinstance(result, _Marker)
        assert "model" in called["kwargs"]

    def test_unknown_provider_raises_value_error(self, monkeypatch):
        # An unknown provider string from ``resolve_model`` should
        # trip the explicit ``raise ValueError(...)`` branch.
        from rift.config import ModelConfig
        monkeypatch.setattr(
            _text, "resolve_model",
            lambda _m: ModelConfig(provider="not-a-vendor", model="x"),
        )
        with pytest.raises(ValueError, match="No provider available"):
            _text.default_provider_factory("anything")
