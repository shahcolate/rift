"""Tests for config parsing."""

import pytest
from rift.config import resolve_model, SuiteConfig, EvalCase


class TestResolveModel:
    def test_anthropic_model(self):
        config = resolve_model("claude-3-5-sonnet")
        assert config.provider == "anthropic"
        assert config.model == "claude-3-5-sonnet"

    def test_openai_model(self):
        config = resolve_model("gpt-4o")
        assert config.provider == "openai"
        assert config.model == "gpt-4o"

    def test_o1_model(self):
        config = resolve_model("o1-preview")
        assert config.provider == "openai"
        assert config.model == "o1-preview"

    def test_unknown_model(self):
        config = resolve_model("my-local-model")
        assert config.provider == "local"


class TestSuiteConfig:
    def test_valid_suite(self):
        suite = SuiteConfig(
            name="test",
            scoring="exact_match",
            cases=[EvalCase(input="hello", expected="world")],
        )
        assert suite.name == "test"
        assert len(suite.cases) == 1

    def test_invalid_scoring(self):
        with pytest.raises(ValueError):
            SuiteConfig(
                name="test",
                scoring="invalid_scorer",
                cases=[EvalCase(input="hello", expected="world")],
            )
