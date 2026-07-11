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

    def test_unknown_model_resolves_lazily_but_fails_live(self):
        # Unknown strings resolve to the 'local' pseudo-provider so cached
        # runs stay keyless — but building a live provider for one must
        # fail with the clean remedy (exit 2), never an all-errored run.
        import pytest

        from rift.config import UnknownModelError
        from rift.runner import _get_provider

        config = resolve_model("my-local-model")
        assert config.provider == "local"
        with pytest.raises(UnknownModelError) as exc_info:
            _get_provider(config)
        assert exc_info.value.exit_code == 2
        assert "@" in exc_info.value.message  # points at the endpoint syntax

    def test_openai_compatible_endpoint_syntax(self):
        config = resolve_model("llama-3.3-70b@http://localhost:8000")
        assert config.provider == "openai_compatible"
        assert config.model == "llama-3.3-70b"
        assert config.api_base == "http://localhost:8000"

    def test_endpoint_syntax_requires_scheme_and_name(self):
        import pytest

        from rift.config import UnknownModelError

        with pytest.raises(UnknownModelError):
            resolve_model("@http://localhost:8000")

    def test_opus_4_8_alias(self):
        # Opus 4.8 must resolve to the Anthropic provider, not fall
        # through to the "local" catch-all (which it would without an
        # explicit alias since the short form doesn't start with "claude").
        config = resolve_model("opus-4-8")
        assert config.provider == "anthropic"
        assert config.model == "claude-opus-4-8"


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
