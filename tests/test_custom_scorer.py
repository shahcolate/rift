"""Tests for user-supplied custom scorers (scoring: custom)."""

from __future__ import annotations

import asyncio
import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from rift.config import SuiteConfig
from rift.scoring import get_scorer
from rift.scoring.custom import load_custom_scorer


@pytest.fixture
def scorer_file(tmp_path) -> Path:
    p = tmp_path / "myscorer.py"
    p.write_text(textwrap.dedent('''
        def score(output, expected):
            return 1.0 if str(expected).lower() in output.lower() else 0.0

        def returns_int(output, expected):
            return 1  # exercises the float() coercion

        class KeywordScorer:
            def score(self, output, expected):
                return float(all(w in output for w in str(expected).split()))

        instance_scorer = KeywordScorer()

        async def ascore(output, expected, context=None):
            return 1.0 if output.strip() == str(expected).strip() else 0.0

        async def ascore_no_ctx(output, expected):
            return 0.5

        not_callable = 42
    '''))
    return p


class TestLoadCustomScorer:
    def test_function(self, scorer_file):
        s = load_custom_scorer(f"{scorer_file}:score")
        assert s.score("the CAT sat", "cat") == 1.0
        assert s.score("dog", "cat") == 0.0
        assert not hasattr(s, "ascore")  # runner treats it as sync

    def test_function_return_coerced_to_float(self, scorer_file):
        s = load_custom_scorer(f"{scorer_file}:returns_int")
        v = s.score("x", "y")
        assert v == 1.0 and isinstance(v, float)

    def test_class_instantiated(self, scorer_file):
        s = load_custom_scorer(f"{scorer_file}:KeywordScorer")
        assert s.score("big red dog", "red dog") == 1.0
        assert s.score("big dog", "red dog") == 0.0

    def test_instance(self, scorer_file):
        s = load_custom_scorer(f"{scorer_file}:instance_scorer")
        assert s.score("a b c", "a b") == 1.0

    def test_async_function(self, scorer_file):
        s = load_custom_scorer(f"{scorer_file}:ascore")
        assert hasattr(s, "ascore")
        assert asyncio.run(s.ascore("4", "4", context="q")) == 1.0
        assert asyncio.run(s.ascore("5", "4")) == 0.0

    def test_async_function_without_context_param(self, scorer_file):
        # The runner always passes context=; the wrapper must not forward it
        # to a function that doesn't accept it.
        s = load_custom_scorer(f"{scorer_file}:ascore_no_ctx")
        assert asyncio.run(s.ascore("x", "y", context="q")) == 0.5

    def test_async_function_with_kwargs_gets_context(self, tmp_path):
        # An async scorer using **kwargs should receive context, not have it
        # silently dropped.
        p = tmp_path / "kw.py"
        p.write_text(
            "async def ascore(output, expected, **kwargs):\n"
            "    return 1.0 if kwargs.get('context') == 'CTX' else 0.0\n"
        )
        s = load_custom_scorer(f"{p}:ascore")
        assert asyncio.run(s.ascore("o", "e", context="CTX")) == 1.0

    def test_non_numeric_return_raises_clear_error(self, tmp_path):
        p = tmp_path / "bad.py"
        p.write_text("def score(output, expected):\n    return None\n")
        s = load_custom_scorer(f"{p}:score")
        with pytest.raises(ValueError, match="non-numeric"):
            s.score("o", "e")

    def test_module_path_import(self, monkeypatch, scorer_file):
        # Importable module form (target with no path markers).
        monkeypatch.syspath_prepend(str(scorer_file.parent))
        s = load_custom_scorer("myscorer:score")
        assert s.score("xcatx", "cat") == 1.0

    def test_relative_path_resolves_against_base_dir(self, scorer_file):
        s = load_custom_scorer("./myscorer.py:score", base_dir=scorer_file.parent)
        assert s.score("xcatx", "cat") == 1.0

    # --- error cases ---

    def test_missing_colon(self, scorer_file):
        with pytest.raises(ValueError, match="module.path:callable"):
            load_custom_scorer("no_colon_here")

    def test_missing_attribute(self, scorer_file):
        with pytest.raises(ValueError, match="not found in"):
            load_custom_scorer(f"{scorer_file}:does_not_exist")

    def test_missing_file(self, tmp_path):
        with pytest.raises(ValueError, match="file not found"):
            load_custom_scorer(f"{tmp_path / 'nope.py'}:score")

    def test_unimportable_module(self):
        with pytest.raises(ValueError, match="could not import"):
            load_custom_scorer("rift_no_such_module_xyz:score")

    def test_non_callable_target(self, scorer_file):
        with pytest.raises(ValueError, match="must be a callable"):
            load_custom_scorer(f"{scorer_file}:not_callable")


class TestGetScorerCustom:
    def test_get_scorer_custom(self, scorer_file):
        s = get_scorer("custom", custom_scorer=f"{scorer_file}:score")
        assert s.score("xcatx", "cat") == 1.0

    def test_get_scorer_custom_requires_spec(self):
        with pytest.raises(ValueError, match="requires a 'custom_scorer'"):
            get_scorer("custom")


class TestSuiteConfigCustom:
    def test_valid(self):
        c = SuiteConfig(name="x", scoring="custom", custom_scorer="./s.py:score",
                        cases=[{"input": "a", "expected": "b"}])
        assert c.custom_scorer == "./s.py:score"

    def test_custom_requires_scorer(self):
        with pytest.raises(ValidationError, match="requires a 'custom_scorer'"):
            SuiteConfig(name="x", scoring="custom",
                        cases=[{"input": "a", "expected": "b"}])

    def test_scorer_requires_custom_scoring(self):
        with pytest.raises(ValidationError, match="only valid when scoring: custom"):
            SuiteConfig(name="x", scoring="exact_match", custom_scorer="./s.py:f",
                        cases=[{"input": "a", "expected": "b"}])

    def test_malformed_spec(self):
        with pytest.raises(ValidationError, match="must be 'target:callable'"):
            SuiteConfig(name="x", scoring="custom", custom_scorer="nocolon",
                        cases=[{"input": "a", "expected": "b"}])


class TestEndToEnd:
    def test_run_suite_with_custom_scorer(self, tmp_path):
        from unittest.mock import patch
        from rift.config import ModelConfig
        from rift.runner import run_suite
        from rift.providers import Completion

        (tmp_path / "sc.py").write_text(
            "def score(output, expected):\n"
            "    return 1.0 if str(expected) in output else 0.0\n"
        )
        suite = SuiteConfig(name="t", scoring="custom", custom_scorer="./sc.py:score",
                            cases=[{"input": "q1", "expected": "foo"},
                                   {"input": "q2", "expected": "bar"}])
        suite._source_dir = tmp_path

        class Stub:
            async def complete(self, prompt, **kw):
                out = "foo!" if "q1" in prompt else "nope"
                return Completion(model="m", input_text=prompt, output_text=out,
                                  latency_ms=1, input_tokens=1, output_tokens=1,
                                  raw_response={})
            async def close(self):
                pass

        with patch("rift.runner._get_provider", lambda cfg: Stub()):
            res = asyncio.run(run_suite(
                suite, ModelConfig(provider="local", model="m"),
                cache_dir=str(tmp_path / "cache"), show_progress=False))
        assert res.scores == [1.0, 0.0]
        assert res.metadata["custom_scorer"] == "./sc.py:score"

    def test_run_suite_async_custom_scorer(self, tmp_path):
        from unittest.mock import patch
        from rift.config import ModelConfig
        from rift.runner import run_suite
        from rift.providers import Completion

        (tmp_path / "sc.py").write_text(
            "async def ascore(output, expected, context=None):\n"
            "    return 1.0 if output.strip() == str(expected) else 0.0\n"
        )
        suite = SuiteConfig(name="t", scoring="custom", custom_scorer="./sc.py:ascore",
                            cases=[{"input": "q", "expected": "ok"}])
        suite._source_dir = tmp_path

        class Stub:
            async def complete(self, prompt, **kw):
                return Completion(model="m", input_text=prompt, output_text="ok",
                                  latency_ms=1, input_tokens=1, output_tokens=1,
                                  raw_response={})
            async def close(self):
                pass

        with patch("rift.runner._get_provider", lambda cfg: Stub()):
            res = asyncio.run(run_suite(
                suite, ModelConfig(provider="local", model="m"),
                cache_dir=str(tmp_path / "cache"), show_progress=False))
        assert res.scores == [1.0]


class TestContextRotInteraction:
    def test_source_dir_survives_context_rot_expansion(self, tmp_path):
        # A custom-scored suite + context-rot must still resolve the scorer
        # file relative to the (original) suite directory.
        from rift.context_rot import expand_with_context_rot
        (tmp_path / "sc.py").write_text(
            "def score(output, expected):\n    return 1.0\n"
        )
        suite = SuiteConfig(name="t", scoring="custom", custom_scorer="./sc.py:score",
                            cases=[{"input": "q", "expected": "a"}])
        suite._source_dir = tmp_path
        expanded = expand_with_context_rot(suite)
        assert expanded._source_dir == tmp_path
        scorer = get_scorer("custom", custom_scorer=expanded.custom_scorer,
                            base_dir=expanded._source_dir)
        assert scorer.score("anything", "a") == 1.0


class TestBundledExample:
    def test_example_suite_loads_and_scores(self):
        from rift.config import load_suite
        s = load_suite("custom_scorer_example")
        assert s.scoring == "custom"
        scorer = get_scorer("custom", custom_scorer=s.custom_scorer,
                            base_dir=s._source_dir)
        assert scorer.score("red, blue and yellow", "red blue yellow") == 1.0
        assert scorer.score("only red", "red blue yellow") == pytest.approx(1 / 3)
