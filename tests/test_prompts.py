"""Tests for user-defined probe prompt overrides (rift.prompts + wiring)."""

from __future__ import annotations

import asyncio

import pytest

from rift import prompts
from rift.config import SuiteConfig


# --- registry validation ----------------------------------------------------

class TestValidateOverrides:
    def test_empty_is_noop(self):
        prompts.validate_overrides(None, None)
        prompts.validate_overrides({}, {})

    def test_unknown_prompt_key_rejected(self):
        with pytest.raises(ValueError, match="unknown prompt override 'nope'"):
            prompts.validate_overrides({"nope": "x {output}"}, None)

    def test_missing_placeholder_rejected(self):
        with pytest.raises(ValueError, match="missing required placeholder"):
            prompts.validate_overrides({"judge_rubric": "no placeholders here"}, None)

    def test_partial_placeholders_rejected(self):
        # has {question} but drops {output} and {target_block}
        with pytest.raises(ValueError, match=r"\{output\}|\{target_block\}"):
            prompts.validate_overrides({"judge_rubric": "{question} only"}, None)

    def test_valid_override_passes(self):
        prompts.validate_overrides(
            {"judge_rubric": "{question} {target_block} {output}"}, None
        )

    def test_empty_string_rejected(self):
        with pytest.raises(ValueError, match="non-empty string"):
            prompts.validate_overrides({"judge_rubric": "   "}, None)

    def test_cue_without_target_rejected(self):
        with pytest.raises(ValueError, match=r"\{target\}"):
            prompts.validate_overrides(None, {"authority": "no placeholder"})

    def test_cue_with_target_passes(self):
        prompts.validate_overrides(None, {"authority": "X says {target}"})

    def test_format_instruction_has_no_required_placeholders(self):
        # faithfulness_format_instruction is appended verbatim.
        prompts.validate_overrides({"faithfulness_format_instruction": "Answer please"}, None)

    def test_unknown_placeholder_rejected(self):
        # Has all required placeholders but an extra {foo} that .format() can't fill.
        with pytest.raises(ValueError, match="undefined placeholder"):
            prompts.validate_overrides(
                {"judge_rubric": "{question} {target_block} {output} {foo}"}, None
            )

    def test_unescaped_brace_rejected(self):
        # A JSON example with single (unescaped) braces would KeyError at format().
        with pytest.raises(ValueError, match="undefined placeholder|valid format"):
            prompts.validate_overrides(
                {"judge_rubric": '{question} {target_block} {output} {"k": 1}'}, None
            )

    def test_default_templates_self_validate(self):
        # The committed defaults (with escaped {{ }} JSON examples) must pass.
        for key in prompts.PROMPT_REGISTRY:
            prompts.validate_overrides({key: prompts._default_for(key)}, None)

    def test_escaped_braces_in_custom_template_pass(self):
        prompts.validate_overrides(
            {"judge_rubric": '{question} {target_block} {output} {{"score": 1}}'}, None
        )

    def test_cue_with_extra_placeholder_rejected(self):
        with pytest.raises(ValueError, match="undefined placeholder"):
            prompts.validate_overrides(None, {"authority": "{target} and {bogus}"})


class TestResolve:
    def test_resolve_returns_override(self):
        assert prompts.resolve("judge_rubric", {"judge_rubric": "X"}) == "X"

    def test_resolve_falls_back_to_default(self):
        default = prompts.resolve("judge_rubric", None)
        assert "{output}" in default  # the committed default

    def test_resolve_cues_defaults(self):
        assert set(prompts.resolve_cues(None)) == {"suggested", "authority", "consensus"}

    def test_resolve_cues_override_and_extend(self):
        merged = prompts.resolve_cues({"authority": "A {target}", "novel": "N {target}"})
        assert merged["authority"] == "A {target}"   # overridden
        assert merged["novel"] == "N {target}"        # extended
        assert "suggested" in merged                  # default kept

    def test_overridden_keys(self):
        keys = prompts.overridden_keys({"judge_rubric": "x"},
                                       {"authority": "y {target}"})
        assert keys == ["judge_rubric", "cue:authority"]

    def test_every_registry_key_has_a_default(self):
        for key in prompts.PROMPT_REGISTRY:
            assert isinstance(prompts._default_for(key), str)


# --- SuiteConfig integration -------------------------------------------------

class TestSuiteConfigOverrides:
    def test_valid_prompts_and_cues_parse(self):
        cfg = SuiteConfig(
            name="x", scoring="llm_judge",
            prompts={"judge_rubric": "{question} {target_block} {output}"},
            cues={"authority": "Counsel says {target}"},
            cases=[{"input": "a", "expected": "b"}],
        )
        assert "judge_rubric" in cfg.prompts
        assert cfg.cues["authority"] == "Counsel says {target}"

    def test_bad_override_raises_at_load(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SuiteConfig(name="x", scoring="llm_judge",
                        prompts={"judge_rubric": "missing output"},
                        cases=[{"input": "a", "expected": "b"}])

    def test_defaults_empty(self):
        cfg = SuiteConfig(name="x", cases=[{"input": "a", "expected": "b"}])
        assert cfg.prompts == {} and cfg.cues is None


# --- judge scorers honor the override ---------------------------------------

class _StubProvider:
    def __init__(self, payload):
        self.payload = payload
        self.prompts_seen = []

    async def complete(self, prompt, **kw):
        from rift.providers import Completion
        self.prompts_seen.append(prompt)
        return Completion(model="stub", input_text=prompt, output_text=self.payload,
                          latency_ms=1.0, input_tokens=1, output_tokens=1,
                          raw_response={})

    async def close(self):
        pass


class TestLLMJudgeOverride:
    def test_custom_rubric_used_and_changes_cache_key(self, tmp_path):
        from rift.scoring.llm_judge import LLMJudgeScorer
        custom = "CUSTOM RUBRIC {question} {target_block} {output}"
        stub = _StubProvider('{"score": 1.0, "reasoning": "ok"}')
        s = LLMJudgeScorer(judge_model="x", provider_factory=lambda m: stub,
                           cache_dir=str(tmp_path), prompt_template=custom)
        score = asyncio.run(s.ascore("out", "exp", context="q"))
        assert score == 1.0
        assert "CUSTOM RUBRIC" in stub.prompts_seen[0]

        # Default-prompt scorer hits a different cache key (no cross-reuse).
        from rift.scoring.llm_judge import _build_judge_prompt
        default_key = s._cache_key(_build_judge_prompt("q", "out", "exp"))
        custom_key = s._cache_key(_build_judge_prompt("q", "out", "exp", template=custom))
        assert default_key != custom_key


class TestFaithfulnessJudgeOverride:
    def test_custom_articulation_prompt_used(self, tmp_path):
        from rift.scoring.faithfulness_judge import FaithfulnessJudge
        custom = "CUSTOM ARTIC {cue} {reasoning} {answer} {target}"
        stub = _StubProvider('{"acknowledged": true, "reasoning": "x"}')
        j = FaithfulnessJudge(judge_model="x", provider_factory=lambda m: stub,
                              cache_dir=str(tmp_path), prompt_template=custom)
        ack = asyncio.run(j.acknowledged("q", "cue", "r", "a", "t"))
        assert ack is True
        assert "CUSTOM ARTIC" in stub.prompts_seen[0]


# --- faithfulness builders honor overrides ----------------------------------

def _base_suite():
    return SuiteConfig(name="t", scoring="exact_match",
                       cases=[{"input": "2+2?", "expected": "4"}])


class TestFaithfulnessBuilderOverrides:
    def test_custom_format_instruction_and_cue(self):
        from rift.faithfulness import build_faithfulness_suite
        derived = build_faithfulness_suite(
            _base_suite(), {0: "5"},
            cue_templates={"authority": "Counsel insists {target}"},
            format_instruction="FINISH WITH Answer: X",
        )
        inputs = [c.input for c in derived.cases]
        assert any("FINISH WITH Answer: X" in t for t in inputs)
        assert any("Counsel insists 5" in t for t in inputs)

    def test_custom_wrong_answer_prompt(self):
        from rift.faithfulness import build_wrong_answer_suite
        ws = build_wrong_answer_suite(
            _base_suite(), wrong_answer_prompt="GIVE WRONG: {question}"
        )
        assert ws.cases[0].input.startswith("GIVE WRONG: 2+2?")

    def test_compute_faithfulness_judges_against_override_cue_text(self):
        # Regression: compute_faithfulness must reconstruct the cue text from
        # the SAME (possibly overridden/extended) templates the suite used, not
        # the global CUES default. Otherwise a new cue's judge {cue} is empty.
        from rift.faithfulness import compute_faithfulness

        class C:
            def __init__(self, o, e, t):
                self.output = o
                self.expected = e
                self.tags = t
                self.input = "q"

        class R:
            def __init__(self, m, c):
                self.model = m
                self.cases = c

        class _ExactScorer:
            def score(self, output, expected):
                from rift.faithfulness import _parse_answer
                _, a = _parse_answer(output)
                return 1.0 if a.strip() == str(expected).strip() else 0.0

        # One control-correct case + a swayed variant on a NEW cue "novel".
        run = R("m", [
            C("Answer: 4", "4", ["faithfulness:control", "origin:0"]),
            C("Answer: 5", "4", ["faithfulness:cue=novel", "origin:0"]),
        ])
        seen_cue_text = {}

        def ack(question, cue_text, reasoning, answer, target):
            seen_cue_text["v"] = cue_text
            return False

        templates = {"novel": "NOVEL-CUE the answer is {target}"}
        res = compute_faithfulness(run, _ExactScorer(), ack, {0: "5"},
                                   cue_templates=templates)
        # The judge must have seen the real (new) cue text, not "".
        assert seen_cue_text["v"] == "NOVEL-CUE the answer is 5"
        assert res.per_case[0] == 0.0  # swayed + not acknowledged -> unfaithful

    def test_custom_cot_templates(self):
        from rift.faithfulness import build_cot_perturbation_suite

        class C:
            def __init__(self, o, e, t):
                self.output = o
                self.expected = e
                self.tags = t

        class R:
            def __init__(self, m, c):
                self.model = m
                self.cases = c

        class _ExactScorer:
            def score(self, output, expected):
                from rift.faithfulness import _parse_answer
                _, a = _parse_answer(output)
                return 1.0 if a.strip() == str(expected).strip() else 0.0

        ctrl = R("m", [C("reason line one\nreason line two\nAnswer: 4", "4",
                         ["faithfulness:control", "origin:0"])])
        pert, _ = build_cot_perturbation_suite(
            _base_suite(), ctrl, _ExactScorer(),
            early_template="EARLY {question} {partial}",
            mistake_template="MISTAKE {question} {reasoning}",
        )
        inputs = [c.input for c in pert.cases]
        assert any(t.startswith("EARLY ") for t in inputs)
        assert any(t.startswith("MISTAKE ") for t in inputs)


# --- runner disclosure -------------------------------------------------------

class TestRunnerDisclosure:
    def test_custom_prompts_stamped_into_metadata(self, tmp_path):
        from unittest.mock import patch
        from rift.config import ModelConfig
        from rift.runner import run_suite
        from rift.providers import Completion

        suite = SuiteConfig(
            name="sem", scoring="llm_judge",
            prompts={"judge_rubric": "R {question} {target_block} {output}"},
            cases=[{"input": "q", "expected": "a"}],
        )

        class StubProvider:
            async def complete(self, prompt, **kw):
                return Completion(model="m", input_text=prompt, output_text="a",
                                  latency_ms=1, input_tokens=1, output_tokens=1,
                                  raw_response={})
            async def close(self): pass

        class StubJudgeProv:
            async def complete(self, prompt, **kw):
                return Completion(model="j", input_text=prompt,
                                  output_text='{"score": 1.0, "reasoning": "ok"}',
                                  latency_ms=1, input_tokens=1, output_tokens=1,
                                  raw_response={})
            async def close(self): pass

        with patch("rift.runner._get_provider", lambda cfg: StubProvider()), \
             patch("rift.scoring.llm_judge._default_provider_factory",
                   lambda m: StubJudgeProv()):
            res = asyncio.run(run_suite(suite, ModelConfig(provider="local", model="m"),
                                        cache_dir=str(tmp_path), show_progress=False))
        assert res.metadata.get("custom_prompts") == ["judge_rubric"]

    def test_no_custom_prompts_no_metadata_key(self, tmp_path):
        from unittest.mock import patch
        from rift.config import ModelConfig
        from rift.runner import run_suite
        from rift.providers import Completion

        suite = SuiteConfig(name="x", scoring="exact_match",
                            cases=[{"input": "q", "expected": "a"}])

        class StubProvider:
            async def complete(self, prompt, **kw):
                return Completion(model="m", input_text=prompt, output_text="a",
                                  latency_ms=1, input_tokens=1, output_tokens=1,
                                  raw_response={})
            async def close(self): pass

        with patch("rift.runner._get_provider", lambda cfg: StubProvider()):
            res = asyncio.run(run_suite(suite, ModelConfig(provider="local", model="m"),
                                        cache_dir=str(tmp_path), show_progress=False))
        assert "custom_prompts" not in res.metadata
