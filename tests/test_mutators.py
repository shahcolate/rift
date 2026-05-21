"""Unit tests for ``rift.mutators``.

The mutator module is the shared LLM-call scaffolding for bisect's
fixing-mutation loop and attribute's template-sensitivity probe.
Tests use a fake provider so no API key is needed.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from rift.mutators import (
    FAMILY_DEFINITIONS,
    MUTATION_DEDUP_JACCARD,
    MUTATION_FAMILIES,
    Mutation,
    _build_mutator_prompt,
    edit_distance,
    propose_mutations,
)
from rift.providers import BaseProvider, Completion


# ---------------------------------------------------------------------------
# Fake provider used by every async test
# ---------------------------------------------------------------------------


class FakeProvider(BaseProvider):
    """Records every call and returns a scripted ``output_text``.

    ``script`` is consumed left-to-right; each call returns the next
    string from the list. If the script is exhausted, the last entry
    is reused (so a test can post a single canned response that
    every family call returns).
    """

    def __init__(self, script: list[str]):
        if not script:
            raise ValueError("FakeProvider requires at least one scripted response")
        self.script = list(script)
        self.calls: list[str] = []

    async def complete(self, prompt: str, **kwargs) -> Completion:
        self.calls.append(prompt)
        text = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if not self.script:
            self.script = [text]  # preserve for any further calls
        return Completion(
            model="fake-model",
            input_text=prompt,
            output_text=text,
            latency_ms=1.0,
            input_tokens=10,
            output_tokens=20,
            raw_response={},
        )

    async def close(self) -> None:
        pass


def _array(*items: dict) -> str:
    return json.dumps(list(items))


# ---------------------------------------------------------------------------
# edit_distance
# ---------------------------------------------------------------------------


class TestEditDistance:
    @pytest.mark.parametrize(
        "a,b,expected",
        [
            ("", "", 0),
            ("abc", "abc", 0),
            ("kitten", "sitting", 3),
            ("flaw", "lawn", 2),
            ("", "abc", 3),
            ("abc", "", 3),
            # Cross-checked against Wikipedia's Levenshtein examples.
            ("Return ONLY the number.", "Output: <number>", 15),
        ],
    )
    def test_known_cases(self, a, b, expected):
        assert edit_distance(a, b) == expected

    def test_symmetry(self):
        assert edit_distance("hello", "world") == edit_distance("world", "hello")

    def test_one_char_change(self):
        assert edit_distance("Output: <num>", "output: <num>") == 1


# ---------------------------------------------------------------------------
# Mutation.build
# ---------------------------------------------------------------------------


class TestMutationBuild:
    def test_computes_edit_distance_and_char_delta(self):
        m = Mutation.build(
            family="paraphrase",
            seed_input="What is 2 + 2?",
            mutated_input="Compute 2 plus 2.",
            rationale="rephrased",
        )
        assert m.family == "paraphrase"
        assert m.edit_distance == edit_distance(
            "What is 2 + 2?", "Compute 2 plus 2."
        )
        assert m.char_delta == len("Compute 2 plus 2.") - len("What is 2 + 2?")

    def test_to_dict_roundtrips(self):
        m = Mutation.build(
            family="typo_fix",
            seed_input="What is the capitol of France?",
            mutated_input="What is the capital of France?",
            rationale="typo: capitol → capital",
        )
        d = m.to_dict()
        m2 = Mutation(**d)
        assert m == m2


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


class TestBuildMutatorPrompt:
    def test_renders_family_definition(self):
        prompt = _build_mutator_prompt(
            original="What is 2 + 2?",
            expected_or_baseline="4",
            challenger_output="5",
            family="typo_fix",
            n=3,
        )
        assert "What is 2 + 2?" in prompt
        assert "Generate 3 minimally-edited rewrites" in prompt
        assert FAMILY_DEFINITIONS["typo_fix"] in prompt
        assert "5" in prompt  # challenger wrong output

    def test_unknown_family_raises(self):
        with pytest.raises(ValueError, match="unknown mutation family"):
            _build_mutator_prompt(
                original="x",
                expected_or_baseline="y",
                challenger_output="z",
                family="not-a-real-family",
                n=1,
            )

    def test_truncates_long_challenger_output(self):
        long_output = "X" * 10_000
        prompt = _build_mutator_prompt(
            original="q",
            expected_or_baseline="a",
            challenger_output=long_output,
            family="paraphrase",
            n=1,
        )
        # Only the first 4KB of challenger output should appear.
        assert prompt.count("X") == 4096


# ---------------------------------------------------------------------------
# propose_mutations: end-to-end (no real network)
# ---------------------------------------------------------------------------


class TestProposeMutations:
    def test_unknown_family_raises_before_any_provider_call(self):
        provider = FakeProvider([_array({"input": "x"})])

        async def go():
            await propose_mutations(
                original="q",
                expected="a",
                challenger_output="wrong",
                families=["not-real"],
                n_per_family=2,
                provider=provider,
                model_id="fake",
            )

        with pytest.raises(ValueError, match="unknown mutation family"):
            asyncio.run(go())
        assert provider.calls == []

    def test_dedups_against_original(self):
        # Three candidates, two of which are byte-identical to the
        # original (or above 0.95 Jaccard). Only the non-duplicate
        # should make it through.
        original = "Return ONLY the integer answer to: What is 17 * 23?"
        response = _array(
            {"input": original, "rationale": "no change"},
            {"input": original + " ", "rationale": "trailing space"},
            {"input": "Return only the integer answer to: What is 17*23?",
             "rationale": "case + spacing tweak"},
        )
        provider = FakeProvider([response])

        result, cost = asyncio.run(propose_mutations(
            original=original,
            expected=391,
            challenger_output="17 * 23 = 391, so the answer is 391.",
            families=["paraphrase"],
            n_per_family=3,
            provider=provider,
            model_id="fake",
        ))
        # The first two are duplicates of the original; only the
        # rewritten case-and-spacing variant survives.
        assert len(result) == 1
        assert "Return only" in result[0].mutated_input

    def test_caches_results(self, tmp_path: Path):
        response = _array({"input": "What is seventeen times twenty-three?",
                           "rationale": "spelled out"})
        provider = FakeProvider([response])

        kwargs = dict(
            original="What is 17*23?",
            expected=391,
            challenger_output="391",
            families=["paraphrase"],
            n_per_family=1,
            provider=provider,
            model_id="fake",
            cache_dir=tmp_path,
        )

        first, _ = asyncio.run(propose_mutations(**kwargs))
        second, _ = asyncio.run(propose_mutations(**kwargs))
        assert first == second
        assert len(provider.calls) == 1  # second call was a cache hit

        # The cache file should exist under the expected subdir.
        cache_files = list((tmp_path / "mutations").glob("*.json"))
        assert len(cache_files) == 1
        cached = json.loads(cache_files[0].read_text())
        assert cached["family"] == "paraphrase"
        assert cached["model_id"] == "fake"
        assert len(cached["mutations"]) == 1

    def test_runs_one_provider_call_per_family(self):
        # Each family should result in exactly one provider call.
        response_a = _array({"input": "rewrite-A"})
        response_b = _array({"input": "rewrite-B"})
        response_c = _array({"input": "rewrite-C"})
        provider = FakeProvider([response_a, response_b, response_c])

        result, _ = asyncio.run(propose_mutations(
            original="original prompt that is long enough for 5-grams",
            expected="x",
            challenger_output="wrong",
            families=["paraphrase", "typo_fix", "format_instruction"],
            n_per_family=1,
            provider=provider,
            model_id="fake",
        ))

        assert len(provider.calls) == 3
        # Each call's prompt should mention its own family.
        seen_families = []
        for call in provider.calls:
            for fam in MUTATION_FAMILIES:
                if FAMILY_DEFINITIONS[fam] in call:
                    seen_families.append(fam)
        assert set(seen_families) == {
            "paraphrase", "typo_fix", "format_instruction",
        }
        # The mutations include one from each family.
        assert {m.family for m in result} == {
            "paraphrase", "typo_fix", "format_instruction",
        }

    def test_cross_family_dedup(self):
        # Two families both emit the same rewrite. Only one should
        # survive after cross-family dedup.
        same_rewrite = "What is the value of 17 multiplied by 23?"
        response_a = _array({"input": same_rewrite, "rationale": "fam-a"})
        response_b = _array({"input": same_rewrite, "rationale": "fam-b"})
        provider = FakeProvider([response_a, response_b])

        result, _ = asyncio.run(propose_mutations(
            original="What is 17*23?",
            expected=391,
            challenger_output="391",
            families=["paraphrase", "clarify_ambiguity"],
            n_per_family=1,
            provider=provider,
            model_id="fake",
        ))
        assert len(result) == 1
        # First family wins; later duplicates are dropped.
        assert result[0].family == "paraphrase"

    def test_empty_families_returns_empty_no_calls(self):
        provider = FakeProvider([_array({"input": "x"})])
        result, cost = asyncio.run(propose_mutations(
            original="q",
            expected="a",
            challenger_output="w",
            families=[],
            n_per_family=3,
            provider=provider,
            model_id="fake",
        ))
        assert result == []
        assert cost == 0.0
        assert provider.calls == []

    def test_structured_expected_does_not_crash_cache_key(self, tmp_path: Path):
        # ``expected`` can be a dict for extraction-style suites. The
        # cache key must serialise it deterministically without raising.
        response = _array({"input": "extract from: invoice 99"})
        provider = FakeProvider([response])

        result, _ = asyncio.run(propose_mutations(
            original="Extract fields from: invoice 99",
            expected={"invoice_number": "99", "total": 420.0},
            challenger_output='{"invoice_number": "99", "total": null}',
            families=["paraphrase"],
            n_per_family=1,
            provider=provider,
            model_id="fake",
            cache_dir=tmp_path,
        ))
        assert len(result) == 1
        assert (tmp_path / "mutations").exists()


# ---------------------------------------------------------------------------
# Dedup threshold sanity
# ---------------------------------------------------------------------------


class TestDedupThreshold:
    def test_threshold_value(self):
        # If this changes, both bisect and attribute callers should
        # be reviewed — the threshold affects how aggressively
        # near-duplicate mutations are dropped.
        assert MUTATION_DEDUP_JACCARD == 0.95

    def test_family_taxonomy_ordering(self):
        # The first three families are the "smallest-edit" tier; the
        # tie-break rule in bisect.py depends on this ordering.
        assert MUTATION_FAMILIES[:3] == (
            "typo_fix", "format_instruction", "clarify_ambiguity",
        )
        # And every family identifier has a matching definition.
        assert set(MUTATION_FAMILIES) == set(FAMILY_DEFINITIONS)
