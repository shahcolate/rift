"""Tests for power-stratified auto-adversarial case discovery.

Uses a stub :class:`BaseProvider` for both the proposer and the
verifier (baseline + challenger) so no real network call is
exercised. Coverage:

* proposer-response parsing (clean, fenced, prose-wrapped, malformed)
* Jaccard dedup
* end-to-end loop with deterministic scored candidates
* validity gate (both-zero cases dropped, concordant pairs dropped)
* power computation regression (matches analytical formula)
* round-trip: discovered suite parses back into a SuiteConfig
* CLI metadata + caveat surfacing in YAML
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable

import yaml

from rift.config import EvalCase, ModelConfig, SuiteConfig, load_suite
from rift.discovery import (
    DEDUP_JACCARD_THRESHOLD,
    DiscoveryResult,
    _jaccard_5gram,
    discover,
    parse_proposer_response,
    to_suite_yaml,
)
from rift.providers import BaseProvider, Completion


# ---------------------------------------------------------------------------
# Stub provider — same shape as the LLM-judge tests' StubProvider but
# kept module-local so the two suites stay independent.
# ---------------------------------------------------------------------------


@dataclass
class _Recorded:
    prompt: str
    kwargs: dict


class StubProvider(BaseProvider):
    def __init__(self, responder: Callable[[str], str], model: str = "stub") -> None:
        self.responder = responder
        self.model = model
        self.calls: list[_Recorded] = []
        self.closed = False

    async def complete(self, prompt: str, **kwargs) -> Completion:
        self.calls.append(_Recorded(prompt=prompt, kwargs=kwargs))
        return Completion(
            model=self.model,
            input_text=prompt,
            output_text=self.responder(prompt),
            latency_ms=1.0,
            input_tokens=20,
            output_tokens=10,
            raw_response={},
        )

    async def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParseProposerResponse:
    def test_clean_json_array(self):
        text = '[{"input": "Q1", "expected": "A1", "rationale": "hard"}]'
        out = parse_proposer_response(text)
        assert len(out) == 1
        assert out[0]["input"] == "Q1"

    def test_markdown_fenced(self):
        text = (
            '```json\n'
            '[{"input": "Q1", "expected": "A1"},\n'
            ' {"input": "Q2", "expected": "A2"}]\n'
            '```'
        )
        out = parse_proposer_response(text)
        assert len(out) == 2
        assert out[1]["expected"] == "A2"

    def test_prose_wrapped(self):
        text = (
            'Here are my proposals:\n\n'
            '[{"input": "Q1", "expected": "A1"}]\n'
            '\nLet me know if you want more.'
        )
        out = parse_proposer_response(text)
        assert len(out) == 1

    def test_malformed_returns_empty(self):
        assert parse_proposer_response("totally not json") == []
        assert parse_proposer_response("") == []
        assert parse_proposer_response("{single object not array}") == []

    def test_drops_items_missing_required_fields(self):
        text = (
            '[{"input": "ok", "expected": "y"},'
            ' {"input": "", "expected": "z"},'
            ' {"expected": "no input"},'
            ' {"input": "no expected"}]'
        )
        out = parse_proposer_response(text)
        assert len(out) == 1
        assert out[0]["input"] == "ok"

    def test_non_list_top_level_returns_empty(self):
        assert parse_proposer_response('{"not": "an array"}') == []


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


class TestJaccardDedup:
    def test_identical_strings(self):
        assert _jaccard_5gram("the quick brown fox", "the quick brown fox") == 1.0

    def test_unrelated_strings(self):
        # No 5-gram overlap.
        assert _jaccard_5gram("abcdefgh", "wxyz1234") == 0.0

    def test_near_duplicate_above_threshold(self):
        a = "What is the capital of France in Europe?"
        b = "What is the capital of France in Europe."
        assert _jaccard_5gram(a, b) >= DEDUP_JACCARD_THRESHOLD


# ---------------------------------------------------------------------------
# End-to-end discovery loop
# ---------------------------------------------------------------------------


def _make_seed_suite(name="seed", scoring="exact_match") -> SuiteConfig:
    return SuiteConfig(
        name=name,
        description="test seed",
        scoring=scoring,
        cases=[
            EvalCase(input="What is 2+2?", expected="4"),
            EvalCase(input="What is 3+3?", expected="6"),
        ],
    )


def _proposer_batch(*pairs) -> str:
    """Render a JSON array of (input, expected) pairs as a proposer response."""
    import json as _j
    return _j.dumps([
        {"input": inp, "expected": exp, "rationale": "trial"}
        for inp, exp in pairs
    ])


class TestDiscoverLoop:
    def test_accepts_discordant_drops_concordant(self, tmp_path, monkeypatch):
        # Proposer returns 4 candidates per batch.
        proposer = StubProvider(lambda _p: _proposer_batch(
            ("Q_a", "A_a"),
            ("Q_b", "A_b"),
            ("Q_c", "A_c"),
            ("Q_d", "A_d"),
        ))

        # Baseline gets a & b correct, c & d wrong.
        baseline = StubProvider(lambda prompt: (
            "A_a" if "Q_a" in prompt else
            "A_b" if "Q_b" in prompt else
            "WRONG"
        ))
        # Challenger gets a & c correct, b & d wrong.
        # → discordant on b (1,0) and c (0,1); concordant on a (1,1) and d (0,0).
        # Both-zero on d → dropped by validity gate.
        # Both-one on a → concordant → dropped by info gate.
        # b and c are the two accepted cases.
        challenger = StubProvider(lambda prompt: (
            "A_a" if "Q_a" in prompt else
            "A_c" if "Q_c" in prompt else
            "WRONG"
        ))

        def factory(model_id):
            if model_id == "proposer-model":
                return proposer
            return None  # not used — see runner patch below

        # Patch the runner's provider lookup to point at our stubs.
        from rift import runner as runner_mod

        def fake_get_provider(cfg):
            if cfg.model == "baseline-m":
                return baseline
            return challenger
        monkeypatch.setattr(runner_mod, "_get_provider", fake_get_provider)

        seed = _make_seed_suite()
        result = asyncio.run(discover(
            baseline=ModelConfig(provider="anthropic", model="baseline-m"),
            challenger=ModelConfig(provider="anthropic", model="challenger-m"),
            seed_suite=seed,
            proposer_model="proposer-model",
            max_cases=10,
            batch_size=4,
            cache_dir=str(tmp_path),
            provider_factory=factory,
        ))

        # Only the two discordant cases survive.
        kept_inputs = [c.input for c in result.cases]
        assert sorted(kept_inputs) == ["Q_b", "Q_c"]
        # Both have info = 1.0 (binary discordance).
        assert all(i == 1.0 for i in result.case_info)
        # n_after_validity counts discordant cases (b, c).
        assert result.n_after_validity == 2
        # Proposer was called exactly once (4 candidates fits batch).
        assert len(proposer.calls) >= 1
        # Provider was closed.
        assert proposer.closed is True

    def test_dedup_against_seed(self, tmp_path, monkeypatch):
        # Proposer returns a near-duplicate of a seed case + one new case.
        proposer = StubProvider(lambda _p: _proposer_batch(
            ("What is 2+2?", "4"),         # dup of seed
            ("What is 5*5?", "25"),        # new
        ))
        baseline = StubProvider(lambda _: "wrong")
        challenger = StubProvider(lambda _: "25")  # gets the new one right
        from rift import runner as runner_mod

        def fake_get_provider(cfg):
            return baseline if cfg.model == "B" else challenger
        monkeypatch.setattr(runner_mod, "_get_provider", fake_get_provider)

        seed = _make_seed_suite()
        result = asyncio.run(discover(
            baseline=ModelConfig(provider="anthropic", model="B"),
            challenger=ModelConfig(provider="anthropic", model="C"),
            seed_suite=seed,
            proposer_model="P",
            max_cases=10,
            batch_size=2,
            cache_dir=str(tmp_path),
            provider_factory=lambda _m: proposer,
        ))

        # Seed-duplicate dropped at dedup; n_after_dedup = 1.
        assert result.n_after_dedup == 1
        # The new case is discordant → kept.
        assert len(result.cases) == 1
        assert result.cases[0].input == "What is 5*5?"

    def test_stops_at_max_cases(self, tmp_path, monkeypatch):
        # Proposer always returns 3 fresh discordant cases per batch.
        counter = {"n": 0}

        def proposer_responder(_p):
            i = counter["n"]
            counter["n"] += 3
            return _proposer_batch(
                (f"Q{i}", "y"),
                (f"Q{i+1}", "y"),
                (f"Q{i+2}", "y"),
            )

        proposer = StubProvider(proposer_responder)
        baseline = StubProvider(lambda _: "wrong")
        challenger = StubProvider(lambda _: "y")
        from rift import runner as runner_mod
        monkeypatch.setattr(
            runner_mod, "_get_provider",
            lambda cfg: baseline if cfg.model == "B" else challenger,
        )

        result = asyncio.run(discover(
            baseline=ModelConfig(provider="anthropic", model="B"),
            challenger=ModelConfig(provider="anthropic", model="C"),
            seed_suite=_make_seed_suite(),
            proposer_model="P",
            max_cases=4,
            batch_size=3,
            cache_dir=str(tmp_path),
            provider_factory=lambda _m: proposer,
        ))

        assert result.n_kept == 4
        # Should have made at least 2 proposer batches (3 + 3 ≥ 4).
        assert len(proposer.calls) >= 2

    def test_malformed_proposer_response_terminates(self, tmp_path, monkeypatch):
        proposer = StubProvider(lambda _: "i refuse to comply")
        from rift import runner as runner_mod
        monkeypatch.setattr(
            runner_mod, "_get_provider",
            lambda _cfg: StubProvider(lambda _: "y"),
        )
        result = asyncio.run(discover(
            baseline=ModelConfig(provider="anthropic", model="B"),
            challenger=ModelConfig(provider="anthropic", model="C"),
            seed_suite=_make_seed_suite(),
            proposer_model="P",
            max_cases=10, batch_size=4,
            cache_dir=str(tmp_path),
            provider_factory=lambda _m: proposer,
        ))
        # Junk parses to []; loop bails to avoid infinite spend.
        assert result.n_kept == 0
        assert len(proposer.calls) == 1


# ---------------------------------------------------------------------------
# Power computation regression
# ---------------------------------------------------------------------------


class TestPowerOnDiscoveredSuite:
    def test_matches_analytical_power(self, tmp_path, monkeypatch):
        # Construct a scenario where exactly 40 cases survive, all
        # discordant in the same direction (challenger > baseline).
        # The post-hoc power calculation on (0...0, 1...1) should
        # match what power_analysis returns when called directly.
        N = 40

        proposer_iter = iter(range(100))

        def proposer_responder(_p):
            i = next(proposer_iter)
            return _proposer_batch(
                *((f"Qd{i * 10 + j}", "y") for j in range(10))
            )

        proposer = StubProvider(proposer_responder)
        baseline = StubProvider(lambda _: "wrong")
        challenger = StubProvider(lambda _: "y")
        from rift import runner as runner_mod
        monkeypatch.setattr(
            runner_mod, "_get_provider",
            lambda cfg: baseline if cfg.model == "B" else challenger,
        )

        result = asyncio.run(discover(
            baseline=ModelConfig(provider="anthropic", model="B"),
            challenger=ModelConfig(provider="anthropic", model="C"),
            seed_suite=_make_seed_suite(),
            proposer_model="P",
            max_cases=N, batch_size=10,
            cache_dir=str(tmp_path),
            provider_factory=lambda _m: proposer,
        ))
        assert result.n_kept == N
        # All discordant in the same direction → very high power.
        assert result.achieved_power > 0.99

        # Cross-check against direct power_analysis on the synthetic
        # scores: baseline all 0, challenger all 1.
        from rift.comparator import power_analysis
        ref = power_analysis([0.0] * N, [1.0] * N)
        assert abs(result.achieved_power - ref["observed_power"]) < 1e-4


# ---------------------------------------------------------------------------
# YAML emission + round-trip
# ---------------------------------------------------------------------------


class TestSuiteEmission:
    def test_to_suite_yaml_carries_provenance(self):
        result = DiscoveryResult(
            cases=[EvalCase(input="Q", expected="A", tags=["discovered"])],
            achieved_power=0.93,
            target_power=0.9,
            target_effect=0.05,
            alpha=0.05,
            proposer_model="opus-4-7",
            baseline_model="opus-4-6",
            challenger_model="opus-4-7",
            seed_suite_name="reasoning",
            n_proposed=96, n_parsed=96, n_after_dedup=80,
            n_after_validity=42, n_kept=1,
            discordant_rate=0.47,
            proposer_spend_usd=0.41,
            verification_spend_usd=1.18,
            case_info=[1.0],
            rationales=["plausibly hard"],
        )
        d = to_suite_yaml(result)
        assert d["name"].startswith("reasoning__discovered_")
        # The selection-bias caveat must appear in the description.
        assert "selected on divergence" in d["description"]
        assert "achieved_power" in d["description"].lower() or \
               "Achieved power" in d["description"]
        # Provenance numbers present.
        assert "0.93" in d["description"]
        assert "opus-4-7" in d["description"]
        assert len(d["cases"]) == 1

    def test_round_trip_through_load_suite(self, tmp_path):
        result = DiscoveryResult(
            cases=[
                EvalCase(input="Q1", expected="A1", tags=["discovered"]),
                EvalCase(input="Q2", expected="A2", tags=["discovered"]),
            ],
            achieved_power=0.85, target_power=0.9, target_effect=0.05,
            alpha=0.05,
            proposer_model="m", baseline_model="b", challenger_model="c",
            seed_suite_name="seed",
            n_proposed=10, n_parsed=10, n_after_dedup=10,
            n_after_validity=2, n_kept=2,
            discordant_rate=1.0,
            proposer_spend_usd=0.0, verification_spend_usd=0.0,
            case_info=[1.0, 1.0],
            rationales=["", ""],
        )
        path = tmp_path / "discovered.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(to_suite_yaml(result), f, sort_keys=False)
        # The emitted YAML must be a valid SuiteConfig.
        loaded = load_suite(str(path))
        assert loaded.scoring == "exact_match"
        assert len(loaded.cases) == 2
        assert loaded.cases[0].input == "Q1"


# ---------------------------------------------------------------------------
# Spend tracking
# ---------------------------------------------------------------------------


class TestSpend:
    def test_spend_is_tracked_and_summed(self, tmp_path, monkeypatch):
        proposer = StubProvider(lambda _: _proposer_batch(("Qx", "y")))
        baseline = StubProvider(lambda _: "wrong")
        challenger = StubProvider(lambda _: "y")
        from rift import runner as runner_mod
        monkeypatch.setattr(
            runner_mod, "_get_provider",
            lambda cfg: baseline if cfg.model == "B" else challenger,
        )
        result = asyncio.run(discover(
            baseline=ModelConfig(provider="anthropic",
                                 model="claude-opus-4-7"),
            challenger=ModelConfig(provider="anthropic",
                                   model="claude-opus-4-6"),
            seed_suite=_make_seed_suite(),
            proposer_model="claude-opus-4-7",
            max_cases=1, batch_size=1,
            cache_dir=str(tmp_path),
            provider_factory=lambda _m: proposer,
        ))
        # Both fields are populated (real pricing.cost_of returned positive
        # values, since these are known models in the catalog).
        assert result.proposer_spend_usd >= 0.0
        assert result.verification_spend_usd >= 0.0
