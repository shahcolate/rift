"""Provider fingerprint capture + integrity surfacing.

A drift detector that caches on the request alone is blind to a silent
server-side weight swap behind a stable model alias. Capturing the
provider-reported fingerprint is the only signal that closes that hole, so
these tests pin both the capture and the integrity checks built on top of it.
"""

from __future__ import annotations

import asyncio

import pytest

from rift.config import EvalCase, ModelConfig, SuiteConfig
from rift.providers import Completion
from rift.reporter import print_fingerprint_report
from rift.runner import CaseResult, RunResult, run_suite


class _StubProvider:
    """Minimal provider returning a scripted fingerprint per call."""

    def __init__(self, fingerprints):
        self._fps = list(fingerprints)
        self.model = "stub-model"
        self._i = 0

    async def complete(self, prompt, **kwargs):
        fp = self._fps[self._i % len(self._fps)]
        self._i += 1
        return Completion(
            model="stub-model",
            input_text=prompt,
            output_text="4",
            latency_ms=1.0,
            input_tokens=1,
            output_tokens=1,
            raw_response={},
            provider_fingerprint=fp,
        )

    async def close(self):
        pass


def _suite(n=3):
    return SuiteConfig(
        name="s",
        scoring="exact_match",
        cases=[EvalCase(input=f"q{i}", expected="4") for i in range(n)],
    )


class TestCompletionSchema:
    def test_from_cache_tolerates_missing_field(self):
        # An old cache blob predates provider_fingerprint.
        old = {
            "model": "m", "input_text": "x", "output_text": "y",
            "latency_ms": 1.0, "input_tokens": 1, "output_tokens": 1,
            "raw_response": {},
        }
        c = Completion.from_cache(old)
        assert c.provider_fingerprint is None

    def test_from_cache_ignores_unknown_field(self):
        # A future cache blob carries a field this version doesn't know.
        data = {
            "model": "m", "input_text": "x", "output_text": "y",
            "latency_ms": 1.0, "input_tokens": 1, "output_tokens": 1,
            "raw_response": {}, "provider_fingerprint": "fp1",
            "some_future_field": 123,
        }
        c = Completion.from_cache(data)
        assert c.provider_fingerprint == "fp1"


class TestRunnerStamping:
    def test_single_fingerprint_stamped(self, tmp_path, monkeypatch):
        prov = _StubProvider(["fp-stable"])
        monkeypatch.setattr("rift.runner._get_provider", lambda cfg: prov)
        cfg = ModelConfig(provider="local", model="stub-model")
        run = asyncio.run(run_suite(_suite(), cfg, cache_dir=str(tmp_path),
                                    show_progress=False))
        assert run.metadata["fingerprints"] == ["fp-stable"]
        assert "fingerprint_rollout" not in run.metadata
        assert all(c.provider_fingerprint == "fp-stable" for c in run.cases)

    def test_rollout_detected(self, tmp_path, monkeypatch):
        # Two distinct fingerprints during one run = a mid-run rollout.
        prov = _StubProvider(["fp-old", "fp-new"])
        monkeypatch.setattr("rift.runner._get_provider", lambda cfg: prov)
        cfg = ModelConfig(provider="local", model="stub-model")
        run = asyncio.run(run_suite(_suite(n=4), cfg, cache_dir=str(tmp_path),
                                    show_progress=False))
        assert run.metadata["fingerprint_rollout"] is True
        assert len(run.metadata["fingerprints"]) == 2

    def test_no_fingerprint_no_metadata(self, tmp_path, monkeypatch):
        prov = _StubProvider([None])
        monkeypatch.setattr("rift.runner._get_provider", lambda cfg: prov)
        cfg = ModelConfig(provider="local", model="stub-model")
        run = asyncio.run(run_suite(_suite(), cfg, cache_dir=str(tmp_path),
                                    show_progress=False))
        assert "fingerprints" not in run.metadata


def _run_with_fp(model, fps):
    cases = [
        CaseResult(case_index=i, input_text="x", expected="4", output="4",
                   score=1.0, latency_ms=1.0, input_tokens=1, output_tokens=1,
                   provider_fingerprint=fp)
        for i, fp in enumerate(fps)
    ]
    return RunResult(model=model, suite_name="s", scoring_method="exact_match",
                     cases=cases)


class TestFingerprintReport:
    def test_clean_case_is_silent(self, capsys):
        b = _run_with_fp("claude-opus-4-7", ["fp-a", "fp-a"])
        c = _run_with_fp("claude-opus-4-8", ["fp-b", "fp-b"])
        flagged = print_fingerprint_report(b, c)
        assert flagged is False

    def test_alias_collision_flagged(self, capsys):
        # Different requested models, identical single served fingerprint.
        b = _run_with_fp("opus-4-8", ["fp-same", "fp-same"])
        c = _run_with_fp("claude-opus-4-8", ["fp-same", "fp-same"])
        flagged = print_fingerprint_report(b, c)
        assert flagged is True
        assert "SAME served fingerprint" in capsys.readouterr().out

    def test_rollout_flagged(self, capsys):
        b = _run_with_fp("m-a", ["fp1", "fp2"])
        c = _run_with_fp("m-b", ["fp3", "fp3"])
        flagged = print_fingerprint_report(b, c)
        assert flagged is True
        assert "multiple fingerprints" in capsys.readouterr().out

    def test_no_fingerprints_silent(self):
        b = _run_with_fp("m-a", [None, None])
        c = _run_with_fp("m-b", [None, None])
        assert print_fingerprint_report(b, c) is False

    def test_same_model_not_collision(self):
        # selftest compares a model to itself on purpose — same model string,
        # same fingerprint is expected, not a collision to flag.
        b = _run_with_fp("m", ["fp", "fp"])
        c = _run_with_fp("m", ["fp", "fp"])
        assert print_fingerprint_report(b, c) is False
