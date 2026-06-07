"""Replication (`trials>1`) and run-to-run variance decomposition.

The central methodological gap a single-trial paired test has is that it
assumes generation noise is zero. These tests pin the variance decomposition
math and the runner plumbing that re-samples each case into distinct cache
entries.
"""

from __future__ import annotations

import asyncio

import pytest

from rift.comparator import variance_components
from rift.config import EvalCase, ModelConfig, SuiteConfig
from rift.providers import Completion
from rift.runner import run_suite


class TestVarianceComponents:
    def test_no_within_variance_is_perfect_icc(self):
        # Every trial identical within a case, cases differ -> all signal.
        vc = variance_components([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
        assert vc["within_case_var"] == 0.0
        assert vc["icc"] == 1.0
        assert vc["noise_floor"] == 0.0

    def test_all_noise_is_low_icc(self):
        # Both cases have identical means (0.5) but wobble every trial:
        # zero between-case variance -> ICC collapses toward 0.
        vc = variance_components([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
        assert vc["between_case_var"] == 0.0
        assert vc["icc"] == 0.0
        assert vc["within_case_var"] > 0.0

    def test_noise_floor_shrinks_with_more_data(self):
        small = variance_components([[1.0, 0.0], [1.0, 0.0]])
        big = variance_components([[1.0, 0.0] * 8, [1.0, 0.0] * 8])
        assert big["noise_floor"] < small["noise_floor"]

    def test_single_trial_cases_have_no_measurable_noise(self):
        # One trial per case: within-case variance is unmeasurable -> 0.
        vc = variance_components([[1.0], [0.0], [1.0]])
        assert vc["within_case_var"] == 0.0
        assert vc["mean_trials"] == 1.0

    def test_empty_is_safe(self):
        vc = variance_components([])
        assert vc["n_cases"] == 0
        assert vc["icc"] == 1.0

    def test_mean_trials_reported(self):
        vc = variance_components([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
        assert vc["mean_trials"] == 3.0


class _CountingProvider:
    """Returns a fixed output, counting how many live completions happened."""

    def __init__(self, output="4"):
        self.calls = 0
        self.output = output
        self.model = "stub"

    async def complete(self, prompt, **kwargs):
        self.calls += 1
        return Completion(
            model="stub", input_text=prompt, output_text=self.output,
            latency_ms=1.0, input_tokens=2, output_tokens=1, raw_response={},
            provider_fingerprint="fp",
        )

    async def close(self):
        pass


def _suite(n=2):
    return SuiteConfig(
        name="s", scoring="exact_match",
        cases=[EvalCase(input=f"q{i}", expected="4") for i in range(n)],
    )


class TestRunnerTrials:
    def test_each_trial_queries_and_caches_separately(self, tmp_path, monkeypatch):
        prov = _CountingProvider()
        monkeypatch.setattr("rift.runner._get_provider", lambda cfg: prov)
        cfg = ModelConfig(provider="local", model="stub")
        run = asyncio.run(run_suite(_suite(n=2), cfg, cache_dir=str(tmp_path),
                                    show_progress=False, trials=3))
        # 2 cases × 3 trials = 6 live completions, 6 distinct cache files.
        assert prov.calls == 6
        assert len(list(tmp_path.glob("*.json"))) == 6
        for c in run.cases:
            assert len(c.trial_scores) == 3
            assert c.score == pytest.approx(1.0)
        assert run.metadata["trials"] == 3

    def test_warm_cache_no_requeries(self, tmp_path, monkeypatch):
        prov = _CountingProvider()
        monkeypatch.setattr("rift.runner._get_provider", lambda cfg: prov)
        cfg = ModelConfig(provider="local", model="stub")
        asyncio.run(run_suite(_suite(n=2), cfg, cache_dir=str(tmp_path),
                              show_progress=False, trials=3))
        prov.calls = 0
        asyncio.run(run_suite(_suite(n=2), cfg, cache_dir=str(tmp_path),
                              show_progress=False, trials=3))
        assert prov.calls == 0  # fully served from per-trial cache

    def test_trial_zero_reuses_legacy_cache(self, tmp_path, monkeypatch):
        # A trials=1 run primes the suffix-free key; a later trials>1 run must
        # hit that same entry for trial 0 and only fetch trials 1..k-1.
        prov = _CountingProvider()
        monkeypatch.setattr("rift.runner._get_provider", lambda cfg: prov)
        cfg = ModelConfig(provider="local", model="stub")
        asyncio.run(run_suite(_suite(n=2), cfg, cache_dir=str(tmp_path),
                              show_progress=False, trials=1))
        assert prov.calls == 2
        prov.calls = 0
        asyncio.run(run_suite(_suite(n=2), cfg, cache_dir=str(tmp_path),
                              show_progress=False, trials=3))
        # trial 0 cached for both cases; only trials 1,2 fetched: 2×2 = 4.
        assert prov.calls == 4

    def test_single_trial_leaves_trial_scores_empty(self, tmp_path, monkeypatch):
        prov = _CountingProvider()
        monkeypatch.setattr("rift.runner._get_provider", lambda cfg: prov)
        cfg = ModelConfig(provider="local", model="stub")
        run = asyncio.run(run_suite(_suite(n=2), cfg, cache_dir=str(tmp_path),
                                    show_progress=False, trials=1))
        assert all(c.trial_scores == [] for c in run.cases)
        assert "trials" not in run.metadata

    def test_partial_trial_failure_is_not_an_error(self, tmp_path, monkeypatch):
        # First trial of every case fails with a NON-transient error (so it
        # fails fast, no retry); later trials succeed. The case has a valid mean
        # score and must NOT be counted as an error (regression test).
        class _FlakyFirst:
            def __init__(self):
                self.model = "stub"
                self.seen: set[str] = set()

            async def complete(self, prompt, **kwargs):
                if prompt not in self.seen:
                    self.seen.add(prompt)
                    raise ValueError("non-transient boom")  # not retried
                return Completion(
                    model="stub", input_text=prompt, output_text="4",
                    latency_ms=1.0, input_tokens=1, output_tokens=1,
                    raw_response={}, provider_fingerprint="fp",
                )

            async def close(self):
                pass

        monkeypatch.setattr("rift.runner._get_provider", lambda cfg: _FlakyFirst())
        cfg = ModelConfig(provider="local", model="stub")
        run = asyncio.run(run_suite(_suite(n=2), cfg, cache_dir=str(tmp_path),
                                    show_progress=False, trials=3))
        for c in run.cases:
            # trial 0 failed, trials 1-2 succeeded -> mean 1.0, no error.
            assert c.score == pytest.approx(1.0)
            assert c.error is None
            assert c.attempts == 1  # real attempt count of the successful work
        assert run.metadata["n_errors"] == 0

    def test_attempts_reflect_cache_hit(self, tmp_path, monkeypatch):
        # Cache hits report attempts=0 (the audit-trail invariant).
        prov = _CountingProvider()
        monkeypatch.setattr("rift.runner._get_provider", lambda cfg: prov)
        cfg = ModelConfig(provider="local", model="stub")
        asyncio.run(run_suite(_suite(n=2), cfg, cache_dir=str(tmp_path),
                              show_progress=False, trials=2))
        run = asyncio.run(run_suite(_suite(n=2), cfg, cache_dir=str(tmp_path),
                                    show_progress=False, trials=2))
        assert all(c.attempts == 0 for c in run.cases)  # fully cached
