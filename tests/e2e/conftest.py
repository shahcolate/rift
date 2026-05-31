"""Shared fixtures for end-to-end CLI tests.

These tests drive the *real* ``rift`` entry point as a subprocess — the same
binary a user runs — rather than calling Python functions in-process. Network
is never touched: each model completion the run needs is pre-seeded into Rift's
on-disk cache in the exact format the runner reads (``_cache_key`` →
``{key}.json`` holding a ``Completion`` dict). A dummy API key is set so the
live-command key preflight passes while the seeded cache prevents any real call.

This exercises the full path a user hits: argv parsing → config/suite load →
runner (cache layer, scoring) → comparator → reporter → exit code → output
files. If any of those links breaks, these tests fail where in-process stubs
would not.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from rift.config import resolve_model
from rift.providers import Completion
from rift.runner import _cache_key

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def workdir(tmp_path) -> Path:
    return tmp_path


class CacheSeeder:
    """Seed Rift's completion cache so a real run resolves offline."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def seed(self, model: str, prompt: str, output: str, *,
             model_params: dict | None = None,
             input_tokens: int = 100, output_tokens: int = 20) -> None:
        """Seed one completion. ``model`` is a CLI identifier/alias."""
        resolved = resolve_model(model).model
        key = _cache_key(resolved, prompt, model_params or {})
        comp = Completion(
            model=resolved, input_text=prompt, output_text=output,
            latency_ms=10.0, input_tokens=input_tokens,
            output_tokens=output_tokens, raw_response={},
        )
        (self.cache_dir / f"{key}.json").write_text(
            json.dumps(asdict(comp), default=str)
        )


@pytest.fixture
def seed_cache(workdir):
    def _make(subdir: str = "cache") -> CacheSeeder:
        return CacheSeeder(workdir / subdir)
    return _make


@pytest.fixture
def run_rift(workdir):
    """Invoke the installed ``rift`` CLI as a subprocess.

    Returns a ``CompletedProcess``. A dummy key for every provider is set so
    the key preflight passes; the seeded cache means no network is used.
    """
    def _run(*args: str, expect_exit: int | None = None,
             env_extra: dict | None = None) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env.update({
            "ANTHROPIC_API_KEY": "dummy-e2e",
            "OPENAI_API_KEY": "dummy-e2e",
            "GEMINI_API_KEY": "dummy-e2e",
            # Never read a developer's real ~/.rift/.env during tests.
            "HOME": str(workdir),
        })
        if env_extra:
            env.update(env_extra)
        proc = subprocess.run(
            [sys.executable, "-m", "rift.cli", *args],
            capture_output=True, text=True, env=env, cwd=str(workdir),
        )
        if expect_exit is not None and proc.returncode != expect_exit:
            raise AssertionError(
                f"rift {' '.join(args)}\n"
                f"expected exit {expect_exit}, got {proc.returncode}\n"
                f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        return proc
    return _run


@pytest.fixture
def write_suite(workdir):
    def _write(name: str, content: str) -> Path:
        p = workdir / name
        p.write_text(content)
        return p
    return _write
