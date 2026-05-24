"""Tests for ``rift demo``.

Three contracts:
1. Replay determinism — headline numbers don't silently shift.
2. Demo script shape — structure invariants for renderers/exporters.
3. HTML export self-containment — "VP opens it offline" contract.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from rift.demo import (
    SCENARIOS,
    DemoAct,
    DemoScript,
    VerdictCard,
    build_opus47_demo_script,
    export_demo_html,
    export_demo_markdown,
    load_scenario,
)


@pytest.fixture(scope="module")
def replayed():
    """One shared replay per test module (cache hits make this cheap)."""
    return load_scenario("opus-46-vs-47")


def test_replay_recorded_runs_deterministic(replayed):
    """Headline numbers must not silently shift on the published benchmark.

    These are the punchline of the demo; if a code change moves them,
    we want CI to catch it before the README claims something else.
    """
    script, base_run, chal_run, drift = replayed

    # Accuracy — challenger > baseline, but not significant.
    assert drift.baseline_mean == pytest.approx(0.8438, abs=1e-3)
    assert drift.challenger_mean == pytest.approx(0.8750, abs=1e-3)
    assert drift.delta > 0
    assert not drift.significant

    # Cost — challenger costs more per correct answer (the whole point).
    assert drift.challenger_cost_per_correct > drift.baseline_cost_per_correct
    cpc_pct = ((drift.challenger_cost_per_correct
                - drift.baseline_cost_per_correct)
               / drift.baseline_cost_per_correct * 100)
    assert 35.0 < cpc_pct < 45.0, f"$/correct delta {cpc_pct:.1f}% outside expected band"

    # Token inflation — the WHY.
    in_ratio = chal_run.total_input_tokens / base_run.total_input_tokens
    assert in_ratio == pytest.approx(1.45, abs=0.01)


def test_build_demo_script_shape(replayed):
    """The script must have exactly four acts, each with both renderers."""
    script, *_ = replayed
    assert isinstance(script, DemoScript)
    assert len(script.acts) == 4
    for act in script.acts:
        assert isinstance(act, DemoAct)
        assert act.title and isinstance(act.title, str)
        assert callable(act.render_fn)
        assert act.body_md and isinstance(act.body_md, str)
        assert act.beat_seconds > 0

    assert isinstance(script.verdict, VerdictCard)
    assert script.verdict.headline
    assert script.verdict.recommendation
    assert len(script.verdict.action_items) >= 3
    assert script.verdict.reproduce_cmd

    assert script.headline_numbers
    assert "accuracy_delta" in script.headline_numbers
    assert "cost_per_correct_pct" in script.headline_numbers


def test_export_demo_html_self_contained(replayed, tmp_path):
    """HTML must be a single self-contained file — no external assets.

    This is the "VP opens it offline" contract. Email clients,
    PDF-from-browser, USB stick — none of those resolve external URLs.
    """
    script, base_run, chal_run, drift = replayed
    out = tmp_path / "memo.html"
    export_demo_html(script, out, base_run, chal_run, drift)

    txt = out.read_text()
    # No external links / scripts.
    assert not re.search(r"<link[^>]+href=", txt), "external stylesheet found"
    assert not re.search(r"<script[^>]+src=", txt), "external script found"
    # Size sanity: small enough to forward in email.
    assert len(txt) < 60_000, f"HTML is {len(txt)} bytes — over the 60KB budget"
    # All four acts (Act 1 is "What we tested", then 3 narrative acts).
    assert "Act 1" in txt and "Act 2" in txt and "Act 3" in txt
    assert "Verdict" in txt
    # KPI grid renders the three headline numbers.
    assert "accuracy delta" in txt
    assert "$/correct delta" in txt
    assert "input tokens" in txt


def test_export_demo_markdown(replayed, tmp_path):
    """Markdown export covers title, all acts, and the verdict."""
    script, *_ = replayed
    out = tmp_path / "memo.md"
    export_demo_markdown(script, out)
    txt = out.read_text()
    assert txt.startswith("# Rift demo")
    for act in script.acts:
        assert act.title in txt or act.body_md.splitlines()[0] in txt
    assert script.verdict.headline in txt
    assert "Reproduce" in txt


def test_scenarios_registry_has_opus47():
    """The registered scenario must point to the build factory."""
    assert "opus-46-vs-47" in SCENARIOS
    spec = SCENARIOS["opus-46-vs-47"]
    assert spec["build"] is build_opus47_demo_script
    assert spec["baseline"] == "opus-4-6"
    assert spec["challenger"] == "opus-4-7"


def test_safe_pct_change_handles_inf_and_zero():
    """Percentage helper returns None on undefined inputs (no nan%)."""
    from rift.demo import _fmt_pct, _safe_pct_change

    # zero correct → cost_per_correct() returns inf → undefined pct
    assert _safe_pct_change(float("inf"), 1.0) is None
    assert _safe_pct_change(1.0, float("inf")) is None
    assert _safe_pct_change(float("nan"), 1.0) is None
    assert _safe_pct_change(0.0, 1.0) is None
    # well-defined case still works
    assert _safe_pct_change(1.0, 1.5) == pytest.approx(50.0)
    # rendering
    assert _fmt_pct(None) == "n/a"
    assert _fmt_pct(50.0) == "+50.0%"
    assert _fmt_pct(-10.0) == "-10.0%"


def test_prime_cache_raises_clearly_on_missing_tags():
    """A case without origin:/distractor: tags must error clearly."""
    import tempfile
    from dataclasses import dataclass

    from rift.demo import prime_cache_from_recording

    @dataclass
    class FakeCase:
        case_index: int
        input: str
        tags: list[str]

    @dataclass
    class FakeSuite:
        name: str
        cases: list
        model_params: dict

    suite = FakeSuite(
        name="bad_suite",
        cases=[FakeCase(case_index=0, input="x", tags=["something_else"])],
        model_params={},
    )
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(ValueError, match="missing required tags"):
            prime_cache_from_recording(suite, "claude-opus-4-6", {}, Path(td))
