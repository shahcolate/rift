"""`rift estimate`: keyless pre-flight cost, grounded in the budget guard's heuristic."""

from __future__ import annotations

import json

from click.testing import CliRunner

from rift.cli import main
from rift.config import load_suite
from rift.estimate import (
    calibration_from_run,
    estimate_grid,
    estimate_panel,
    estimate_stage,
)
from rift.observatory import EST_OUTPUT_TOKENS_PER_CASE, estimate_stage_cost, load_panel
from rift.pricing import PRICING


def test_stage_matches_budget_guard_heuristic():
    # The whole point: the number the user sees is the number the
    # observatory guard checks against max_cost_usd.
    suite = load_suite("reasoning")
    st = estimate_stage("fable-5-1", suite)
    assert st.cost_usd == estimate_stage_cost("claude-fable-5-1", suite)
    assert st.calls == len(suite.cases)
    assert st.output_tokens == EST_OUTPUT_TOKENS_PER_CASE * len(suite.cases)
    assert st.priced and not st.keyless


def test_alias_resolution_and_price_ratio():
    suite = load_suite("reasoning")
    fable = estimate_stage("fable-5-1", suite)
    opus = estimate_stage("opus-5", suite)
    assert fable.resolved_model == "claude-fable-5-1"
    # Same token workload → cost ratio is exactly the list-price ratio.
    assert abs(fable.cost_usd / opus.cost_usd - 2.0) < 1e-9


def test_trials_and_pushback_scale_calls():
    suite = load_suite("reasoning")
    one = estimate_stage("sonnet-5", suite)
    three = estimate_stage("sonnet-5", suite, trials=3)
    assert three.calls == 3 * one.calls
    assert abs(three.cost_usd - 3 * one.cost_usd) < 1e-9
    pb = estimate_stage("sonnet-5", suite, pushback=True)
    assert pb.calls == 2 * one.calls
    assert pb.cost_usd > one.cost_usd


def test_unpriced_hosted_model_is_flagged_and_bounded_high():
    suite = load_suite("reasoning")
    st = estimate_stage("claude-brand-new-9", suite)
    assert st.priced is False
    assert "NOT in pricing catalog" in st.note
    top = max(PRICING.values(), key=lambda p: p.cost(1, 1))
    assert st.cost_usd == top.cost(st.input_tokens, st.output_tokens)


def test_riftlm_and_self_hosted_cost_zero(tmp_path):
    suite = load_suite("reasoning")
    st = estimate_stage("llama-3.3-70b@http://localhost:8000", suite)
    assert st.keyless and st.cost_usd == 0.0


def test_calibration_prefers_own_model_row_else_heaviest(tmp_path):
    suite = load_suite("reasoning")
    cmp_json = {
        "baseline": {"model": "claude-opus-4-7", "suite_name": "reasoning",
                     "cases": [{"input_tokens": 100, "output_tokens": 50}]},
        "challenger": {"model": "claude-fable-5", "suite_name": "reasoning",
                       "cases": [{"input_tokens": 100, "output_tokens": 1000}]},
    }
    path = tmp_path / "reasoning.json"
    path.write_text(json.dumps(cmp_json))
    name, rows = calibration_from_run(path)
    assert name == "reasoning"
    assert rows["claude-fable-5"]["output_tokens"] == 1000

    own = estimate_stage("fable-5", suite, calibration=rows)
    assert own.output_tokens == 1000 and "measured (claude-fable-5)" in own.note
    # A model not in the file borrows the HEAVIEST row (conservative).
    other = estimate_stage("opus-5", suite, calibration=rows)
    assert other.output_tokens == 1000 and "measured on claude-fable-5" in other.note


def test_grid_and_panel_totals():
    grid = estimate_grid(["fable-5-1", "sonnet-5"], ["reasoning", "code_generation"])
    assert len(grid.stages) == 4
    assert grid.total_usd == sum(s.cost_usd for s in grid.stages)
    assert {s.suite for s in grid.stages} == {"reasoning", "code_generation"}

    panel = load_panel("observatory/panel.yaml")
    est = estimate_panel(panel)
    assert len(est.stages) == len(panel.endpoints) * len(panel.suites)
    # Endpoint ids (not raw model strings) label the rows.
    assert {s.model for s in est.stages} == {ep.id for ep in panel.endpoints}
    # The sycophancy suite carries the pushback doubling.
    syc = [s for s in est.stages if s.suite == panel.sycophancy_on]
    assert all(s.calls == 2 * s.n_cases for s in syc)


def test_committed_panel_fits_its_own_cap():
    # If the panel's heuristic estimate exceeds max_cost_usd the guard would
    # skip stages on EVERY pass — the panel would be misconfigured on main.
    panel = load_panel("observatory/panel.yaml")
    est = estimate_panel(panel)
    assert est.total_usd <= panel.max_cost_usd
    assert not est.unpriced_models, (
        f"panel endpoints without a pricing entry: {est.unpriced_models}"
    )


def test_cli_grid_and_panel_are_keyless(monkeypatch):
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    runner = CliRunner()
    r = runner.invoke(main, ["estimate", "--model", "fable-5-1", "--model", "opus-5",
                             "--suite", "reasoning"])
    assert r.exit_code == 0, r.output
    assert "fable-5-1" in r.output and "Total" in r.output

    r = runner.invoke(main, ["estimate", "--panel", "observatory/panel.yaml"])
    assert r.exit_code == 0, r.output
    assert "max_cost_usd" in r.output and "within" in r.output


def test_cli_usage_errors():
    runner = CliRunner()
    assert runner.invoke(main, ["estimate"]).exit_code != 0
    r = runner.invoke(main, ["estimate", "--panel", "observatory/panel.yaml",
                             "--model", "opus-5"])
    assert r.exit_code != 0 and "replaces" in r.output
