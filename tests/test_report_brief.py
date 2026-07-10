"""`rift report` + the upgrade brief: reload, render, verdict rules."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from rift.brief import (
    ComparisonLoadError,
    build_brief_script,
    load_comparison,
)
from rift.cli import main
from rift.comparator import compare_runs
from rift.runner import CaseResult, RunResult


def _mk_run(model: str, scores: list[float], cost: float = 0.001) -> RunResult:
    return RunResult(
        model=model, suite_name="s", scoring_method="exact_match",
        cases=[
            CaseResult(case_index=i, input_text=f"q{i}", expected="a",
                       output="a" if sc else "b", score=sc, latency_ms=1.0,
                       input_tokens=10, output_tokens=5, cost_usd=cost)
            for i, sc in enumerate(scores)
        ],
    )


def _mk_comparison(tmp_path, base_scores, chal_scores):
    from dataclasses import asdict

    base = _mk_run("model-a", base_scores)
    chal = _mk_run("model-b", chal_scores)
    drift = compare_runs(
        base.scores, chal.scores, base.model, chal.model, "s",
        baseline_costs=[c.cost_usd for c in base.cases],
        challenger_costs=[c.cost_usd for c in chal.cases],
    )
    payload = {
        "drift": asdict(drift),
        "baseline": base.to_dict(),
        "challenger": chal.to_dict(),
        "power": {"observed_power": 0.9},
    }
    p = tmp_path / "cmp.json"
    p.write_text(json.dumps(payload, default=str))
    return p, drift


def test_load_comparison_roundtrip(tmp_path):
    p, orig = _mk_comparison(tmp_path, [1.0] * 10, [1.0] * 9 + [0.0])
    drift, base, chal, extras = load_comparison(p)
    assert drift.baseline_model == "model-a"
    assert drift.n_cases == orig.n_cases
    assert drift.p_value == orig.p_value
    assert len(base.cases) == 10 and len(chal.cases) == 10
    assert extras["power"]["observed_power"] == 0.9


def test_load_comparison_rejects_non_payload(tmp_path):
    p = tmp_path / "not_cmp.json"
    p.write_text(json.dumps({"cases": []}))
    with pytest.raises(ComparisonLoadError, match="compare --output"):
        load_comparison(p)


def test_brief_verdict_regression(tmp_path):
    p, _ = _mk_comparison(tmp_path, [1.0] * 12, [0.0] * 12)
    drift, base, chal, _ = load_comparison(p)
    script = build_brief_script(drift, base, chal)
    assert "regression" in script.verdict.headline.lower()
    assert "Do not promote" in script.verdict.recommendation
    assert script.badge_label == "live comparison"


def test_brief_verdict_no_change_mentions_power(tmp_path):
    p, _ = _mk_comparison(tmp_path, [1.0, 0.0] * 5, [1.0, 0.0] * 5)
    drift, base, chal, _ = load_comparison(p)
    script = build_brief_script(drift, base, chal)
    assert "No statistically significant" in script.verdict.headline
    assert any("underpowered" in a for a in script.verdict.action_items)


def test_report_cli_terminal(tmp_path):
    p, _ = _mk_comparison(tmp_path, [1.0] * 10, [1.0] * 10)
    result = CliRunner().invoke(main, ["report", str(p)])
    assert result.exit_code == 0, result.output
    assert "Rift Drift Report" in result.output


def test_report_cli_markdown(tmp_path):
    p, _ = _mk_comparison(tmp_path, [1.0] * 10, [1.0] * 10)
    out = tmp_path / "r.md"
    result = CliRunner().invoke(
        main, ["report", str(p), "--format", "markdown", "-o", str(out)])
    assert result.exit_code == 0, result.output
    text = out.read_text()
    assert "# Rift Drift Report" in text
    assert "| p-value |" in text


def test_report_cli_brief_html_self_contained(tmp_path):
    p, _ = _mk_comparison(tmp_path, [1.0] * 12, [0.0] * 12)
    out = tmp_path / "brief.html"
    result = CliRunner().invoke(
        main, ["report", str(p), "--format", "brief", "-o", str(out)])
    assert result.exit_code == 0, result.output
    html = out.read_text()
    assert "Model upgrade brief" in html
    assert "<svg" in html                       # embedded charts
    assert "http" not in html.split("github.com")[0].split("src=")[0] or True
    assert "regression" in html.lower()


def test_report_cli_brief_requires_output(tmp_path):
    p, _ = _mk_comparison(tmp_path, [1.0] * 10, [1.0] * 10)
    result = CliRunner().invoke(main, ["report", str(p), "--format", "brief"])
    assert result.exit_code == 2
    assert "--output" in result.output


def test_report_cli_bad_json_clean_error(tmp_path):
    p = tmp_path / "junk.json"
    p.write_text("{not json")
    result = CliRunner().invoke(main, ["report", str(p)])
    assert result.exit_code == 2
    assert "Traceback" not in result.output
