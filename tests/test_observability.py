"""Tests for the observability metrics export (rift.observability)."""

from __future__ import annotations

import json

import pytest

from rift.comparator import compare_runs
from rift.observability import (
    comparison_metrics,
    render,
    render_json,
    render_prometheus,
    run_metrics,
)


def _drift(baseline=None, challenger=None, **kw):
    baseline = baseline or [1, 1, 0, 1, 0, 1, 1, 0]
    challenger = challenger or [1, 0, 0, 0, 0, 1, 1, 0]
    return compare_runs(
        baseline_scores=baseline, challenger_scores=challenger,
        baseline_model=kw.get("bm", "claude-opus-4-7"),
        challenger_model=kw.get("cm", "claude-opus-4-8"),
        suite_name=kw.get("suite", "reasoning"), alpha=0.05,
        baseline_costs=kw.get("bc", [0.01] * len(baseline)),
        challenger_costs=kw.get("cc", [0.02] * len(challenger)),
    )


class TestComparisonMetrics:
    def test_core_metrics_and_labels(self):
        m = comparison_metrics(_drift())[0]
        assert m.labels["baseline"] == "claude-opus-4-7"
        assert m.labels["challenger"] == "claude-opus-4-8"
        assert m.labels["suite"] == "reasoning"
        assert "test_used" in m.labels
        assert m.values["n_cases"] == 8
        assert m.values["drift_delta"] == pytest.approx(0.375 - 0.625)
        assert m.values["regression"] in (0, 1)
        assert m.values["drift_significant"] in (0, 1)

    def test_subgroups_become_extra_series_with_label(self):
        drift = _drift()
        drift.subgroups = {"hard": _drift([1, 0], [0, 0]),
                           "easy": _drift([1, 1], [1, 1])}
        series = comparison_metrics(drift)
        assert len(series) == 3  # main + 2 subgroups
        subs = {m.labels.get("subgroup") for m in series}
        assert subs == {None, "hard", "easy"}

    def test_non_finite_omitted(self):
        # Zero correct on challenger -> cost_per_correct is inf; must be dropped.
        drift = _drift([1, 1, 1, 1], [0, 0, 0, 0],
                       bc=[0.01] * 4, cc=[0.01] * 4)
        m = comparison_metrics(drift)[0]
        for k, v in m.values.items():
            assert v == v and v not in (float("inf"), float("-inf")), k


class TestRunMetrics:
    def test_run_metrics(self):
        from rift.runner import CaseResult, RunResult
        run = RunResult(
            model="gpt-4o", suite_name="extraction", scoring_method="exact_match",
            cases=[
                CaseResult(0, "i", "e", "o", 1.0, 10.0, 5, 7, 0.01),
                CaseResult(1, "i", "e", "o", 0.0, 10.0, 5, 7, 0.02),
            ],
        )
        m = run_metrics(run)[0]
        assert m.labels == {"model": "gpt-4o", "suite": "extraction",
                            "scoring": "exact_match"}
        assert m.values["n_cases"] == 2
        assert m.values["mean_score"] == pytest.approx(0.5)
        assert m.values["total_cost_usd"] == pytest.approx(0.03)
        assert m.values["total_input_tokens"] == 10


class TestRenderJSON:
    def test_valid_json_schema(self):
        payload = json.loads(render_json(comparison_metrics(_drift())))
        assert payload["schema"] == "rift.metrics/v1"
        assert "generated_at" in payload
        assert len(payload["series"]) == 1
        assert "labels" in payload["series"][0]
        assert "metrics" in payload["series"][0]

    def test_render_dispatch(self):
        s = comparison_metrics(_drift())
        assert render(s, "json").startswith("{")
        assert render(s, "prometheus").startswith("# HELP")
        with pytest.raises(ValueError, match="unknown metrics format"):
            render(s, "csv")


class TestRenderPrometheus:
    def test_help_type_once_per_metric(self):
        drift = _drift()
        drift.subgroups = {"hard": _drift([1, 0], [0, 0])}
        out = render_prometheus(comparison_metrics(drift))
        help_counts: dict[str, int] = {}
        type_counts: dict[str, int] = {}
        for line in out.splitlines():
            if line.startswith("# HELP "):
                help_counts[line.split()[2]] = help_counts.get(line.split()[2], 0) + 1
            elif line.startswith("# TYPE "):
                type_counts[line.split()[2]] = type_counts.get(line.split()[2], 0) + 1
        assert all(c == 1 for c in help_counts.values()), help_counts
        assert all(c == 1 for c in type_counts.values()), type_counts
        assert 'subgroup="hard"' in out

    def test_label_escaping(self):
        out = render_prometheus(comparison_metrics(
            _drift(bm="e\\f", cm='c"d', suite="a\nb")))
        assert 'suite="a\\nb"' in out
        assert 'challenger="c\\"d"' in out
        assert 'baseline="e\\\\f"' in out

    def test_namespaced_names(self):
        out = render_prometheus(comparison_metrics(_drift()))
        assert "rift_drift_delta{" in out
        assert "rift_n_cases{" in out

    def test_integer_values_not_floatified(self):
        out = render_prometheus(comparison_metrics(_drift()))
        line = next(ln for ln in out.splitlines()
                    if ln.startswith("rift_n_cases{"))
        assert line.endswith(" 8")

    def test_no_non_finite_in_output(self):
        out = render_prometheus(comparison_metrics(
            _drift([1, 1, 1, 1], [0, 0, 0, 0], bc=[0.01] * 4, cc=[0.01] * 4)))
        assert "inf" not in out.lower()
        assert "nan" not in out.lower()


class TestWriteMetrics:
    def test_write_both_formats(self, tmp_path):
        from rift.observability import write_metrics
        s = comparison_metrics(_drift())
        pj = tmp_path / "sub" / "m.json"
        pp = tmp_path / "m.prom"
        write_metrics(s, str(pj), "json")
        write_metrics(s, str(pp), "prometheus")
        assert json.loads(pj.read_text())["schema"] == "rift.metrics/v1"
        assert pp.read_text().startswith("# HELP")
