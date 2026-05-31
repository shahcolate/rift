"""Export comparison / run results as flat metrics for dashboards.

Rift's ``--output`` writes a rich, nested JSON for humans and re-analysis. This
module instead emits a *flat, stable* set of named metrics suitable for piping
into a time-series store or observability stack:

* ``json``       — ``{"schema", "generated_at", "series": [{labels, metrics}]}``.
  Easy to ship to a log pipeline or load into anything.
* ``prometheus`` — Prometheus text exposition format, for the node_exporter
  textfile collector or a pushgateway.

This is a point-in-time snapshot (one emission per run). For continuous
monitoring, wire the output file into your collector / scheduler. Non-finite
values (e.g. an undefined cost-per-correct when a run has zero correct cases)
are omitted so the JSON stays valid and Prometheus stays clean.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field

NAMESPACE = "rift"
SCHEMA = "rift.metrics/v1"

# Help text per (un-prefixed) metric name, surfaced in the Prometheus output.
_HELP: dict[str, str] = {
    "n_cases": "Number of paired eval cases compared.",
    "baseline_mean": "Baseline mean score.",
    "challenger_mean": "Challenger mean score.",
    "drift_delta": "Challenger minus baseline mean score.",
    "drift_delta_pct": "Drift delta as a percentage of the baseline mean.",
    "drift_p_value": "P-value of the drift significance test.",
    "drift_ci_lower": "Lower bound of the drift delta confidence interval.",
    "drift_ci_upper": "Upper bound of the drift delta confidence interval.",
    "drift_significant": "1 if the drift is statistically significant, else 0.",
    "regression": "1 if a significant regression was detected, else 0.",
    "effect_size": "Effect size of the drift (see effect_size_kind label).",
    "n_regressed_cases": "Number of cases that regressed.",
    "n_improved_cases": "Number of cases that improved.",
    "baseline_cost_usd": "Total baseline cost in USD.",
    "challenger_cost_usd": "Total challenger cost in USD.",
    "baseline_cost_per_correct": "Baseline USD per fully-correct case.",
    "challenger_cost_per_correct": "Challenger USD per fully-correct case.",
    "cost_normalized_delta_usd": "Challenger minus baseline USD per correct case.",
    "mean_score": "Mean score over the run.",
    "total_cost_usd": "Total run cost in USD.",
    "total_input_tokens": "Total input tokens over the run.",
    "total_output_tokens": "Total output tokens over the run.",
}


@dataclass
class Metrics:
    """One labelled series of named metric values."""

    labels: dict[str, str]
    values: dict[str, float]
    help: dict[str, str] = field(default_factory=dict)


def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _drift_to_metrics(drift, extra_labels: dict[str, str]) -> Metrics:
    labels = {
        "baseline": drift.baseline_model,
        "challenger": drift.challenger_model,
        "suite": drift.suite_name,
        "test_used": drift.test_used,
        "effect_size_kind": drift.effect_size_kind,
        **extra_labels,
    }
    regressed = bool(drift.significant and drift.delta < 0)
    values: dict[str, float] = {
        "n_cases": drift.n_cases,
        "baseline_mean": drift.baseline_mean,
        "challenger_mean": drift.challenger_mean,
        "drift_delta": drift.delta,
        "drift_delta_pct": drift.delta_pct,
        "drift_p_value": drift.p_value,
        "drift_ci_lower": drift.ci_lower,
        "drift_ci_upper": drift.ci_upper,
        "drift_significant": int(drift.significant),
        "regression": int(regressed),
        "effect_size": drift.effect_size,
        "n_regressed_cases": len(drift.regressed_cases),
        "n_improved_cases": len(drift.improved_cases),
        "baseline_cost_usd": drift.baseline_cost_usd,
        "challenger_cost_usd": drift.challenger_cost_usd,
        "baseline_cost_per_correct": drift.baseline_cost_per_correct,
        "challenger_cost_per_correct": drift.challenger_cost_per_correct,
    }
    if drift.cost_delta_ci_defined:
        values["cost_normalized_delta_usd"] = drift.cost_normalized_delta_usd
    # Drop any non-finite value (keeps JSON valid; e.g. cost-per-correct = inf).
    values = {k: v for k, v in values.items() if _finite(v)}
    return Metrics(labels=labels, values=values,
                   help={k: _HELP[k] for k in values if k in _HELP})


def comparison_metrics(drift) -> list[Metrics]:
    """Flatten a :class:`DriftResult` (plus any subgroups) into metric series."""
    series = [_drift_to_metrics(drift, {})]
    for name, sub in (getattr(drift, "subgroups", None) or {}).items():
        series.append(_drift_to_metrics(sub, {"subgroup": name}))
    return series


def run_metrics(run) -> list[Metrics]:
    """Flatten a single :class:`RunResult` into one metric series."""
    labels = {
        "model": run.model,
        "suite": run.suite_name,
        "scoring": run.scoring_method,
    }
    values = {
        "n_cases": len(run.cases),
        "mean_score": run.mean_score,
        "total_cost_usd": run.total_cost_usd,
        "total_input_tokens": run.total_input_tokens,
        "total_output_tokens": run.total_output_tokens,
    }
    values = {k: v for k, v in values.items() if _finite(v)}
    return [Metrics(labels=labels, values=values,
                    help={k: _HELP[k] for k in values if k in _HELP})]


# ----------------------------- rendering ------------------------------------

def _escape_label(v: str) -> str:
    return str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _format_value(v: float) -> str:
    f = float(v)
    return str(int(f)) if f.is_integer() else repr(f)


def render_prometheus(series: list[Metrics], namespace: str = NAMESPACE) -> str:
    """Render series in Prometheus text exposition format.

    HELP/TYPE are emitted once per metric name (Prometheus requirement), then
    one sample line per series that carries that metric.
    """
    # Preserve first-seen metric order across all series.
    names: list[str] = []
    for m in series:
        for name in m.values:
            if name not in names:
                names.append(name)

    lines: list[str] = []
    for name in names:
        full = f"{namespace}_{name}"
        help_text = next((m.help.get(name) for m in series if name in m.help), None)
        if help_text:
            lines.append(f"# HELP {full} {help_text}")
        lines.append(f"# TYPE {full} gauge")
        for m in series:
            if name not in m.values:
                continue
            if m.labels:
                lbl = ",".join(f'{k}="{_escape_label(v)}"'
                               for k, v in m.labels.items())
                lines.append(f"{full}{{{lbl}}} {_format_value(m.values[name])}")
            else:
                lines.append(f"{full} {_format_value(m.values[name])}")
    return "\n".join(lines) + "\n"


def render_json(series: list[Metrics]) -> str:
    payload = {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "series": [{"labels": m.labels, "metrics": m.values} for m in series],
    }
    return json.dumps(payload, indent=2)


def render(series: list[Metrics], fmt: str) -> str:
    if fmt == "json":
        return render_json(series)
    if fmt == "prometheus":
        return render_prometheus(series)
    raise ValueError(f"unknown metrics format '{fmt}'; use 'json' or 'prometheus'")


def write_metrics(series: list[Metrics], path: str, fmt: str) -> None:
    from pathlib import Path

    text = render(series, fmt)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
