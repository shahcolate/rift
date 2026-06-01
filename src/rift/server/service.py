"""The run → compare → persist → alert pipeline.

This is the heart of the control plane and the only place that touches the
eval engine. It is deliberately a *thin* wrapper: it loads the suite, runs both
models with :func:`rift.runner.run_suite`, scores drift with
:func:`rift.comparator.compare_runs`, persists the result, and fires an alert
on a regression — reusing the exact functions the CLI uses, so a monitored run
and a ``rift compare`` produce identical numbers.

The engine symbols are imported at module scope so unit tests can monkeypatch
``service.run_suite`` / ``service.compare_runs`` and exercise the pipeline
without any network or API keys.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..comparator import DriftResult, compare_runs
from ..config import load_suite, resolve_model
from ..runner import RunResult, run_suite
from . import alerts, store


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


async def _run_pair(
    suite: Any, base_cfg: Any, chal_cfg: Any, cache_dir: str | None
) -> tuple[RunResult, RunResult]:
    """Run baseline then challenger on the same suite, paired & keyless-if-cached.

    Sequential (not gathered) so the two share the completion cache cleanly and
    so rate limits are easier to reason about — the CLI does the same.
    """
    base = await run_suite(suite, base_cfg, cache_dir=cache_dir, show_progress=False)
    chal = await run_suite(suite, chal_cfg, cache_dir=cache_dir, show_progress=False)
    return base, chal


def _compute(monitor: dict[str, Any]):
    """Execute a monitor's comparison and return ``(base, chal, drift)``.

    Two paths:
    * ``replay``: replay committed recorded outcomes (keyless) via the same
      machinery that powers ``rift demo``. ``drift`` carries subgroups.
    * live: load the suite and run both models against their providers.
    """
    if monitor.get("replay"):
        # Imported lazily: the demo replay pulls in suite expansion + the
        # recorded-outcomes file, which the generic live path doesn't need.
        from ..demo import replay_recorded_run

        cache_dir = monitor.get("cache_dir")
        return replay_recorded_run(
            monitor["baseline"],
            monitor["challenger"],
            cache_dir=Path(cache_dir) if cache_dir else None,
        )

    suite = load_suite(monitor["suite"])
    base_cfg = resolve_model(monitor["baseline"])
    chal_cfg = resolve_model(monitor["challenger"])
    base, chal = asyncio.run(
        _run_pair(suite, base_cfg, chal_cfg, monitor.get("cache_dir"))
    )
    drift = compare_runs(
        baseline_scores=base.scores,
        challenger_scores=chal.scores,
        baseline_model=base.model,
        challenger_model=chal.model,
        suite_name=suite.name,
        baseline_costs=[c.cost_usd for c in base.cases],
        challenger_costs=[c.cost_usd for c in chal.cases],
    )
    return base, chal, drift


def is_regression(drift: DriftResult) -> bool:
    """Whether a comparison counts as a regression worth alerting on.

    Rift treats cost-normalized drift as a first-class signal, so a monitor
    regresses when *either*:

    * accuracy significantly regressed (the CLI exit-code-1 condition), OR
    * cost-per-correct significantly rose — the whole 95% paired-bootstrap CI
      on the ``$/correct`` delta sits above zero (challenger is strictly more
      expensive per correct answer).

    The second case is exactly the silent regression the demo surfaces: same
    (or better) accuracy, materially higher bill.
    """
    accuracy_regression = bool(
        drift.significant and drift.drift_direction == "regression"
    )
    cost_regression = bool(
        drift.cost_delta_ci_defined and drift.cost_delta_ci_lower > 0.0
    )
    return accuracy_regression or cost_regression


def execute_monitor(
    db_path: str | Path, monitor: dict[str, Any]
) -> dict[str, Any]:
    """Run one monitor end-to-end and persist the result.

    Inserts a ``running`` row first so the dashboard can show in-flight runs,
    then finalizes it as ``ok`` / ``regression`` / ``error``. Any engine error
    (a missing API key, a provider failure, a bad suite) is captured into the
    run row rather than propagated — so a scheduled job never dies and the
    failure is visible in the UI. Returns the finalized run row.
    """
    run_id = store.create_run(
        db_path, monitor_id=int(monitor["id"]), started_at=_now()
    )
    try:
        base, chal, drift = _compute(monitor)
        regression = is_regression(drift)
        store.update_run(
            db_path,
            run_id,
            completed_at=_now(),
            status="regression" if regression else "ok",
            p_value=drift.p_value,
            delta=drift.delta,
            delta_pct=drift.delta_pct,
            cost_normalized_delta_usd=(
                drift.cost_normalized_delta_usd
                if drift.cost_delta_ci_defined else None
            ),
            significant=int(drift.significant),
            regression=int(regression),
            drift_json=_dumps(asdict(drift)),
            baseline_run_json=_dumps(base.to_dict()),
            challenger_run_json=_dumps(chal.to_dict()),
        )
        if regression:
            # Always record the regression event; deliver to the webhook only
            # if one is configured (alerts.fire is a no-op delivery otherwise).
            alerts.fire(db_path, monitor, run_id, asdict(drift))
    except Exception as exc:  # noqa: BLE001 — a failed run is data, not a crash
        store.update_run(
            db_path,
            run_id,
            completed_at=_now(),
            status="error",
            error=f"{type(exc).__name__}: {exc}",
        )
    got = store.get_run(db_path, run_id)
    assert got is not None
    return got


def _dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, default=str)


def drift_from_dict(d: dict[str, Any]) -> DriftResult:
    """Rebuild a :class:`DriftResult` (incl. nested subgroups) from its dict.

    The inverse of ``dataclasses.asdict(drift)`` stored in ``runs.drift_json``.
    Lets the metrics endpoint reuse :func:`rift.observability.comparison_metrics`
    unchanged instead of re-deriving the metric shape.
    """
    d = dict(d)
    subgroups = {
        name: drift_from_dict(sub) for name, sub in (d.pop("subgroups", None) or {}).items()
    }
    return DriftResult(**d, subgroups=subgroups)


# --------------------------------------------------------------------------- #
# Demo seeding — a populated, keyless dashboard out of the box.
# --------------------------------------------------------------------------- #

DEMO_MONITOR_NAME = "Demo · Opus 4.6 → 4.7 (context-rot)"


def seed_demo_monitor(db_path: str | Path) -> dict[str, Any]:
    """Create (once) and run the offline Opus 4.6→4.7 demo monitor.

    Mirrors ``rift demo``: replays committed outcomes, so the dashboard shows a
    real regression finding and a fired alert with no API keys. Idempotent on
    the monitor (won't duplicate), but each call adds a fresh run so the
    time-series grows — handy for showing the dashboard live.
    """
    existing = next(
        (m for m in store.list_monitors(db_path) if m["name"] == DEMO_MONITOR_NAME),
        None,
    )
    if existing is None:
        existing = store.create_monitor(
            db_path,
            name=DEMO_MONITOR_NAME,
            suite="context_rot_reasoning",
            baseline="opus-4-6",
            challenger="opus-4-7",
            schedule_cron=None,
            enabled=True,
            alert_webhook=None,
            cache_dir=None,
            replay=True,
        )
    execute_monitor(db_path, existing)
    return existing
