"""FastAPI application factory for the Rift control plane.

Exposes both a JSON/REST API (``/api/...``, ``/metrics``, ``/healthz``) and a
server-rendered HTML dashboard (``/``, ``/monitors/...``). The app owns a
:class:`~rift.server.scheduler.MonitorScheduler` started/stopped via the
ASGI lifespan, and reuses the existing engine through
:mod:`rift.server.service`.

Build one with :func:`create_app`. ``rift serve`` (see ``rift.cli``) wraps this
in uvicorn; tests build it directly with ``start_scheduler=False`` and a temp
SQLite path.

NOTE: this module intentionally does *not* use ``from __future__ import
annotations``. FastAPI resolves endpoint annotations at decoration time, and
the request/form types are imported locally inside :func:`create_app`; keeping
annotations as real objects (not strings) lets FastAPI see them directly.
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from ..config import BUILTIN_SUITES_DIR
from . import charts, service, store
from .schemas import MonitorCreate, MonitorUpdate

_HERE = Path(__file__).parent
_TEMPLATES = _HERE / "templates"
_STATIC = _HERE / "static"


def _builtin_suite_names() -> list[str]:
    try:
        return sorted(p.stem for p in BUILTIN_SUITES_DIR.glob("*.yaml"))
    except OSError:
        return []


def create_app(
    db_path: str | Path = "rift_monitors.db",
    *,
    start_scheduler: bool = True,
    cache_root: str | Path | None = None,
) -> "Any":
    """Construct the FastAPI app.

    Parameters
    ----------
    db_path:
        SQLite file for monitors/runs/alerts. Created if absent.
    start_scheduler:
        Start the cron scheduler on app startup. Disable in tests to avoid
        background threads.
    cache_root:
        Reserved seam for per-monitor completion-cache isolation (multi-tenant
        roadmap). Unused in the single-tenant MVP beyond being recorded.
    """
    from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, Request
    from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates

    db_path = str(db_path)
    store.init_db(db_path)

    templates = Jinja2Templates(directory=str(_TEMPLATES))
    templates.env.globals["sparkline"] = charts.sparkline
    templates.env.globals["status_badge"] = charts.status_badge

    @asynccontextmanager
    async def lifespan(app: "FastAPI"):
        sched = None
        if start_scheduler:
            from .scheduler import MonitorScheduler

            sched = MonitorScheduler(db_path)
            sched.start()
        app.state.scheduler = sched
        try:
            yield
        finally:
            if sched is not None:
                sched.shutdown()

    app = FastAPI(title="Rift", description="Continuous LLM drift monitoring",
                  lifespan=lifespan)
    app.state.db_path = db_path
    app.state.cache_root = str(cache_root) if cache_root else None
    app.state.scheduler = None

    if _STATIC.is_dir():
        app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

    # -- helpers ----------------------------------------------------------- #

    def _sync_scheduler(monitor: dict | None, monitor_id: int | None = None) -> None:
        sched = app.state.scheduler
        if sched is None:
            return
        if monitor is not None:
            sched.sync_monitor(monitor)
        elif monitor_id is not None:
            sched.remove_monitor(monitor_id)

    def _require_monitor(monitor_id: int) -> dict:
        m = store.get_monitor(db_path, monitor_id)
        if m is None:
            raise HTTPException(status_code=404, detail="monitor not found")
        return m

    # ===================================================================== #
    # JSON API
    # ===================================================================== #

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/monitors")
    def api_list_monitors() -> list[dict[str, Any]]:
        out = []
        for m in store.list_monitors(db_path):
            m = dict(m)
            m["latest_run"] = store.latest_run(db_path, m["id"])
            out.append(m)
        return out

    @app.post("/api/monitors", status_code=201)
    def api_create_monitor(payload: MonitorCreate) -> dict[str, Any]:
        m = store.create_monitor(db_path, **payload.model_dump())
        _sync_scheduler(m)
        return m

    @app.get("/api/monitors/{monitor_id}")
    def api_get_monitor(monitor_id: int) -> dict[str, Any]:
        return _require_monitor(monitor_id)

    @app.patch("/api/monitors/{monitor_id}")
    def api_update_monitor(monitor_id: int, payload: MonitorUpdate) -> dict[str, Any]:
        _require_monitor(monitor_id)
        m = store.update_monitor(
            db_path, monitor_id, **payload.model_dump(exclude_unset=True)
        )
        _sync_scheduler(m)
        assert m is not None
        return m

    @app.delete("/api/monitors/{monitor_id}", status_code=204)
    def api_delete_monitor(monitor_id: int) -> None:
        _require_monitor(monitor_id)
        store.delete_monitor(db_path, monitor_id)
        _sync_scheduler(None, monitor_id=monitor_id)

    @app.post("/api/monitors/{monitor_id}/run", status_code=202)
    def api_run_monitor(monitor_id: int, background: BackgroundTasks) -> dict[str, Any]:
        m = _require_monitor(monitor_id)
        background.add_task(service.execute_monitor, db_path, m)
        return {"status": "queued", "monitor_id": monitor_id}

    @app.get("/api/monitors/{monitor_id}/runs")
    def api_list_runs(monitor_id: int, limit: int | None = None) -> list[dict[str, Any]]:
        _require_monitor(monitor_id)
        return store.list_runs(db_path, monitor_id, limit=limit)

    @app.get("/api/runs/{run_id}")
    def api_get_run(run_id: int) -> dict[str, Any]:
        run = store.get_run(db_path, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        # Parse the stored JSON blobs into structured fields for convenience.
        for key in ("drift_json", "baseline_run_json", "challenger_run_json"):
            if run.get(key):
                try:
                    run[key.removesuffix("_json")] = json.loads(run[key])
                except (ValueError, TypeError):
                    pass
        return run

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        """Prometheus exposition of the latest run per enabled monitor.

        Reuses :func:`rift.observability.comparison_metrics` so the metric
        names/labels match ``rift compare --metrics-out`` exactly.
        """
        from ..observability import comparison_metrics, render_prometheus

        series = []
        for m in store.list_monitors(db_path):
            if not m["enabled"]:
                continue
            latest = store.latest_run(db_path, m["id"])
            if not latest or latest.get("status") not in ("ok", "regression"):
                continue
            full = store.get_run(db_path, latest["id"])
            if not full or not full.get("drift_json"):
                continue
            try:
                drift = service.drift_from_dict(json.loads(full["drift_json"]))
            except (ValueError, TypeError):
                continue
            for metric in comparison_metrics(drift):
                metric.labels["monitor"] = m["name"]
                series.append(metric)
        return render_prometheus(series)

    # ===================================================================== #
    # HTML dashboard
    # ===================================================================== #

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        monitors = []
        for m in store.list_monitors(db_path):
            runs = store.list_runs(db_path, m["id"], limit=30)
            deltas = [r["delta"] for r in runs if r.get("delta") is not None]
            monitors.append({
                "monitor": m,
                "latest": store.latest_run(db_path, m["id"]),
                "n_runs": len(runs),
                "deltas": deltas,
            })
        return templates.TemplateResponse(
            request, "index.html", {"monitors": monitors}
        )

    @app.get("/monitors/new", response_class=HTMLResponse)
    def new_monitor_form(request: Request):
        return templates.TemplateResponse(
            request, "new.html", {"suites": _builtin_suite_names()}
        )

    @app.post("/monitors")
    def create_monitor_form(
        request: Request,
        name: str = Form(...),
        suite: str = Form(...),
        baseline: str = Form(...),
        challenger: str = Form(...),
        schedule_cron: str = Form(""),
        alert_webhook: str = Form(""),
        replay: bool = Form(False),
    ):
        m = store.create_monitor(
            db_path,
            name=name,
            suite=suite,
            baseline=baseline,
            challenger=challenger,
            schedule_cron=schedule_cron or None,
            alert_webhook=alert_webhook or None,
            replay=replay,
        )
        _sync_scheduler(m)
        return RedirectResponse(f"/monitors/{m['id']}", status_code=303)

    @app.post("/monitors/{monitor_id}/run")
    def run_monitor_form(
        monitor_id: int, background: "BackgroundTasks"
    ):
        m = _require_monitor(monitor_id)
        background.add_task(service.execute_monitor, db_path, m)
        return RedirectResponse(f"/monitors/{monitor_id}", status_code=303)

    @app.post("/monitors/{monitor_id}/toggle")
    def toggle_monitor_form(monitor_id: int):
        m = _require_monitor(monitor_id)
        m2 = store.update_monitor(db_path, monitor_id, enabled=not m["enabled"])
        _sync_scheduler(m2)
        return RedirectResponse(f"/monitors/{monitor_id}", status_code=303)

    @app.post("/monitors/{monitor_id}/delete")
    def delete_monitor_form(monitor_id: int):
        _require_monitor(monitor_id)
        store.delete_monitor(db_path, monitor_id)
        _sync_scheduler(None, monitor_id=monitor_id)
        return RedirectResponse("/", status_code=303)

    @app.get("/monitors/{monitor_id}", response_class=HTMLResponse)
    def monitor_detail(request: Request, monitor_id: int):
        m = _require_monitor(monitor_id)
        runs = store.list_runs(db_path, monitor_id)
        deltas = [r["delta"] for r in runs if r.get("delta") is not None]
        cost_deltas = [
            r["cost_normalized_delta_usd"] for r in runs
            if r.get("cost_normalized_delta_usd") is not None
        ]
        latest_full = None
        if runs:
            latest_full = api_get_run(runs[-1]["id"])
        alerts_list = store.list_alerts(db_path, monitor_id)
        next_run = None
        sched = app.state.scheduler
        if sched is not None and sched.has_job(monitor_id):
            job = sched._scheduler.get_job(f"monitor:{monitor_id}")  # noqa: SLF001
            next_run = str(getattr(job, "next_run_time", "") or "")
        return templates.TemplateResponse(
            request,
            "monitor.html",
            {
                "monitor": m,
                "runs": list(reversed(runs)),  # newest first in the table
                "deltas": deltas,
                "cost_deltas": cost_deltas,
                "latest": latest_full,
                "alerts": alerts_list,
                "next_run": next_run,
            },
        )

    return app
