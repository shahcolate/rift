"""Tests for the self-hosted control plane (rift.server).

All tests are keyless and offline: the eval engine (run_suite / compare_runs)
is monkeypatched with canned results, so nothing here touches a provider or the
network. The scheduler is exercised without waiting on cron, and webhook
delivery is stubbed.
"""

from __future__ import annotations

import json

import pytest

# The server extra is optional; skip the whole module cleanly if it's absent.
pytest.importorskip("fastapi")
pytest.importorskip("apscheduler")

from fastapi.testclient import TestClient  # noqa: E402

from rift.comparator import DriftResult  # noqa: E402
from rift.config import resolve_model  # noqa: E402
from rift.providers import MissingAPIKeyError  # noqa: E402
from rift.runner import CaseResult, RunResult  # noqa: E402
from rift.server import alerts, service, store  # noqa: E402
from rift.server.app import create_app  # noqa: E402
from rift.server.scheduler import MonitorScheduler  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures & builders
# --------------------------------------------------------------------------- #


@pytest.fixture
def db(tmp_path):
    p = tmp_path / "rift.db"
    store.init_db(str(p))
    return str(p)


def _run(model: str, scores, costs) -> RunResult:
    cases = [
        CaseResult(
            case_index=i, input_text=f"q{i}", expected="x", output="y",
            score=float(s), latency_ms=1.0, input_tokens=10, output_tokens=5,
            cost_usd=float(c),
        )
        for i, (s, c) in enumerate(zip(scores, costs))
    ]
    return RunResult(model=model, suite_name="reasoning",
                     scoring_method="exact_match", cases=cases)


def _drift(*, delta=0.0, p=1.0, cost_ci_lower=0.0, cost_defined=False) -> DriftResult:
    """A hand-built DriftResult for deterministic regression/no-regression tests."""
    return DriftResult(
        baseline_model="opus-4-7", challenger_model="opus-4-8",
        suite_name="reasoning", n_cases=8,
        baseline_mean=0.5, challenger_mean=0.5 + delta, delta=delta,
        delta_pct=delta * 100, p_value=p, ci_lower=delta, ci_upper=delta,
        significant=(p < 0.05), test_used="mcnemar_exact",
        regressed_cases=[], improved_cases=[],
        cost_normalized_delta_usd=cost_ci_lower,
        cost_delta_ci_lower=cost_ci_lower, cost_delta_ci_upper=cost_ci_lower + 0.01,
        cost_delta_ci_defined=cost_defined,
    )


def _patch_engine(monkeypatch, *, base_run=None, chal_run=None, drift=None):
    """Patch service.run_suite (async) and service.compare_runs with canned data.

    Runs are keyed by the *resolved* model id (``opus-4-7`` → ``claude-opus-4-7``)
    since that's what ``run_suite`` receives via ``resolve_model``.
    """
    base_model = resolve_model("opus-4-7").model
    chal_model = resolve_model("opus-4-8").model
    base_run = base_run or _run(base_model, [1, 1, 0, 1], [0.01] * 4)
    chal_run = chal_run or _run(chal_model, [1, 1, 0, 1], [0.01] * 4)
    runs = {base_run.model: base_run, chal_run.model: chal_run}

    async def fake_run_suite(suite, model_config, **kwargs):
        return runs[model_config.model]

    monkeypatch.setattr(service, "run_suite", fake_run_suite)
    if drift is not None:
        monkeypatch.setattr(service, "compare_runs", lambda **kw: drift)


def _make_monitor(db, **overrides):
    fields = dict(name="m", suite="reasoning", baseline="opus-4-7",
                  challenger="opus-4-8")
    fields.update(overrides)
    return store.create_monitor(db, **fields)


# --------------------------------------------------------------------------- #
# store
# --------------------------------------------------------------------------- #


class TestStore:
    def test_monitor_crud(self, db):
        m = _make_monitor(db, schedule_cron="0 3 * * *", alert_webhook="http://x")
        assert m["id"] == 1 and m["enabled"] is True and m["replay"] is False
        assert store.get_monitor(db, 1)["name"] == "m"
        assert len(store.list_monitors(db)) == 1

        m2 = store.update_monitor(db, 1, enabled=False, name="renamed")
        assert m2["enabled"] is False and m2["name"] == "renamed"

        assert store.delete_monitor(db, 1) is True
        assert store.get_monitor(db, 1) is None
        assert store.list_monitors(db) == []

    def test_run_lifecycle_and_summary_columns(self, db):
        _make_monitor(db)
        rid = store.create_run(db, monitor_id=1)
        assert store.get_run(db, rid)["status"] == "running"

        store.update_run(db, rid, status="ok", delta=0.1, p_value=0.5,
                         drift_json='{"a": 1}')
        full = store.get_run(db, rid)
        assert full["status"] == "ok" and full["drift_json"] == '{"a": 1}'

        # list_runs returns summary columns only (no heavy blobs).
        summary = store.list_runs(db, 1)[0]
        assert "drift_json" not in summary and summary["delta"] == 0.1
        assert store.latest_run(db, 1)["id"] == rid

    def test_list_runs_limit_orders_ascending(self, db):
        _make_monitor(db)
        ids = [store.create_run(db, monitor_id=1) for _ in range(5)]
        last_three = store.list_runs(db, 1, limit=3)
        assert [r["id"] for r in last_three] == ids[-3:]  # most-recent 3, ascending

    def test_alerts(self, db):
        _make_monitor(db)
        store.record_alert(db, monitor_id=1, run_id=None, kind="regression",
                           delivered=True, payload="{}")
        rows = store.list_alerts(db, 1)
        assert len(rows) == 1 and rows[0]["delivered"] == 1

    def test_delete_cascades_runs(self, db):
        _make_monitor(db)
        store.create_run(db, monitor_id=1)
        store.delete_monitor(db, 1)
        assert store.list_runs(db, 1) == []


# --------------------------------------------------------------------------- #
# service pipeline
# --------------------------------------------------------------------------- #


class TestService:
    def test_no_regression_run_is_ok(self, db, monkeypatch):
        _patch_engine(monkeypatch, drift=_drift(delta=0.0, p=1.0))
        m = _make_monitor(db)
        run = service.execute_monitor(db, m)
        assert run["status"] == "ok"
        assert run["regression"] is False
        assert store.list_alerts(db, 1) == []
        # Blobs persisted for drill-down.
        assert json.loads(run["drift_json"])["suite_name"] == "reasoning"
        assert json.loads(run["baseline_run_json"])["model"] == "claude-opus-4-7"

    def test_accuracy_regression_fires_and_delivers_webhook(self, db, monkeypatch):
        _patch_engine(monkeypatch, drift=_drift(delta=-0.3, p=0.01))
        posted = {}

        def fake_post(url, payload, timeout=10.0):
            posted["url"] = url
            posted["payload"] = payload
            return 200

        monkeypatch.setattr(alerts, "_post", fake_post)
        m = _make_monitor(db, alert_webhook="http://hooks.example/x")
        run = service.execute_monitor(db, m)

        assert run["status"] == "regression" and run["regression"] is True
        assert posted["url"] == "http://hooks.example/x"
        assert "regression" in posted["payload"]["text"]
        assert "blocks" in posted["payload"]
        a = store.list_alerts(db, 1)[0]
        assert a["delivered"] == 1 and a["kind"] == "regression"

    def test_cost_regression_flags_even_when_accuracy_improves(self, db, monkeypatch):
        # Accuracy up (delta +0.2, not significant) but $/correct CI above 0.
        _patch_engine(
            monkeypatch,
            drift=_drift(delta=0.2, p=0.9, cost_ci_lower=0.01, cost_defined=True),
        )
        m = _make_monitor(db)
        run = service.execute_monitor(db, m)
        assert run["status"] == "regression"
        # Recorded even without a webhook configured (delivered=0).
        a = store.list_alerts(db, 1)[0]
        assert a["delivered"] == 0

    def test_missing_key_becomes_error_run_not_crash(self, db, monkeypatch):
        async def boom(suite, model_config, **kwargs):
            raise MissingAPIKeyError("anthropic")

        monkeypatch.setattr(service, "run_suite", boom)
        m = _make_monitor(db)
        run = service.execute_monitor(db, m)  # must not raise
        assert run["status"] == "error"
        assert "MissingAPIKeyError" in run["error"]

    def test_is_regression_logic(self):
        assert service.is_regression(_drift(delta=-0.2, p=0.01)) is True
        assert service.is_regression(_drift(delta=0.2, p=0.01)) is False  # improvement
        assert service.is_regression(_drift(delta=0.0, p=0.5)) is False
        assert service.is_regression(
            _drift(cost_ci_lower=0.01, cost_defined=True)
        ) is True
        # Cost delta defined but CI straddles 0 → not a regression.
        assert service.is_regression(
            _drift(cost_ci_lower=-0.01, cost_defined=True)
        ) is False


# --------------------------------------------------------------------------- #
# HTTP API
# --------------------------------------------------------------------------- #


@pytest.fixture
def client(db):
    app = create_app(db, start_scheduler=False)
    return TestClient(app)


class TestAPI:
    def test_healthz(self, client):
        assert client.get("/healthz").json() == {"status": "ok"}

    def test_monitor_crud_via_api(self, client):
        r = client.post("/api/monitors", json={
            "name": "nightly", "suite": "reasoning",
            "baseline": "opus-4-7", "challenger": "opus-4-8",
            "schedule_cron": "0 3 * * *",
        })
        assert r.status_code == 201
        mid = r.json()["id"]

        assert client.get(f"/api/monitors/{mid}").json()["name"] == "nightly"
        assert client.get("/api/monitors").json()[0]["latest_run"] is None

        patched = client.patch(f"/api/monitors/{mid}", json={"enabled": False})
        assert patched.json()["enabled"] is False

        assert client.delete(f"/api/monitors/{mid}").status_code == 204
        assert client.get(f"/api/monitors/{mid}").status_code == 404

    def test_trigger_run_records_history(self, client, db, monkeypatch):
        _patch_engine(monkeypatch, drift=_drift(delta=0.0, p=1.0))
        mid = client.post("/api/monitors", json={
            "name": "m", "suite": "reasoning",
            "baseline": "opus-4-7", "challenger": "opus-4-8",
        }).json()["id"]

        # Background task runs within the TestClient request cycle.
        assert client.post(f"/api/monitors/{mid}/run").status_code == 202
        runs = client.get(f"/api/monitors/{mid}/runs").json()
        assert len(runs) == 1 and runs[0]["status"] == "ok"

    def test_get_run_parses_blobs(self, client, db, monkeypatch):
        _patch_engine(monkeypatch, drift=_drift(delta=-0.3, p=0.01))
        mid = client.post("/api/monitors", json={
            "name": "m", "suite": "reasoning",
            "baseline": "opus-4-7", "challenger": "opus-4-8",
        }).json()["id"]
        client.post(f"/api/monitors/{mid}/run")
        run_id = client.get(f"/api/monitors/{mid}/runs").json()[0]["id"]

        detail = client.get(f"/api/runs/{run_id}").json()
        assert detail["drift"]["suite_name"] == "reasoning"
        assert detail["baseline_run"]["model"] == "claude-opus-4-7"

    def test_metrics_endpoint(self, client, db, monkeypatch):
        _patch_engine(monkeypatch, drift=_drift(delta=-0.3, p=0.01))
        mid = client.post("/api/monitors", json={
            "name": "prod-suite", "suite": "reasoning",
            "baseline": "opus-4-7", "challenger": "opus-4-8",
        }).json()["id"]
        client.post(f"/api/monitors/{mid}/run")

        text = client.get("/metrics").text
        assert "rift_drift_delta" in text
        assert 'monitor="prod-suite"' in text
        assert "# TYPE rift_drift_delta gauge" in text

    def test_html_pages_render(self, client):
        assert client.get("/").status_code == 200
        assert client.get("/monitors/new").status_code == 200
        # Create via the HTML form, then view the detail page.
        r = client.post("/monitors", data={
            "name": "via-form", "suite": "reasoning",
            "baseline": "opus-4-7", "challenger": "opus-4-8",
        })
        assert r.status_code == 200  # followed the 303 redirect
        assert "via-form" in client.get("/monitors/1").text


# --------------------------------------------------------------------------- #
# scheduler
# --------------------------------------------------------------------------- #


class TestScheduler:
    def test_enable_registers_disable_removes_job(self, db):
        sched = MonitorScheduler(db)
        m = _make_monitor(db, schedule_cron="0 3 * * *")
        sched.sync_monitor(m)
        assert sched.has_job(m["id"]) is True

        m2 = store.update_monitor(db, m["id"], enabled=False)
        sched.sync_monitor(m2)
        assert sched.has_job(m["id"]) is False

    def test_no_cron_means_no_job(self, db):
        sched = MonitorScheduler(db)
        m = _make_monitor(db)  # no schedule_cron
        sched.sync_monitor(m)
        assert sched.has_job(m["id"]) is False

    def test_malformed_cron_is_ignored(self, db):
        sched = MonitorScheduler(db)
        m = _make_monitor(db, schedule_cron="not a cron")
        sched.sync_monitor(m)  # must not raise
        assert sched.has_job(m["id"]) is False

    def test_remove_monitor(self, db):
        sched = MonitorScheduler(db)
        m = _make_monitor(db, schedule_cron="0 3 * * *")
        sched.sync_monitor(m)
        sched.remove_monitor(m["id"])
        assert sched.has_job(m["id"]) is False


# --------------------------------------------------------------------------- #
# demo seed (keyless replay) — light smoke test
# --------------------------------------------------------------------------- #


class TestDemoSeed:
    def test_seed_demo_monitor_is_keyless_and_flags_cost_regression(self, db):
        m = service.seed_demo_monitor(db)
        assert m["replay"] is True
        runs = store.list_runs(db, m["id"])
        assert len(runs) == 1
        # The Opus 4.6→4.7 replay is a cost regression (accuracy ~flat/up).
        assert runs[0]["status"] == "regression"
        assert store.list_alerts(db, m["id"])  # an alert was recorded

    def test_seed_is_idempotent_on_monitor_but_adds_runs(self, db):
        service.seed_demo_monitor(db)
        service.seed_demo_monitor(db)
        assert len(store.list_monitors(db)) == 1
        assert len(store.list_runs(db, 1)) == 2
