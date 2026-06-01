"""Regression alerting — fire a webhook when a monitor detects a regression.

The payload is Slack-compatible (a ``text`` summary plus Block Kit ``blocks``),
which also POSTs cleanly to Discord, a generic webhook receiver, or anything
that accepts JSON. Delivery is best-effort: a failed POST is recorded as an
undelivered alert row, never an exception that kills the run that triggered it.

Network I/O is isolated in :func:`_post` so tests can stub it without a server.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import store


def build_payload(monitor: dict[str, Any], drift: dict[str, Any]) -> dict[str, Any]:
    """Build the Slack-compatible alert body from a monitor + drift dict.

    ``drift`` is the serialized :class:`~rift.comparator.DriftResult`
    (``dataclasses.asdict``), so this stays decoupled from the dataclass.
    """
    base = drift.get("baseline_model", monitor["baseline"])
    chal = drift.get("challenger_model", monitor["challenger"])
    delta_pp = float(drift.get("delta", 0.0)) * 100.0
    p = float(drift.get("p_value", 1.0))
    suite = drift.get("suite_name", monitor["suite"])
    cost_delta = drift.get("cost_normalized_delta_usd")
    cost_line = ""
    if drift.get("cost_delta_ci_defined") and cost_delta is not None:
        cost_line = f"  •  Δ $/correct: {float(cost_delta):+.4f}"

    headline = (
        f"⚠️ Rift detected a regression on *{monitor['name']}* "
        f"({base} → {chal}, suite `{suite}`)"
    )
    detail = (
        f"Accuracy {delta_pp:+.2f}pp  •  p={p:.4f}  •  "
        f"test={drift.get('test_used', 'n/a')}{cost_line}"
    )
    return {
        "text": f"{headline}\n{detail}",
        "blocks": [
            {"type": "section",
             "text": {"type": "mrkdwn", "text": headline}},
            {"type": "section",
             "text": {"type": "mrkdwn", "text": detail}},
        ],
    }


def _post(url: str, payload: dict[str, Any], timeout: float = 10.0) -> int:
    """POST ``payload`` as JSON to ``url``; return the HTTP status code.

    Isolated for testability — patch this in unit tests to avoid network I/O.
    """
    import httpx

    resp = httpx.post(url, json=payload, timeout=timeout)
    return resp.status_code


def fire(
    db_path: str | Path,
    monitor: dict[str, Any],
    run_id: int,
    drift: dict[str, Any],
) -> dict[str, Any]:
    """Deliver a regression alert for ``monitor`` and record it.

    Returns the payload that was (attempted to be) sent. Never raises on a
    delivery failure — the failure is persisted as an undelivered alert row.
    """
    payload = build_payload(monitor, drift)
    webhook = monitor.get("alert_webhook")
    delivered = False
    record = dict(payload)
    if webhook:
        try:
            status = _post(webhook, payload)
            delivered = 200 <= status < 300
            record["_delivery"] = {"status": status}
        except Exception as exc:  # noqa: BLE001 — delivery must never crash a run
            record["_delivery"] = {"error": f"{type(exc).__name__}: {exc}"}
    store.record_alert(
        db_path,
        monitor_id=int(monitor["id"]),
        run_id=run_id,
        kind="regression",
        delivered=delivered,
        payload=json.dumps(record),
    )
    return payload
