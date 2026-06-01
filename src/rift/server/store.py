"""SQLite persistence for the control plane.

Deliberately ORM-free: the schema is tiny (three tables) and stdlib
``sqlite3`` keeps the ``[server]`` extra small. A fresh connection is opened
per call so the scheduler's worker threads never share a connection (sqlite
objects are not safe to pass across threads).

Large blobs (the full ``DriftResult`` and both ``RunResult`` JSONs) are stored
as TEXT columns. List endpoints select only the summary columns so paginating
history never deserializes megabytes of per-case IO; ``get_run`` returns the
blobs for the drill-down view.

This module is the single seam a future multi-tenant store would replace:
every function takes ``db_path`` explicitly and returns plain dicts, so the
service/API layers never touch sqlite directly.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

# Columns returned for run *lists* — everything except the heavy JSON blobs.
_RUN_SUMMARY_COLS = (
    "id, monitor_id, started_at, completed_at, status, p_value, delta, "
    "delta_pct, cost_normalized_delta_usd, significant, regression, error"
)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _last_id(cur: sqlite3.Cursor) -> int:
    """The id of the just-inserted row (``lastrowid`` is typed Optional)."""
    assert cur.lastrowid is not None, "INSERT did not yield a rowid"
    return int(cur.lastrowid)


def _connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str | Path) -> None:
    """Create the schema if it doesn't exist. Idempotent."""
    p = Path(db_path)
    if p.parent and str(p.parent) not in ("", "."):
        p.parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS monitors (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT    NOT NULL,
                suite         TEXT    NOT NULL,
                baseline      TEXT    NOT NULL,
                challenger    TEXT    NOT NULL,
                schedule_cron TEXT,
                enabled       INTEGER NOT NULL DEFAULT 1,
                alert_webhook TEXT,
                cache_dir     TEXT,
                replay        INTEGER NOT NULL DEFAULT 0,
                created_at    TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS runs (
                id                         INTEGER PRIMARY KEY AUTOINCREMENT,
                monitor_id                 INTEGER NOT NULL
                                             REFERENCES monitors(id) ON DELETE CASCADE,
                started_at                 TEXT,
                completed_at               TEXT,
                status                     TEXT    NOT NULL DEFAULT 'running',
                p_value                    REAL,
                delta                      REAL,
                delta_pct                  REAL,
                cost_normalized_delta_usd  REAL,
                significant                INTEGER,
                regression                 INTEGER,
                drift_json                 TEXT,
                baseline_run_json          TEXT,
                challenger_run_json        TEXT,
                error                      TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_runs_monitor
                ON runs(monitor_id, id DESC);

            CREATE TABLE IF NOT EXISTS alerts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                monitor_id  INTEGER NOT NULL,
                run_id      INTEGER,
                created_at  TEXT    NOT NULL,
                kind        TEXT    NOT NULL,
                delivered   INTEGER NOT NULL DEFAULT 0,
                payload     TEXT
            );
            """
        )


# --------------------------------------------------------------------------- #
# Monitors
# --------------------------------------------------------------------------- #


def create_monitor(
    db_path: str | Path,
    *,
    name: str,
    suite: str,
    baseline: str,
    challenger: str,
    schedule_cron: str | None = None,
    enabled: bool = True,
    alert_webhook: str | None = None,
    cache_dir: str | None = None,
    replay: bool = False,
) -> dict[str, Any]:
    with _connect(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO monitors
                 (name, suite, baseline, challenger, schedule_cron, enabled,
                  alert_webhook, cache_dir, replay, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (name, suite, baseline, challenger, schedule_cron, int(enabled),
             alert_webhook, cache_dir, int(replay), _now()),
        )
        monitor_id = _last_id(cur)
    got = get_monitor(db_path, monitor_id)
    assert got is not None
    return got


def list_monitors(db_path: str | Path) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM monitors ORDER BY id"
        ).fetchall()
    return [_monitor_row(r) for r in rows]


def get_monitor(db_path: str | Path, monitor_id: int) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM monitors WHERE id = ?", (monitor_id,)
        ).fetchone()
    return _monitor_row(row) if row else None


def update_monitor(
    db_path: str | Path, monitor_id: int, **fields: Any
) -> dict[str, Any] | None:
    """Patch a subset of monitor columns. Unknown keys are ignored."""
    allowed = {
        "name", "suite", "baseline", "challenger", "schedule_cron",
        "enabled", "alert_webhook", "cache_dir", "replay",
    }
    sets = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not sets:
        return get_monitor(db_path, monitor_id)
    # Coerce booleans to the 0/1 ints sqlite stores.
    for bool_col in ("enabled", "replay"):
        if bool_col in sets:
            sets[bool_col] = int(bool(sets[bool_col]))
    assignments = ", ".join(f"{k} = ?" for k in sets)
    with _connect(db_path) as conn:
        conn.execute(
            f"UPDATE monitors SET {assignments} WHERE id = ?",
            (*sets.values(), monitor_id),
        )
    return get_monitor(db_path, monitor_id)


def delete_monitor(db_path: str | Path, monitor_id: int) -> bool:
    with _connect(db_path) as conn:
        cur = conn.execute("DELETE FROM monitors WHERE id = ?", (monitor_id,))
        return cur.rowcount > 0


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #


def create_run(
    db_path: str | Path, *, monitor_id: int, started_at: str | None = None
) -> int:
    """Insert a ``status='running'`` row and return its id."""
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO runs (monitor_id, started_at, status) VALUES (?, ?, 'running')",
            (monitor_id, started_at or _now()),
        )
        return _last_id(cur)


def update_run(db_path: str | Path, run_id: int, **fields: Any) -> None:
    """Patch run columns (used to finalize a row as ok/regression/error)."""
    allowed = {
        "completed_at", "status", "p_value", "delta", "delta_pct",
        "cost_normalized_delta_usd", "significant", "regression",
        "drift_json", "baseline_run_json", "challenger_run_json", "error",
    }
    sets = {k: v for k, v in fields.items() if k in allowed}
    if not sets:
        return
    assignments = ", ".join(f"{k} = ?" for k in sets)
    with _connect(db_path) as conn:
        conn.execute(
            f"UPDATE runs SET {assignments} WHERE id = ?",
            (*sets.values(), run_id),
        )


def get_run(db_path: str | Path, run_id: int) -> dict[str, Any] | None:
    """Full run row including the heavy JSON blobs."""
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    return _run_row(row) if row else None


def list_runs(
    db_path: str | Path, monitor_id: int, limit: int | None = None
) -> list[dict[str, Any]]:
    """Run history (summary columns only), oldest→newest for time-series plots."""
    q = (
        f"SELECT {_RUN_SUMMARY_COLS} FROM runs WHERE monitor_id = ? ORDER BY id"
    )
    params: tuple[Any, ...] = (monitor_id,)
    if limit is not None:
        # Take the most recent N, then re-sort ascending for charting.
        q = (
            f"SELECT * FROM (SELECT {_RUN_SUMMARY_COLS} FROM runs "
            f"WHERE monitor_id = ? ORDER BY id DESC LIMIT ?) ORDER BY id"
        )
        params = (monitor_id, limit)
    with _connect(db_path) as conn:
        rows = conn.execute(q, params).fetchall()
    return [dict(r) for r in rows]


def latest_run(db_path: str | Path, monitor_id: int) -> dict[str, Any] | None:
    """Most recent run (summary columns) for a monitor, or None."""
    with _connect(db_path) as conn:
        row = conn.execute(
            f"SELECT {_RUN_SUMMARY_COLS} FROM runs WHERE monitor_id = ? "
            "ORDER BY id DESC LIMIT 1",
            (monitor_id,),
        ).fetchone()
    return dict(row) if row else None


# --------------------------------------------------------------------------- #
# Alerts
# --------------------------------------------------------------------------- #


def record_alert(
    db_path: str | Path,
    *,
    monitor_id: int,
    run_id: int | None,
    kind: str,
    delivered: bool,
    payload: str,
) -> int:
    with _connect(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO alerts
                 (monitor_id, run_id, created_at, kind, delivered, payload)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (monitor_id, run_id, _now(), kind, int(delivered), payload),
        )
        return _last_id(cur)


def list_alerts(
    db_path: str | Path, monitor_id: int, limit: int = 20
) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM alerts WHERE monitor_id = ? ORDER BY id DESC LIMIT ?",
            (monitor_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------- #
# Row coercion
# --------------------------------------------------------------------------- #


def _monitor_row(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["enabled"] = bool(d["enabled"])
    d["replay"] = bool(d["replay"])
    return d


def _run_row(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    for b in ("significant", "regression"):
        if d.get(b) is not None:
            d[b] = bool(d[b])
    return d
