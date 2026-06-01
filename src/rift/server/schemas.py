"""Request/response models for the control-plane API.

Reuses pydantic (already a core Rift dependency) — no new dependency for
validation. These models are intentionally thin: the database row dicts from
:mod:`rift.server.store` are the source of truth, and these just shape the
HTTP boundary.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class MonitorCreate(BaseModel):
    """Payload to define a new monitor.

    A *monitor* is a standing comparison: run ``suite`` against ``baseline``
    and ``challenger`` on ``schedule_cron`` (a standard 5-field cron string;
    omit to leave it manual-trigger-only).
    """

    name: str = Field(..., min_length=1)
    suite: str
    baseline: str
    challenger: str
    schedule_cron: str | None = None
    enabled: bool = True
    alert_webhook: str | None = None
    cache_dir: str | None = None
    # Replay against committed recorded outcomes instead of calling live APIs
    # (keyless). Powers the offline demo and backtests.
    replay: bool = False


class MonitorUpdate(BaseModel):
    """Patch payload — every field optional."""

    name: str | None = None
    suite: str | None = None
    baseline: str | None = None
    challenger: str | None = None
    schedule_cron: str | None = None
    enabled: bool | None = None
    alert_webhook: str | None = None
    cache_dir: str | None = None
    replay: bool | None = None
