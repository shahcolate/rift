"""Cron scheduling for monitors, backed by APScheduler.

A :class:`MonitorScheduler` keeps an APScheduler ``BackgroundScheduler`` in
sync with the enabled monitors that carry a ``schedule_cron``. Job ids are
derived from the monitor id so re-syncing a monitor replaces its job rather
than stacking duplicates. Jobs run in scheduler worker threads, where
:func:`rift.server.service.execute_monitor` can safely call ``asyncio.run`` (no
event loop is running in those threads).

Manual "run now" triggers do **not** go through here — the API runs them as a
FastAPI background task — so this class is only concerned with the recurring
schedule.
"""

from __future__ import annotations

from pathlib import Path

from . import service, store


def _job_id(monitor_id: int) -> str:
    return f"monitor:{monitor_id}"


class MonitorScheduler:
    """Owns the APScheduler instance and reconciles it with the DB."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        # Imported here so the dependency is only required when a scheduler is
        # actually constructed (keeps `import rift.server` cheap).
        from apscheduler.schedulers.background import BackgroundScheduler

        self._scheduler = BackgroundScheduler()
        self._started = False

    # -- lifecycle ---------------------------------------------------------- #

    def start(self) -> None:
        """Start the scheduler and register every enabled, scheduled monitor."""
        if not self._started:
            self._scheduler.start()
            self._started = True
        for monitor in store.list_monitors(self.db_path):
            self.sync_monitor(monitor)

    def shutdown(self) -> None:
        if self._started:
            self._scheduler.shutdown(wait=False)
            self._started = False

    # -- reconciliation ----------------------------------------------------- #

    def sync_monitor(self, monitor: dict) -> None:
        """Add/replace/remove the job for a single monitor to match its state."""
        monitor_id = int(monitor["id"])
        self.remove_monitor(monitor_id)
        cron = monitor.get("schedule_cron")
        if not monitor.get("enabled") or not cron:
            return
        from apscheduler.triggers.cron import CronTrigger

        try:
            trigger = CronTrigger.from_crontab(cron)
        except ValueError:
            # A malformed cron string shouldn't take down the whole sync; the
            # monitor simply stays manual-trigger-only.
            return
        self._scheduler.add_job(
            self._run,
            trigger=trigger,
            id=_job_id(monitor_id),
            args=[monitor_id],
            replace_existing=True,
            misfire_grace_time=None,
            coalesce=True,
        )

    def remove_monitor(self, monitor_id: int) -> None:
        if self._scheduler.get_job(_job_id(monitor_id)):
            self._scheduler.remove_job(_job_id(monitor_id))

    def has_job(self, monitor_id: int) -> bool:
        return self._scheduler.get_job(_job_id(monitor_id)) is not None

    # -- job body ----------------------------------------------------------- #

    def _run(self, monitor_id: int) -> None:
        """Re-read the monitor (it may have changed) and execute it."""
        monitor = store.get_monitor(self.db_path, monitor_id)
        if monitor and monitor.get("enabled"):
            service.execute_monitor(self.db_path, monitor)
