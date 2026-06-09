"""End-to-end: `rift observe --from-runs` → data dir → `rift observatory-site`.

Drives the real CLI as a subprocess, fully offline (replay mode needs no
keys and no cache seeding — the saved runs ARE the data).
"""

from __future__ import annotations

import json
from pathlib import Path

from ..test_observatory import make_run


def _run(scores, *, model, fingerprint, completed_at):
    # Same synthetic-run builder the unit tests use, so the replay path is
    # exercised with the exact metadata shape the live runner produces.
    return make_run(scores, model=model, suite="e2e_suite",
                    fingerprint=fingerprint, completed_at=completed_at)


def test_observe_replay_then_site(workdir: Path, run_rift):
    week1 = _run([1.0] * 12, model="my-model", fingerprint="fp-1",
                 completed_at="2026-01-05T00:00:00Z")
    week2 = _run([0.0] * 8 + [1.0] * 4, model="my-model", fingerprint="fp-2",
                 completed_at="2026-01-12T00:00:00Z")
    week1.save(workdir / "week1.json")
    week2.save(workdir / "week2.json")

    data_dir = workdir / "obs-data"
    proc1 = run_rift("observe", "--from-runs", "week1.json",
                     "--data-dir", str(data_dir), expect_exit=0)
    assert "my-model" in proc1.stdout

    proc2 = run_rift("observe", "--from-runs", "week2.json",
                     "--data-dir", str(data_dir), expect_exit=0)
    # Week 2 regresses AND the fingerprint changed → both surface.
    assert "score_drift" in proc2.stdout
    assert "fingerprint_change" in proc2.stdout

    index_lines = (data_dir / "index.jsonl").read_text().strip().splitlines()
    assert len(index_lines) == 2
    entry = json.loads(index_lines[1])
    assert entry["date"] == "2026-01-12"
    assert entry["fingerprints"] == ["fp-2"]
    events = [json.loads(line) for line in
              (data_dir / "events.jsonl").read_text().strip().splitlines()]
    kinds = {e["kind"] for e in events}
    # Exact event vocabulary for this scenario: the gated regression, the
    # concurrent fingerprint change, and the calibration notice the
    # accuracy collapse drags along (Brier/ECE move with it). Anything
    # else appearing here is a detector bug.
    assert kinds == {"score_drift", "fingerprint_change", "notice"}

    site = workdir / "site"
    run_rift("observatory-site", "--data-dir", str(data_dir),
             "--out", str(site), expect_exit=0)
    front = (site / "index.html").read_text()
    assert "my-model" in front and "score drift" in front
    assert (site / "model" / "my-model.html").exists()
    assert (site / "data" / "index.jsonl").exists()


def test_observe_refuses_duplicate_observation(workdir: Path, run_rift):
    week = _run([1.0] * 3, model="m", fingerprint="fp",
                completed_at="2026-01-05T00:00:00Z")
    week.save(workdir / "week.json")
    data_dir = workdir / "obs-data"
    run_rift("observe", "--from-runs", "week.json",
             "--data-dir", str(data_dir), expect_exit=0)
    proc = run_rift("observe", "--from-runs", "week.json",
                    "--data-dir", str(data_dir))
    assert proc.returncode != 0
    assert "append-only" in (proc.stdout + proc.stderr)
