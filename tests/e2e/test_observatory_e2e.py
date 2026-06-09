"""End-to-end: `rift observe --from-runs` → data dir → `rift observatory-site`.

Drives the real CLI as a subprocess, fully offline (replay mode needs no
keys and no cache seeding — the saved runs ARE the data).
"""

from __future__ import annotations

import json
from pathlib import Path

from rift.runner import CaseResult, RunResult


def _run(scores: list[float], *, model: str, fingerprint: str,
         completed_at: str) -> RunResult:
    cases = [
        CaseResult(
            case_index=i, input_text=f"q{i}", expected=f"a{i}",
            output=f"Answer: a{i}\nConfidence: 0.8", score=s,
            latency_ms=5.0, input_tokens=50, output_tokens=20,
            cost_usd=0.001, provider_fingerprint=fingerprint,
        )
        for i, s in enumerate(scores)
    ]
    return RunResult(
        model=model, suite_name="e2e_suite", scoring_method="exact_match",
        cases=cases, completed_at=completed_at,
        metadata={"fingerprints": [fingerprint]},
    )


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
    # The accuracy collapse also moves Brier/ECE → a calibration notice
    # rides along with the gated score_drift.
    assert {"score_drift", "fingerprint_change"} <= kinds
    assert "silent_swap" not in kinds

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
