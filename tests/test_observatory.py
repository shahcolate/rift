"""Observatory: records, append-only data dir, budget guard, drift events."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rift.config import EvalCase, SuiteConfig
from rift.observatory import (
    BudgetTracker,
    DriftEvent,
    append_events,
    append_records,
    build_record,
    detect_drift,
    endpoint_slug,
    estimate_stage_cost,
    load_events,
    load_index,
    load_panel,
    load_selftest,
    panel_version_hash,
    replay_panel,
)
from rift.runner import CaseResult, RunResult


def make_run(
    scores: list[float],
    *,
    model: str = "panel-model",
    suite: str = "panel_suite",
    fingerprint: str | None = "fp-a",
    rollout: bool = False,
    outputs: list[str] | None = None,
    errors: list[str | None] | None = None,
    expected: list[str] | None = None,
    completed_at: str = "2026-01-05T00:00:00Z",
) -> RunResult:
    """Synthetic RunResult in the exact shape the runner produces."""
    cases = []
    for i, s in enumerate(scores):
        err = errors[i] if errors else None
        cases.append(CaseResult(
            case_index=i,
            input_text=f"question {i}",
            expected=(expected[i] if expected else f"answer-{i}"),
            output="" if err else (
                outputs[i] if outputs else f"Answer: answer-{i}\nConfidence: 0.9"
            ),
            score=s,
            latency_ms=10.0,
            input_tokens=0 if err else 100,
            output_tokens=0 if err else 50,
            cost_usd=0.0 if err else 0.001,
            error=err,
            provider_fingerprint=None if err else fingerprint,
        ))
    metadata: dict = {"n_errors": sum(1 for c in cases if c.error)}
    if fingerprint:
        metadata["fingerprints"] = (
            [fingerprint, f"{fingerprint}-next"] if rollout else [fingerprint]
        )
        if rollout:
            metadata["fingerprint_rollout"] = True
    return RunResult(
        model=model, suite_name=suite, scoring_method="exact_match",
        cases=cases, started_at="", completed_at=completed_at,
        metadata=metadata,
    )


def record_week(
    data_dir: Path,
    date: str,
    scores: list[float],
    *,
    endpoint: str = "ep",
    fingerprint: str | None = "fp-a",
    rollout: bool = False,
    pushback_scores: list[float] | None = None,
    outputs: list[str] | None = None,
    errors: list[str | None] | None = None,
    expected: list[str] | None = None,
    suite: str = "panel_suite",
) -> list[dict]:
    """Build + append one (date, endpoint, suite) observation."""
    run = make_run(scores, model=endpoint, suite=suite,
                   fingerprint=fingerprint, rollout=rollout, outputs=outputs,
                   errors=errors, expected=expected)
    pushback = None
    if pushback_scores is not None:
        pushback = make_run(pushback_scores, model=endpoint,
                            suite=f"{suite}__pushback",
                            fingerprint=fingerprint)
    rec = build_record(run, endpoint, date, pushback_run=pushback)
    return append_records([rec], data_dir)


class TestPanelVersion:
    def test_stable_for_identical_pairs(self):
        pairs = [("q1", "a1"), ("q2", {"k": 1})]
        assert panel_version_hash(pairs) == panel_version_hash(list(pairs))

    def test_sensitive_to_any_change(self):
        base = [("q1", "a1"), ("q2", "a2")]
        assert panel_version_hash(base) != panel_version_hash([("q1", "a1")])
        assert panel_version_hash(base) != panel_version_hash(
            [("q1", "a1"), ("q2", "CHANGED")]
        )
        # Order matters: pairing is positional.
        assert panel_version_hash(base) != panel_version_hash(base[::-1])


class TestBuildRecord:
    def test_derived_block_and_strip_io(self):
        run = make_run([1.0, 0.0, 1.0])
        rec = build_record(run, "ep", "2026-01-05")
        d = rec.derived
        assert d["mean_score"] == pytest.approx(0.6667, abs=1e-3)
        assert d["n_cases"] == 3 and d["n_errors"] == 0
        assert d["fingerprints"] == ["fp-a"]
        assert d["aborted"] is False
        # Confidence parsed from raw outputs BEFORE stripping.
        assert d["calibration"]["confidences"] == [0.9, 0.9, 0.9]
        assert d["refusal"]["rate"] == 0.0
        # Stored run is stripped; scores survive.
        assert rec.run["cases"][0]["input_text"] == ""
        assert rec.run["cases"][0]["output"] == ""
        assert rec.run["cases"][0]["score"] == 1.0

    def test_sycophancy_block(self):
        run = make_run([1.0, 1.0, 0.0])
        push = make_run([1.0, 0.0, 0.0], suite="panel_suite__pushback")
        rec = build_record(run, "ep", "2026-01-05", pushback_run=push)
        s = rec.derived["sycophancy"]
        assert s["flip_rate"] == 0.5  # 1 of 2 originally-correct flipped
        assert s["orig_correct"] == [1, 1, 0]
        assert s["push_correct"] == [1, 0, 0]
        assert s["valid"] == [1, 1, 1]
        # Full stage spend (base + probe) is what the index accounts.
        assert rec.derived["cost_usd"] == pytest.approx(0.006)

    def test_pushback_transport_error_is_not_a_flip(self):
        # An exhausted-retry pushback completion scores 0.0 for transport
        # reasons; counting it as the model caving would publish a spurious
        # sycophancy notice.
        run = make_run([1.0, 1.0, 1.0, 1.0])
        push = make_run([0.0, 1.0, 1.0, 1.0], suite="panel_suite__pushback",
                        errors=["HTTPStatusError: 429"] + [None] * 3)
        rec = build_record(run, "ep", "2026-01-05", pushback_run=push)
        s = rec.derived["sycophancy"]
        assert s["valid"] == [0, 1, 1, 1]
        assert s["n_originally_correct"] == 3  # errored pair excluded
        assert s["flip_rate"] == 0.0

    def test_majority_errors_marks_aborted(self):
        errs: list[str | None] = ["boom", "boom", None]
        run = make_run([0.0, 0.0, 1.0], errors=errs)
        rec = build_record(run, "ep", "2026-01-05")
        assert rec.derived["aborted"] is True

    def test_refusal_detected_in_outputs(self):
        outs = ["I cannot help with that request.", "Answer: answer-1"]
        run = make_run([0.0, 1.0], outputs=outs)
        rec = build_record(run, "ep", "2026-01-05")
        assert rec.derived["refusal"]["flags"] == [True, False]
        assert rec.derived["refusal"]["rate"] == 0.5


class TestDataDir:
    def test_append_and_load_index(self, tmp_path):
        entries = record_week(tmp_path, "2026-01-05", [1.0, 0.0])
        assert len(entries) == 1
        idx = load_index(tmp_path)
        assert idx == entries
        e = idx[0]
        assert e["endpoint"] == "ep" and e["suite"] == "panel_suite"
        assert e["record"].startswith("records/2026-01-05/ep/")
        assert (tmp_path / e["record"]).exists()

    def test_append_only_refuses_overwrite(self, tmp_path):
        import click

        record_week(tmp_path, "2026-01-05", [1.0])
        with pytest.raises(click.ClickException, match="append-only"):
            record_week(tmp_path, "2026-01-05", [1.0])

    def test_duplicate_batch_writes_nothing(self, tmp_path):
        # A duplicate anywhere in the batch must be caught BEFORE any file
        # is written — a partial batch would orphan records without index
        # lines and permanently block the date.
        import click

        run_a = make_run([1.0], model="ep-a")
        dup1 = build_record(run_a, "ep-a", "2026-01-05")
        dup2 = build_record(make_run([0.0], model="ep-a"), "ep-a", "2026-01-05")
        other = build_record(make_run([1.0], model="ep-b"), "ep-b", "2026-01-05")
        with pytest.raises(click.ClickException, match="append-only"):
            append_records([other, dup1, dup2], tmp_path)
        assert not (tmp_path / "index.jsonl").exists()
        assert not list(tmp_path.glob("records/**/*.json"))

    def test_events_roundtrip(self, tmp_path):
        ev = DriftEvent(date="2026-01-05", endpoint="ep", kind="notice",
                        summary="hello")
        append_events([ev], tmp_path)
        loaded = load_events(tmp_path)
        assert loaded[0]["kind"] == "notice"
        assert loaded[0]["summary"] == "hello"

    def test_load_selftest(self, tmp_path):
        st_dir = tmp_path / "selftest"
        st_dir.mkdir()
        (st_dir / f"{endpoint_slug('ep/x')}.json").write_text(
            json.dumps({"false_regression_rate": 0.02, "reps": 500})
        )
        st = load_selftest(tmp_path, "ep/x")
        assert st is not None and st["false_regression_rate"] == 0.02
        assert load_selftest(tmp_path, "missing") is None


class TestDetectDrift:
    def test_no_prior_no_events(self, tmp_path):
        record_week(tmp_path, "2026-01-05", [1.0] * 6)
        assert detect_drift(tmp_path, "2026-01-05") == []

    def test_stable_endpoint_no_events(self, tmp_path):
        record_week(tmp_path, "2026-01-05", [1.0, 0.0, 1.0, 1.0])
        record_week(tmp_path, "2026-01-12", [1.0, 0.0, 1.0, 1.0])
        assert detect_drift(tmp_path, "2026-01-12") == []

    def test_regression_fires_score_drift_after_bh(self, tmp_path):
        record_week(tmp_path, "2026-01-05", [1.0] * 12)
        record_week(tmp_path, "2026-01-12", [0.0] * 8 + [1.0] * 4)
        events = detect_drift(tmp_path, "2026-01-12")
        kinds = {e.kind for e in events}
        assert "score_drift" in kinds
        ev = next(e for e in events if e.kind == "score_drift")
        assert ev.delta is not None and ev.delta < 0
        assert ev.p is not None and ev.p < 0.05
        assert ev.q is not None  # BH-adjusted q stamped on survivors
        assert "Significant after BH" in ev.summary

    def test_silent_swap_fingerprint_changed_scores_held(self, tmp_path):
        record_week(tmp_path, "2026-01-05", [1.0, 0.0, 1.0, 1.0],
                    fingerprint="fp-a")
        record_week(tmp_path, "2026-01-12", [1.0, 0.0, 1.0, 1.0],
                    fingerprint="fp-b")
        events = detect_drift(tmp_path, "2026-01-12")
        assert [e.kind for e in events] == ["silent_swap"]
        ev = events[0]
        assert ev.fingerprints_before == ["fp-a"]
        assert ev.fingerprints_after == ["fp-b"]

    def test_fingerprint_change_with_significant_drift(self, tmp_path):
        record_week(tmp_path, "2026-01-05", [1.0] * 12, fingerprint="fp-a")
        record_week(tmp_path, "2026-01-12", [0.0] * 8 + [1.0] * 4,
                    fingerprint="fp-b")
        events = detect_drift(tmp_path, "2026-01-12")
        kinds = {e.kind for e in events}
        assert "score_drift" in kinds
        assert "fingerprint_change" in kinds
        assert "silent_swap" not in kinds

    def test_rollout_event(self, tmp_path):
        record_week(tmp_path, "2026-01-05", [1.0] * 4)
        record_week(tmp_path, "2026-01-12", [1.0] * 4, rollout=True)
        events = detect_drift(tmp_path, "2026-01-12")
        assert any(e.kind == "rollout" for e in events)

    def test_panel_change_skips_paired_test(self, tmp_path):
        record_week(tmp_path, "2026-01-05", [1.0] * 12)
        record_week(tmp_path, "2026-01-12", [0.0] * 12,
                    expected=[f"new-answer-{i}" for i in range(12)])
        events = detect_drift(tmp_path, "2026-01-12")
        assert [e.kind for e in events] == ["panel_changed"]

    def test_transport_errors_do_not_read_as_drift(self, tmp_path):
        # 8 of 12 cases fail with provider errors (score 0.0). Those zeros
        # are outage, not behavior — pairing must exclude them.
        record_week(tmp_path, "2026-01-05", [1.0] * 12)
        errs: list[str | None] = ["HTTPStatusError: 500"] * 8 + [None] * 4
        record_week(tmp_path, "2026-01-12", [0.0] * 8 + [1.0] * 4, errors=errs)
        events = detect_drift(tmp_path, "2026-01-12")
        assert not any(e.kind == "score_drift" for e in events)

    def test_aborted_record_skipped_entirely(self, tmp_path):
        record_week(tmp_path, "2026-01-05", [1.0] * 4)
        errs: list[str | None] = ["boom"] * 3 + [None]
        record_week(tmp_path, "2026-01-12", [0.0] * 3 + [1.0], errors=errs)
        assert detect_drift(tmp_path, "2026-01-12") == []

    def test_regression_after_outage_week_still_detected(self, tmp_path):
        # Week 2 is a provider outage (aborted). A week-3 regression must be
        # compared against week 1 (the last clean observation), not silently
        # dropped because the immediate prior is unusable.
        record_week(tmp_path, "2026-01-05", [1.0] * 12)
        errs: list[str | None] = ["boom"] * 8 + [None] * 4
        record_week(tmp_path, "2026-01-12", [0.0] * 8 + [1.0] * 4, errors=errs)
        record_week(tmp_path, "2026-01-19", [0.0] * 8 + [1.0] * 4)
        events = detect_drift(tmp_path, "2026-01-19")
        ev = next(e for e in events if e.kind == "score_drift")
        assert "2026-01-05" in ev.summary  # paired against the clean week

    def test_fingerprint_change_without_any_test_is_not_silent_swap(
            self, tmp_path):
        # Panel changed AND fingerprint changed the same week: no paired
        # test ran, so the feed must not claim "the scores held".
        record_week(tmp_path, "2026-01-05", [1.0] * 6, fingerprint="fp-a")
        record_week(tmp_path, "2026-01-12", [1.0] * 6, fingerprint="fp-b",
                    expected=[f"new-{i}" for i in range(6)])
        events = detect_drift(tmp_path, "2026-01-12")
        kinds = {e.kind for e in events}
        assert "silent_swap" not in kinds
        assert "panel_changed" in kinds
        fp_ev = next(e for e in events if e.kind == "fingerprint_change")
        assert "UNKNOWN" in fp_ev.summary

    def test_cross_suite_fingerprint_split_is_a_rollout(self, tmp_path):
        # Two suites in the same pass served different fingerprints: the
        # pass straddles a rollout even though each run was internally
        # consistent (no per-run rollout flag).
        record_week(tmp_path, "2026-01-05", [1.0] * 4, suite="suite_a",
                    fingerprint="fp-old")
        record_week(tmp_path, "2026-01-05", [1.0] * 4, suite="suite_b",
                    fingerprint="fp-new")
        events = detect_drift(tmp_path, "2026-01-05")
        assert any(e.kind == "rollout" for e in events)

    def test_sycophancy_notice_with_mcnemar(self, tmp_path):
        record_week(tmp_path, "2026-01-05", [1.0] * 10,
                    pushback_scores=[1.0] * 10)
        record_week(tmp_path, "2026-01-12", [1.0] * 10,
                    pushback_scores=[0.0] * 6 + [1.0] * 4)
        events = detect_drift(tmp_path, "2026-01-12")
        notices = [e for e in events if e.kind == "notice"]
        assert len(notices) == 1
        assert "flip rate" in notices[0].summary
        assert notices[0].p is not None and notices[0].p < 0.05
        assert "not part of the gated comparison" in notices[0].summary

    def test_refusal_notice(self, tmp_path):
        record_week(tmp_path, "2026-01-05", [1.0] * 10)
        outs = ["I cannot help with that."] * 3 + [
            f"Answer: answer-{i}\nConfidence: 0.9" for i in range(3, 10)
        ]
        record_week(tmp_path, "2026-01-12", [0.0] * 3 + [1.0] * 7,
                    outputs=outs)
        events = detect_drift(tmp_path, "2026-01-12")
        assert any(e.kind == "notice" and "refusal" in e.summary
                   for e in events)

    def test_epoch_baseline_quoted_in_summary(self, tmp_path):
        record_week(tmp_path, "2026-01-05", [1.0] * 12)
        record_week(tmp_path, "2026-01-12", [1.0] * 12)
        record_week(tmp_path, "2026-01-19", [0.0] * 8 + [1.0] * 4)
        events = detect_drift(tmp_path, "2026-01-19")
        ev = next(e for e in events if e.kind == "score_drift")
        assert "Vs 2026-01-05 baseline" in ev.summary

    def test_bh_pools_across_endpoints(self, tmp_path):
        # Two endpoints, one real regression and one borderline mover: BH
        # runs over both p-values; the strong one must survive.
        for ep in ("ep-a", "ep-b"):
            record_week(tmp_path, "2026-01-05", [1.0] * 12, endpoint=ep)
        record_week(tmp_path, "2026-01-12", [0.0] * 10 + [1.0] * 2,
                    endpoint="ep-a")
        record_week(tmp_path, "2026-01-12", [0.0] * 2 + [1.0] * 10,
                    endpoint="ep-b")
        events = detect_drift(tmp_path, "2026-01-12")
        drifts = {e.endpoint for e in events if e.kind == "score_drift"}
        assert "ep-a" in drifts
        assert "ep-b" not in drifts  # 2 discordant pairs: p=0.5, never sig


class TestBudget:
    def _suite(self, n_chars: int = 400) -> SuiteConfig:
        return SuiteConfig(
            name="s", cases=[EvalCase(input="x" * n_chars, expected="y")]
        )

    def test_estimate_heuristic_from_suite(self):
        # gpt-4o: $2.50/M in, $10/M out. 400 chars → 100 in-tokens; 300 out.
        est = estimate_stage_cost("gpt-4o", self._suite())
        assert est == pytest.approx(100 * 2.5e-6 + 300 * 10e-6)

    def test_estimate_prefers_prior_observation(self):
        prior = {"input_tokens": 1_000_000, "output_tokens": 0}
        est = estimate_stage_cost("gpt-4o", self._suite(), prior_entry=prior)
        assert est == pytest.approx(2.50)

    def test_unknown_model_estimates_conservatively(self):
        # A hosted model missing from the pricing catalog also records $0
        # ACTUAL cost, so a $0 estimate would make the budget cap a no-op
        # exactly when prices are least known. Unknown models estimate at
        # the catalog maximum (and warn) instead.
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            est = estimate_stage_cost("my-local-llm", self._suite())
        assert est > 0.0
        assert any("pricing" in str(w.message) for w in caught)

    def test_riftlm_estimates_zero(self):
        # In-process checkpoints are genuinely free.
        assert estimate_stage_cost("riftlm:models/x.npz@abc", self._suite()) == 0.0

    def test_tracker_aborts_at_cap_and_stays_aborted(self):
        b = BudgetTracker(1.0)
        assert b.allows(0.4)
        b.add(0.5)
        assert b.allows(0.4)   # 0.5 + 0.4 <= 1.0
        b.add(0.4)
        assert not b.allows(0.2)  # would cross the cap
        assert b.aborted
        assert not b.allows(0.0)  # aborted is sticky


class TestReplay:
    def test_replay_pairs_pushback_and_dates_from_run(self, tmp_path):
        base = make_run([1.0, 1.0, 0.0], completed_at="2026-02-03T10:00:00Z")
        push = make_run([1.0, 0.0, 0.0], suite="panel_suite__pushback",
                        completed_at="2026-02-03T11:00:00Z")
        base_p, push_p = tmp_path / "base.json", tmp_path / "push.json"
        base.save(base_p)
        push.save(push_p)
        records = replay_panel([base_p, push_p])
        assert len(records) == 1
        rec = records[0]
        assert rec.endpoint == "panel-model"
        assert rec.date == "2026-02-03"
        assert rec.derived["sycophancy"]["flip_rate"] == 0.5

    def test_replay_date_override(self, tmp_path):
        run = make_run([1.0])
        p = tmp_path / "run.json"
        run.save(p)
        records = replay_panel([p], date="2026-03-01")
        assert records[0].date == "2026-03-01"


class TestPanelConfig:
    def test_load_panel(self):
        panel = load_panel(Path(__file__).parent.parent
                           / "observatory" / "panel.yaml")
        # Panel v2 (Fable 5.1 launch): three Anthropic tiers side by side
        # plus the two non-Anthropic frontier endpoints.
        assert [ep.id for ep in panel.endpoints] == [
            "claude-fable-5-1", "claude-opus-5", "claude-sonnet-5",
            "gpt-5.5", "gemini-3.5-flash",
        ]
        assert "hard_reasoning" in panel.suites
        assert panel.sycophancy_on in panel.suites
        assert panel.max_cost_usd == 6.0

    def test_panel_validation(self, tmp_path):
        bad = tmp_path / "panel.yaml"
        bad.write_text("endpoints:\n  - id: m\nsuites: [a]\nsycophancy_on: b\n")
        with pytest.raises(ValueError, match="sycophancy_on"):
            load_panel(bad)
        bad.write_text("endpoints: []\nsuites: [a]\n")
        with pytest.raises(ValueError, match="no endpoints"):
            load_panel(bad)
