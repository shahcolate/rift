"""Rift Observatory: a longitudinal public record of model behavior.

`rift compare` answers "did THIS upgrade break anything?" once. The
observatory asks a different question on a schedule: **has the model
behind this endpoint changed since last week — and would anyone have
told you?** It runs a small fixed panel (accuracy suites + behavioral
probes) against pinned live endpoints, appends each observation to an
on-disk data directory, and runs the same paired statistics `compare`
uses against the previous observation. Fingerprint changes are tracked
independently of scores, so a silent server-side swap shows up even
when the scores hold.

Design constraints, in order:

1. **Append-only.** Observations and events are never rewritten;
   the longitudinal record is the product. ``index.jsonl`` and
   ``events.jsonl`` only grow.
2. **Strip-io by default.** Stored records keep per-case scores,
   tags, costs, and errors (everything the paired tests need) but not
   prompt/completion text. Anything that must be derived from raw
   output text (confidence parse, refusal classification, sycophancy
   flips) is computed *before* stripping and stored in the record's
   ``derived`` block.
3. **Budget-capped.** A scheduled run must never surprise anyone's
   bill. Stage-level pre-flight estimates against :mod:`rift.pricing`
   abort remaining stages when the cap would be exceeded; partial
   data is still committed.
4. **Pairing-guarded.** Every record carries a ``panel_version`` hash
   over the suite's ``(input, expected)`` pairs. The paired test only
   runs when the two observations hash identically; otherwise a
   ``panel_changed`` event is emitted instead of a bogus comparison.

Data layout (one directory, typically an orphan git branch)::

    index.jsonl                                # 1 line per (date, endpoint, suite)
    events.jsonl                               # append-only drift feed
    records/<date>/<endpoint-slug>/<suite>.json
    selftest/<endpoint-slug>.json              # latest SelfTestResult per endpoint
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
import yaml

from .calibration import compute_calibration, parse_confidence
from .comparator import benjamini_hochberg, compare_runs
from .config import SuiteConfig, load_suite, resolve_model
from .pricing import PRICING, lookup
from .refusal import classify_output
from .runner import RunResult, run_suite
from .sycophancy import PUSHBACK_SUITE_SUFFIX, build_pushback_suite

RECORD_SCHEMA = 1


class DuplicateObservationError(click.ClickException):
    """A (date, endpoint, suite) observation already exists in the data dir.

    Subclasses ``click.ClickException`` (repo convention, see
    ``config.SuiteNotFoundError``) so a same-date re-run produces a clean
    one-line message and exit 1, never a traceback.
    """

    exit_code = 1

# Probe shifts past these thresholds emit a (non-gated) ``notice`` event.
# Deliberately coarse: probe metrics on a small panel are noisy, and the
# notice tier exists to point a human at a chart, not to gate anything.
NOTICE_THRESHOLD = 0.10

# Paired comparisons need at least this many mutually error-free cases;
# below it the test is pure noise and we skip rather than mislead.
MIN_PAIRED_CASES = 3

# Output-token planning heuristic for a stage with no prior observation:
# ~4 chars/token on input, plus a flat per-case output allowance.
EST_OUTPUT_TOKENS_PER_CASE = 300


# ---------------------------------------------------------------------------
# Panel configuration
# ---------------------------------------------------------------------------


@dataclass
class EndpointSpec:
    """One endpoint under observation.

    ``id`` is the stable name observations are filed under (it survives
    alias re-pointing — that is the point); ``model`` is whatever
    :func:`rift.config.resolve_model` should route, alias or dated id.
    ``epoch_baseline`` optionally pins the long-run comparison record to a
    specific observation date; default is the endpoint's first observation.
    """

    id: str
    model: str
    epoch_baseline: str | None = None


@dataclass
class PanelConfig:
    """The observatory panel: what runs, against what, under what cap."""

    endpoints: list[EndpointSpec]
    suites: list[str]
    sycophancy_on: str | None = None
    max_cost_usd: float = 3.0
    alpha: float = 0.05


def load_panel(path: str | Path) -> PanelConfig:
    """Load and validate a panel YAML."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    endpoints = [
        EndpointSpec(
            id=e["id"],
            model=e.get("model", e["id"]),
            epoch_baseline=e.get("epoch_baseline"),
        )
        for e in data.get("endpoints", [])
    ]
    if not endpoints:
        raise ValueError(f"Panel {path} defines no endpoints.")
    suites = list(data.get("suites", []))
    if not suites:
        raise ValueError(f"Panel {path} defines no suites.")
    sycophancy_on = data.get("sycophancy_on")
    if sycophancy_on is not None and sycophancy_on not in suites:
        raise ValueError(
            f"Panel {path}: sycophancy_on={sycophancy_on!r} is not one of "
            f"the panel suites {suites}."
        )
    return PanelConfig(
        endpoints=endpoints,
        suites=suites,
        sycophancy_on=sycophancy_on,
        max_cost_usd=float(data.get("max_cost_usd", 3.0)),
        alpha=float(data.get("alpha", 0.05)),
    )


def endpoint_slug(endpoint_id: str) -> str:
    """Filesystem-safe slug for an endpoint id."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", endpoint_id)


def panel_version_hash(pairs: list[tuple[str, Any]]) -> str:
    """Pairing-validity hash over a suite's canonical ``(input, expected)``.

    Two observations are only paired-comparable when this hash matches —
    same cases, same order, same expectations. Computed from the pairs
    rather than the YAML file so a comment edit doesn't invalidate history.
    """
    h = hashlib.sha256()
    for inp, expected in pairs:
        h.update(inp.encode())
        h.update(b"\x1f")
        h.update(json.dumps(expected, sort_keys=True, default=str).encode())
        h.update(b"\x1e")
    return h.hexdigest()[:16]


def _suite_panel_version(suite: SuiteConfig) -> str:
    return panel_version_hash([(c.input, c.expected) for c in suite.cases])


def _run_panel_version(run: RunResult) -> str:
    """Panel hash from a full (non-stripped) RunResult — replay path."""
    return panel_version_hash([(c.input_text, c.expected) for c in run.cases])


# ---------------------------------------------------------------------------
# Observation records
# ---------------------------------------------------------------------------


# derived-block scalars copied verbatim onto the index line. One tuple to
# extend when build_record gains a metric the dashboard should chart —
# _metric_by_date in observatory_site.py reads the index, not the records.
_INDEX_SCALAR_KEYS = (
    "mean_score", "n_cases", "n_errors", "fingerprints",
    "fingerprint_rollout", "cost_usd", "input_tokens", "output_tokens",
    "cost_per_correct", "aborted",
)


@dataclass
class ObservationRecord:
    """One (date, endpoint, suite) observation: stripped run + derived block."""

    date: str
    endpoint: str
    model: str
    suite: str
    panel_version: str
    derived: dict
    run: dict          # RunResult.to_dict(strip_io=True)
    schema: int = RECORD_SCHEMA

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ObservationRecord":
        known = cls.__dataclass_fields__  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def index_entry(self, record_path: str) -> dict:
        """The compact ``index.jsonl`` line for this record."""
        d = self.derived
        cal = d.get("calibration") or {}
        syc = d.get("sycophancy") or {}
        return {
            "date": self.date,
            "endpoint": self.endpoint,
            "model": self.model,
            "suite": self.suite,
            "panel_version": self.panel_version,
            **{k: d[k] for k in _INDEX_SCALAR_KEYS},
            "brier": cal.get("brier"),
            "ece": cal.get("ece"),
            "refusal_rate": (d.get("refusal") or {}).get("rate"),
            "flip_rate": syc.get("flip_rate"),
            "pushback_input_tokens": syc.get("pushback_input_tokens"),
            "pushback_output_tokens": syc.get("pushback_output_tokens"),
            "record": record_path,
        }


def finite_or_none(v) -> float | None:
    """Coerce to a finite float or None.

    The shared sanitizer for everything read back from the data dir:
    records round-trip through ``json.dump(default=str)``, so a value can
    come back as None, a string, inf, or NaN depending on rift version.
    """
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def build_record(
    run: RunResult,
    endpoint_id: str,
    date: str,
    panel_version: str | None = None,
    pushback_run: RunResult | None = None,
) -> ObservationRecord:
    """Build an observation record from a *non-stripped* run.

    Everything that needs raw output text — confidence parse, refusal
    classification, sycophancy flips — is derived here, then the stored
    run dict is stripped. ``pushback_run`` is the sycophancy follow-up
    produced by :func:`rift.sycophancy.build_pushback_suite`, when the
    panel runs that probe on this suite.
    """
    n_errors = sum(1 for c in run.cases if c.error)
    n_cases = len(run.cases)

    cal = compute_calibration(run)
    calibration = None
    if cal.n_parsed > 0:
        calibration = {
            "brier": cal.brier,
            "ece": cal.ece,
            "overconfidence": cal.overconfidence,
            "n_parsed": cal.n_parsed,
            "n_unparsed": cal.n_unparsed,
            # Per-case vector survives strip-io so future probes can re-pair.
            "confidences": [parse_confidence(c.output or "") for c in run.cases],
        }

    refusal_flags = [classify_output(c.output or "")[0] for c in run.cases]
    refusal = {
        "rate": round(sum(refusal_flags) / n_cases, 4) if n_cases else 0.0,
        "flags": refusal_flags,
    }

    sycophancy = None
    if pushback_run is not None:
        # Flips are computed only over pairs where neither leg errored: an
        # exhausted-retry pushback completion scores 0.0 for transport
        # reasons and must not read as the model caving under pressure.
        orig_correct = [1 if c.score >= 0.999 else 0 for c in run.cases]
        push_correct = [1 if c.score >= 0.999 else 0
                        for c in pushback_run.cases]
        valid = [
            0 if (b.error or p.error) else 1
            for b, p in zip(run.cases, pushback_run.cases)
        ]
        eligible = [i for i, v in enumerate(valid) if v and orig_correct[i]]
        flipped = [i for i in eligible if not push_correct[i]]
        wrong = [i for i, v in enumerate(valid) if v and not orig_correct[i]]
        recovered = [i for i in wrong if push_correct[i]]
        sycophancy = {
            "flip_rate": round(len(flipped) / len(eligible), 4) if eligible else 0.0,
            "recovery_rate": round(len(recovered) / len(wrong), 4) if wrong else 0.0,
            "n_originally_correct": len(eligible),
            "n_flipped_to_wrong": len(flipped),
            # Per-case 0/1 vectors so a later observation can run McNemar on
            # the held-under-pushback outcome without the raw outputs.
            # ``valid`` marks pairs where neither leg errored — flips are
            # only defined on those.
            "orig_correct": orig_correct,
            "push_correct": push_correct,
            "valid": valid,
            "pushback_cost_usd": round(pushback_run.total_cost_usd, 6),
            # Probe-stage token counts, surfaced so next pass's budget
            # pre-flight can estimate the pushback stage from its own
            # workload rather than the (shorter) base prompts.
            "pushback_input_tokens": pushback_run.total_input_tokens,
            "pushback_output_tokens": pushback_run.total_output_tokens,
        }

    # ``cost_usd`` is the full stage spend including the probe follow-up —
    # the number the budget tracker and the spend report account against.
    # Token counts stay base-run-only (they parameterize the base-stage
    # budget estimate); the probe's tokens live in the sycophancy block.
    pushback_cost = pushback_run.total_cost_usd if pushback_run else 0.0
    derived = {
        "mean_score": round(run.mean_score, 4),
        "n_cases": n_cases,
        "n_errors": n_errors,
        "cost_usd": round(run.total_cost_usd + pushback_cost, 6),
        "input_tokens": run.total_input_tokens,
        "output_tokens": run.total_output_tokens,
        "cost_per_correct": finite_or_none(round(run.cost_per_correct(), 6)),
        "fingerprints": run.metadata.get("fingerprints", []),
        "fingerprint_rollout": bool(run.metadata.get("fingerprint_rollout", False)),
        "calibration": calibration,
        "refusal": refusal,
        "sycophancy": sycophancy,
        # A majority-errored run (provider outage mid-stage) is recorded for
        # the audit trail but excluded from paired stats — its zeros are
        # transport failures, not model behavior.
        "aborted": bool(n_cases and n_errors * 2 >= n_cases),
        "rift_version": _rift_version(),
    }

    return ObservationRecord(
        date=date,
        endpoint=endpoint_id,
        model=run.model,
        suite=run.suite_name,
        panel_version=panel_version or _run_panel_version(run),
        derived=derived,
        run=run.to_dict(strip_io=True),
    )


def _rift_version() -> str:
    from . import __version__
    return __version__


# ---------------------------------------------------------------------------
# Data directory I/O (append-only)
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file, tolerating a truncated final line.

    Appends aren't atomic (unlike the tmp+rename record files), so a
    crash mid-append can leave a partial last line. Refusing to load
    would wedge every future pass on a one-line repair; instead the
    partial line is dropped with a warning. A corrupt line anywhere
    else is real damage and still raises.
    """
    if not path.exists():
        return []
    out: list[dict] = []
    lines = [ln.strip() for ln in path.read_text().splitlines()]
    lines = [ln for ln in lines if ln]
    for i, line in enumerate(lines):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                import warnings

                warnings.warn(
                    f"{path}: dropping truncated final line (crashed "
                    "append?) — the affected observation is lost but the "
                    "series stays loadable.",
                    stacklevel=2,
                )
                break
            raise
    return out


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


def load_index(data_dir: str | Path) -> list[dict]:
    return _read_jsonl(Path(data_dir) / "index.jsonl")


def load_events(data_dir: str | Path) -> list[dict]:
    return _read_jsonl(Path(data_dir) / "events.jsonl")


def load_record(data_dir: str | Path, index_entry: dict) -> ObservationRecord:
    path = Path(data_dir) / index_entry["record"]
    with open(path) as f:
        return ObservationRecord.from_dict(json.load(f))


def load_selftest(data_dir: str | Path, endpoint_id: str) -> dict | None:
    """Latest stored SelfTestResult dict for an endpoint, if any."""
    path = Path(data_dir) / "selftest" / f"{endpoint_slug(endpoint_id)}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def append_records(records: list[ObservationRecord],
                   data_dir: str | Path) -> list[dict]:
    """Write record files and append their index lines. Returns the lines.

    Refuses to overwrite an existing record — the data directory is
    append-only; a re-run of the same date must be deliberate (remove that
    date's records and index/events lines first) rather than a silent
    history rewrite. Every path is validated *before* anything is written,
    so a duplicate never leaves earlier records of the same batch orphaned
    on disk without index lines (a state that would block all future
    passes for that date).
    """
    data_dir = Path(data_dir)
    planned: list[tuple[ObservationRecord, str]] = []
    seen: set[str] = set()
    for rec in records:
        rel = (
            f"records/{rec.date}/{endpoint_slug(rec.endpoint)}/"
            f"{rec.suite}.json"
        )
        if rel in seen or (data_dir / rel).exists():
            raise DuplicateObservationError(
                f"Observation already recorded: {data_dir / rel}. The "
                f"observatory data dir is append-only; refusing to "
                f"overwrite. To deliberately re-record this date, remove "
                f"its records/ files and index/events lines first."
            )
        seen.add(rel)
        planned.append((rec, rel))

    entries: list[dict] = []
    for rec, rel in planned:
        path = data_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        try:
            with open(tmp, "w") as f:
                json.dump(rec.to_dict(), f, indent=2, default=str)
            tmp.replace(path)
        except BaseException:
            # Never leave a half-written temp inside the append-only dir —
            # the weekly job's `git add -A` would commit it forever.
            tmp.unlink(missing_ok=True)
            raise
        entries.append(rec.index_entry(rel))
    _append_jsonl(data_dir / "index.jsonl", entries)
    return entries


def append_events(events: list["DriftEvent"], data_dir: str | Path) -> None:
    _append_jsonl(Path(data_dir) / "events.jsonl",
                  [e.to_dict() for e in events])


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------


@dataclass
class DriftEvent:
    """One line of the public drift feed."""

    date: str
    endpoint: str
    kind: str          # score_drift | fingerprint_change | silent_swap |
                       # rollout | panel_changed | notice
    suite: str = ""    # empty for endpoint-level (fingerprint) events
    delta: float | None = None
    p: float | None = None
    q: float | None = None          # BH-adjusted, score_drift only
    ci: list[float] | None = None
    fingerprints_before: list[str] = field(default_factory=list)
    fingerprints_after: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _paired_case_indices(prev_run: dict, curr_run: dict) -> list[int]:
    """Indices comparable across two stored runs: neither side errored.

    Errored cases carry score 0.0 from a transport failure, not from the
    model; pairing them would manufacture drift out of an outage.
    """
    prev_cases = prev_run["cases"]
    curr_cases = curr_run["cases"]
    n = min(len(prev_cases), len(curr_cases))
    return [
        i for i in range(n)
        if not prev_cases[i].get("error") and not curr_cases[i].get("error")
    ]


def _latest_prior(entries: list[dict], date: str) -> dict | None:
    """Latest *usable* index entry strictly before ``date`` (entries
    pre-filtered to one (endpoint, suite) group).

    Aborted (majority-errored) entries are skipped, not just rejected: a
    regression landing the week after an outage must still be compared
    against the last clean observation, or it would never enter the feed.
    Relies on ISO dates sorting lexically.
    """
    prior = [e for e in entries if e["date"] < date and not e.get("aborted")]
    return max(prior, key=lambda e: e["date"]) if prior else None


def _epoch_entry(entries: list[dict], date: str,
                 pinned_date: str | None) -> dict | None:
    """The long-run comparison record: pinned date if present (and clean),
    else the group's first non-aborted observation. None when it would
    equal the current date. Aborted entries are excluded — an outage week
    must never become the baseline the feed quotes deltas against."""
    prior = sorted(
        (e for e in entries if e["date"] < date and not e.get("aborted")),
        key=lambda e: e["date"],
    )
    if not prior:
        return None
    if pinned_date:
        pinned = [e for e in prior if e["date"] == pinned_date]
        if pinned:
            return pinned[0]
    return prior[0]


def detect_drift(
    data_dir: str | Path,
    date: str,
    alpha: float = 0.05,
    epoch_baselines: dict[str, str] | None = None,
) -> list[DriftEvent]:
    """Compare ``date``'s observations against the previous ones.

    Per (endpoint, suite): a paired test (the same
    :func:`rift.comparator.compare_runs` the CI gate uses) against the
    most recent prior observation, gated by ``panel_version`` equality.
    All primary p-values from this date are pooled through
    Benjamini–Hochberg; a ``score_drift`` event fires only when the
    BH-adjusted q clears ``alpha`` — a weekly run across endpoints ×
    suites is exactly the multiplicity setting BH exists for.

    Per endpoint: the union of server fingerprints across the date's
    suites is diffed against the prior date's union. A change with no
    BH-significant score drift anywhere on the endpoint is a
    ``silent_swap`` — the served model changed and the panel couldn't
    tell from scores alone. Probe metrics (sycophancy flip rate,
    ECE/Brier, refusal rate) emit non-gated ``notice`` events past
    :data:`NOTICE_THRESHOLD`.

    ``epoch_baselines`` maps endpoint id → pinned record date for the
    long-run comparison quoted in summaries (reported, never gated).
    """
    data_dir = Path(data_dir)
    index = load_index(data_dir)
    today = [e for e in index if e["date"] == date]
    if not today:
        return []
    epoch_baselines = epoch_baselines or {}

    events: list[DriftEvent] = []
    # Primary tests pooled for BH at the end; each event carries its endpoint.
    primary: list[DriftEvent] = []
    # Endpoints where at least one paired test actually ran this date —
    # _fingerprint_events must not claim "scores held" for endpoints whose
    # comparisons were all skipped (panel change, abort, too few cases).
    tested_endpoints: set[str] = set()

    by_group: dict[tuple[str, str], list[dict]] = {}
    for e in index:
        by_group.setdefault((e["endpoint"], e["suite"]), []).append(e)

    for entry in today:
        endpoint, suite = entry["endpoint"], entry["suite"]
        group = by_group[(endpoint, suite)]
        prev = _latest_prior(group, date)
        if prev is None:
            continue
        if entry.get("aborted"):
            continue  # transport-failure zeros must not read as drift
        if entry["panel_version"] != prev["panel_version"]:
            events.append(DriftEvent(
                date=date, endpoint=endpoint, suite=suite, kind="panel_changed",
                summary=(
                    f"{endpoint}/{suite}: panel changed since {prev['date']} "
                    f"({prev['panel_version']} → {entry['panel_version']}); "
                    f"paired comparison skipped. Longitudinal history restarts "
                    f"at this observation."
                ),
            ))
            continue

        prev_rec = load_record(data_dir, prev)
        curr_rec = load_record(data_dir, entry)
        idx = _paired_case_indices(prev_rec.run, curr_rec.run)
        if len(idx) < MIN_PAIRED_CASES:
            continue
        prev_cases = prev_rec.run["cases"]
        curr_cases = curr_rec.run["cases"]
        drift = compare_runs(
            baseline_scores=[float(prev_cases[i]["score"]) for i in idx],
            challenger_scores=[float(curr_cases[i]["score"]) for i in idx],
            baseline_model=f"{endpoint}@{prev['date']}",
            challenger_model=f"{endpoint}@{date}",
            suite_name=suite,
            alpha=alpha,
            baseline_costs=[float(prev_cases[i]["cost_usd"]) for i in idx],
            challenger_costs=[float(curr_cases[i]["cost_usd"]) for i in idx],
        )

        tested_endpoints.add(endpoint)

        # Long-run note: same paired, error-excluded basis as the headline —
        # a raw mean diff would count an errored case's 0.0 as behavior and
        # publish outage noise in the same sentence as the careful test.
        epoch = _epoch_entry(group, date, epoch_baselines.get(endpoint))
        epoch_note = ""
        if epoch is not None and epoch["date"] != prev["date"] \
                and epoch["panel_version"] == entry["panel_version"]:
            epoch_rec = load_record(data_dir, epoch)
            eidx = _paired_case_indices(epoch_rec.run, curr_rec.run)
            if len(eidx) >= MIN_PAIRED_CASES:
                e_cases = epoch_rec.run["cases"]
                epoch_delta = (
                    sum(float(curr_cases[i]["score"]) for i in eidx)
                    - sum(float(e_cases[i]["score"]) for i in eidx)
                ) / len(eidx)
                epoch_note = (
                    f" Vs {epoch['date']} baseline: {epoch_delta:+.4f} "
                    f"(paired, n={len(eidx)})."
                )

        cost_note = ""
        if drift.cost_delta_ci_defined:
            cost_note = (
                f" $/correct {drift.baseline_cost_per_correct:.4f} → "
                f"{drift.challenger_cost_per_correct:.4f}."
            )
        ev = DriftEvent(
            date=date, endpoint=endpoint, suite=suite, kind="score_drift",
            delta=drift.delta, p=drift.p_value,
            ci=[drift.ci_lower, drift.ci_upper],
            # The summary names the comparison date explicitly: when an
            # outage week was skipped, "previous observation" is NOT last
            # week and the feed must say so.
            summary=(
                f"{endpoint}/{suite} vs {prev['date']}: "
                f"{drift.baseline_mean:.4f} → "
                f"{drift.challenger_mean:.4f} ({drift.delta:+.4f}, "
                f"{drift.test_used}, p={drift.p_value:.4g}, "
                f"n={len(idx)}).{cost_note}{epoch_note}"
            ),
        )
        primary.append(ev)

        events.extend(_probe_notices(date, endpoint, suite,
                                     prev_rec, curr_rec, idx))

    # BH across every primary test this date; only survivors become events.
    sig_endpoints: set[str] = set()
    if primary:
        qs, rejected = benjamini_hochberg(
            [ev.p for ev in primary], alpha=alpha  # type: ignore[misc]
        )
        for ev, q, rej in zip(primary, qs, rejected):
            if rej:
                ev.q = round(q, 6)
                ev.summary += f" Significant after BH (q={q:.4g})."
                events.append(ev)
                sig_endpoints.add(ev.endpoint)

    events.extend(_fingerprint_events(index, today, date,
                                      sig_endpoints, tested_endpoints))
    return events


def _probe_notices(date: str, endpoint: str, suite: str,
                   prev_rec: ObservationRecord, curr_rec: ObservationRecord,
                   paired_idx: list[int]) -> list[DriftEvent]:
    """Non-gated notice events for probe-metric shifts past the threshold."""
    out: list[DriftEvent] = []
    prev_d, curr_d = prev_rec.derived, curr_rec.derived

    prev_s, curr_s = prev_d.get("sycophancy"), curr_d.get("sycophancy")
    if prev_s and curr_s:
        flip_delta = curr_s["flip_rate"] - prev_s["flip_rate"]
        if abs(flip_delta) >= NOTICE_THRESHOLD:
            # McNemar on held-under-pushback, restricted to cases control-
            # correct AND error-free on both legs in BOTH observations (the
            # only ones a flip is defined on). Records written before the
            # ``valid`` vector existed are treated as all-valid.
            prev_v = prev_s.get("valid") or [1] * len(prev_s["orig_correct"])
            curr_v = curr_s.get("valid") or [1] * len(curr_s["orig_correct"])
            shared = [
                i for i in paired_idx
                if i < len(prev_s["orig_correct"]) and i < len(curr_s["orig_correct"])
                and prev_s["orig_correct"][i] and curr_s["orig_correct"][i]
                and prev_v[i] and curr_v[i]
            ]
            p = None
            if len(shared) >= MIN_PAIRED_CASES:
                d = compare_runs(
                    baseline_scores=[float(prev_s["push_correct"][i]) for i in shared],
                    challenger_scores=[float(curr_s["push_correct"][i]) for i in shared],
                    baseline_model=endpoint, challenger_model=endpoint,
                    suite_name=f"{suite}(pushback)", bootstrap_n=0,
                )
                p = d.p_value
            out.append(DriftEvent(
                date=date, endpoint=endpoint, suite=suite, kind="notice",
                delta=round(flip_delta, 4), p=p,
                summary=(
                    f"{endpoint}/{suite}: sycophancy flip rate "
                    f"{prev_s['flip_rate']:.0%} → {curr_s['flip_rate']:.0%}"
                    + (f" (McNemar p={p:.4g} on {len(shared)} shared "
                       f"control-correct cases)" if p is not None else "")
                    + ". Notice only — not part of the gated comparison."
                ),
            ))

    prev_c, curr_c = prev_d.get("calibration"), curr_d.get("calibration")
    if prev_c and curr_c:
        for metric in ("ece", "brier"):
            pv, cv = prev_c.get(metric), curr_c.get(metric)
            if pv is None or cv is None:
                continue
            if math.isnan(pv) or math.isnan(cv):
                continue
            delta = cv - pv
            if abs(delta) >= NOTICE_THRESHOLD:
                out.append(DriftEvent(
                    date=date, endpoint=endpoint, suite=suite, kind="notice",
                    delta=round(delta, 4),
                    summary=(
                        f"{endpoint}/{suite}: {metric.upper()} "
                        f"{pv:.3f} → {cv:.3f} ({delta:+.3f}). Notice only."
                    ),
                ))

    prev_r = (prev_d.get("refusal") or {}).get("rate")
    curr_r = (curr_d.get("refusal") or {}).get("rate")
    if prev_r is not None and curr_r is not None:
        delta = curr_r - prev_r
        if abs(delta) >= NOTICE_THRESHOLD:
            out.append(DriftEvent(
                date=date, endpoint=endpoint, suite=suite, kind="notice",
                delta=round(delta, 4),
                summary=(
                    f"{endpoint}/{suite}: refusal rate {prev_r:.0%} → "
                    f"{curr_r:.0%} ({delta:+.0%}). Notice only."
                ),
            ))
    return out


def _fingerprint_events(index: list[dict], today: list[dict], date: str,
                        sig_endpoints: set[str],
                        tested_endpoints: set[str]) -> list[DriftEvent]:
    """Endpoint-level fingerprint diff vs the prior observation date.

    ``sig_endpoints`` are endpoints with a BH-significant score test this
    date; ``tested_endpoints`` are endpoints where at least one paired test
    actually ran. The distinction matters: ``silent_swap``'s claim that
    "the scores held" is only honest when scores were compared at all.
    """
    out: list[DriftEvent] = []
    endpoints = sorted({e["endpoint"] for e in today})
    for endpoint in endpoints:
        mine = [e for e in index if e["endpoint"] == endpoint]
        curr = [e for e in mine if e["date"] == date]
        prior_dates = sorted({e["date"] for e in mine if e["date"] < date})
        curr_fps = sorted({fp for e in curr for fp in e.get("fingerprints", [])})

        # A rollout is any pass whose scores straddle two snapshots: either
        # one run saw >1 fingerprint (the runner's flag), or different
        # suites in the same pass were served different fingerprints.
        if any(e.get("fingerprint_rollout") for e in curr) or len(curr_fps) > 1:
            out.append(DriftEvent(
                date=date, endpoint=endpoint, kind="rollout",
                fingerprints_after=curr_fps,
                summary=(
                    f"{endpoint}: served snapshot changed *during* this "
                    f"pass ({len(curr_fps)} fingerprints: "
                    f"{', '.join(curr_fps)}). Scores straddle a rollout and "
                    f"are not internally comparable."
                ),
            ))

        if not prior_dates or not curr_fps:
            continue
        prev_entries = [e for e in mine if e["date"] == prior_dates[-1]]
        prev_fps = sorted({fp for e in prev_entries
                           for fp in e.get("fingerprints", [])})
        if not prev_fps or prev_fps == curr_fps:
            continue
        if endpoint in sig_endpoints:
            kind = "fingerprint_change"
            tail = ("Score drift on this endpoint is significant this date — "
                    "see the score_drift entries.")
        elif endpoint in tested_endpoints:
            kind = "silent_swap"
            # Careful wording: "not significant" is NOT "scores held" — on a
            # small panel a real drop can sit under the test's minimum
            # detectable effect. Claim only what the test supports.
            tail = ("The model changed under the alias and no statistically "
                    "significant score change was detected — which on a "
                    "small panel may reflect limited power, not stability; "
                    "check the endpoint's accuracy chart. A request-keyed "
                    "cache or an accuracy-only check would never see this "
                    "change at all.")
        else:
            kind = "fingerprint_change"
            tail = ("No paired score comparison ran for this endpoint this "
                    "date (panel changed, record aborted, or too few "
                    "comparable cases), so whether behavior moved is "
                    "UNKNOWN — this is not a verified silent swap.")
        out.append(DriftEvent(
            date=date, endpoint=endpoint, kind=kind,
            fingerprints_before=prev_fps, fingerprints_after=curr_fps,
            summary=(
                f"{endpoint}: server fingerprint changed since "
                f"{prior_dates[-1]} ({', '.join(prev_fps)} → "
                f"{', '.join(curr_fps)}). {tail}"
            ),
        ))
    return out


# ---------------------------------------------------------------------------
# Budget guard
# ---------------------------------------------------------------------------


def estimate_stage_cost(model: str, suite: SuiteConfig,
                        prior_entry: dict | None = None) -> float:
    """Pre-flight USD estimate for running ``suite`` against ``model``.

    Prefers the endpoint's most recent observed token counts for the same
    suite (the best predictor of next week's run); falls back to a
    chars/4 input heuristic plus a flat output allowance.

    A model with no pricing entry estimates at the CATALOG MAXIMUM, not
    at $0: a hosted model missing from ``pricing.PRICING`` (new release,
    renamed id) also records $0 *actual* cost, so a $0 estimate would
    turn the hard budget cap into a no-op exactly when prices are least
    known. The conservative estimate makes the guard trip early and
    loudly instead — add the real price to the catalog to unblock.
    RiftLM checkpoints are genuinely free (in-process) and estimate 0.
    """
    if model.startswith("riftlm:"):
        return 0.0
    price = lookup(model)
    if price is None:
        price = max(PRICING.values(),
                    key=lambda p: p.cost(1_000_000, 1_000_000))
        import warnings

        warnings.warn(
            f"No pricing entry for {model!r}; the budget guard is "
            "estimating at the catalog's most expensive rate. Add the "
            "model to rift/pricing.py PRICING for accurate budgeting.",
            stacklevel=2,
        )
    if prior_entry and prior_entry.get("input_tokens"):
        return price.cost(prior_entry["input_tokens"],
                          prior_entry["output_tokens"])
    est_in = sum(len(c.input) // 4 for c in suite.cases)
    est_out = EST_OUTPUT_TOKENS_PER_CASE * len(suite.cases)
    return price.cost(est_in, est_out)


class BudgetTracker:
    """Stage-level budget gate: estimate before, accumulate actual after.

    Note the asymmetry by design: estimates gate *upcoming* stages, while
    ``spent`` accrues the *recorded* cost of completed ones. Cached
    completions still report their token-derived cost, so a fully-cached
    re-run can "spend" the budget on paper without a single API call —
    conservative, and the right bias for an unattended scheduled job.
    """

    def __init__(self, max_cost_usd: float):
        self.max_cost_usd = max_cost_usd
        self.spent = 0.0
        self.aborted = False
        # "endpoint/suite" stages skipped after the cap tripped — surfaced
        # in the pass report so a longitudinal gap is legible as a budget
        # decision, not a mystery hole in the series.
        self.skipped: list[str] = []

    def allows(self, estimate: float) -> bool:
        if self.aborted:
            return False
        if self.spent + estimate > self.max_cost_usd:
            self.aborted = True
            return False
        return True

    def add(self, actual: float) -> None:
        self.spent += actual


# ---------------------------------------------------------------------------
# Panel execution
# ---------------------------------------------------------------------------


def _utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


async def run_panel(
    panel: PanelConfig,
    data_dir: str | Path,
    date: str | None = None,
    cache_dir: str | None = None,
    concurrency: int = 5,
    max_cost_usd: float | None = None,
    endpoints: list[str] | None = None,
    show_progress: bool = True,
) -> tuple[list[ObservationRecord], BudgetTracker]:
    """Run the live observatory panel. Returns (records, budget).

    Per-endpoint failures are isolated: an endpoint whose provider raises
    (beyond the runner's own retries) is skipped with its partial stages
    intact, so one outage never loses the date's other observations.
    ``MissingAPIKeyError`` is NOT isolated — it is user-fixable and the
    whole run is wrong without it.
    """
    from .providers import MissingAPIKeyError

    date = date or _utc_today()
    budget = BudgetTracker(
        panel.max_cost_usd if max_cost_usd is None else max_cost_usd
    )
    prior_index = load_index(data_dir)

    def _prior_entry(endpoint_id: str, suite: SuiteConfig,
                     version: str) -> dict | None:
        # Keyed on the suite's OWN name (panel keys are file stems, which can
        # differ — e.g. panel "extraction" loads suite "structured_extraction")
        # and on panel_version, so a panel edit can't price next week's run
        # from the old panel's token counts.
        mine = [e for e in prior_index
                if e["endpoint"] == endpoint_id and e["suite"] == suite.name
                and not e.get("aborted")
                and e.get("panel_version") == version]
        return max(mine, key=lambda e: e["date"]) if mine else None

    suite_configs = {name: load_suite(name) for name in panel.suites}
    suite_versions = {
        name: _suite_panel_version(cfg) for name, cfg in suite_configs.items()
    }
    selected = [
        ep for ep in panel.endpoints
        if endpoints is None or ep.id in endpoints
    ]

    records: list[ObservationRecord] = []
    for ep in selected:
        try:
            # Inside the isolation block: resolve_model can now raise (a
            # riftlm endpoint whose checkpoint is absent on this runner),
            # and one bad endpoint must not abort the whole panel pass.
            model_config = resolve_model(ep.model)
            for suite_name in panel.suites:
                suite = suite_configs[suite_name]
                version = suite_versions[suite_name]
                prior = _prior_entry(ep.id, suite, version)
                if not budget.allows(
                    estimate_stage_cost(model_config.model, suite, prior)
                ):
                    budget.skipped.append(f"{ep.id}/{suite_name}")
                    break
                run = await run_suite(
                    suite, model_config, concurrency=concurrency,
                    cache_dir=cache_dir, show_progress=show_progress,
                )
                budget.add(run.total_cost_usd)

                pushback_run = None
                if suite_name == panel.sycophancy_on:
                    pushback_suite = build_pushback_suite(suite, run)
                    # Estimate the probe from ITS prior workload (pushback
                    # prompts embed the model's previous answer and run
                    # systematically longer than the base suite's).
                    pb_prior = None
                    if prior and prior.get("pushback_input_tokens"):
                        pb_prior = {
                            "input_tokens": prior["pushback_input_tokens"],
                            "output_tokens": prior["pushback_output_tokens"],
                        }
                    if budget.allows(
                        estimate_stage_cost(model_config.model,
                                            pushback_suite, pb_prior)
                    ):
                        pushback_run = await run_suite(
                            pushback_suite, model_config,
                            concurrency=concurrency, cache_dir=cache_dir,
                            show_progress=show_progress,
                        )
                        budget.add(pushback_run.total_cost_usd)

                records.append(build_record(
                    run, ep.id, date,
                    panel_version=version,
                    pushback_run=pushback_run,
                ))
            if budget.aborted:
                remaining = selected[selected.index(ep) + 1:]
                budget.skipped.extend(f"{r.id}/*" for r in remaining)
                break
        except MissingAPIKeyError:
            raise
        except Exception as exc:  # noqa: BLE001 — endpoint isolation
            import warnings
            warnings.warn(
                f"Observatory endpoint {ep.id!r} failed and was skipped "
                f"this run: {type(exc).__name__}: {exc}",
                stacklevel=2,
            )
            continue
    return records, budget


def replay_panel(
    run_files: list[str | Path],
    date: str | None = None,
) -> list[ObservationRecord]:
    """Build observation records from saved RunResult JSONs — no network.

    Endpoint id is the run's model; a run whose ``suite_name`` ends in
    ``__pushback`` (the :func:`build_pushback_suite` convention) is paired
    as the sycophancy follow-up of the same model's base-suite run in the
    same batch. Run files must be full saves (not ``--strip-io``) — the
    derived block needs the raw outputs.
    """
    runs = [RunResult.load(p) for p in run_files]
    pushbacks: dict[tuple[str, str], RunResult] = {}
    bases: list[RunResult] = []
    for r in runs:
        if r.suite_name.endswith(PUSHBACK_SUITE_SUFFIX):
            base_name = r.suite_name[: -len(PUSHBACK_SUITE_SUFFIX)]
            pushbacks[(r.model, base_name)] = r
        else:
            bases.append(r)

    records: list[ObservationRecord] = []
    for run in bases:
        rec_date = date or (run.completed_at or "")[:10] or _utc_today()
        records.append(build_record(
            run, endpoint_id=run.model, date=rec_date,
            pushback_run=pushbacks.get((run.model, run.suite_name)),
        ))
    return records
