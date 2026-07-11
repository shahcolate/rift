"""Tests for the frontier head-to-head benchmark driver.

Everything here is keyless: the replay path rebuilds reports from
synthetic run JSONs, and the live path is only exercised up to the
budget pre-flight refusal (which fires before any provider work).
"""

import importlib.util
import json
from pathlib import Path

import pytest

from rift.runner import CaseResult, RunResult

_DRIVER = Path(__file__).parent.parent / "benchmarks" / "frontier" / "run_frontier.py"
_spec = importlib.util.spec_from_file_location("run_frontier", _DRIVER)
frontier = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(frontier)

BASELINE = "fable-5"
CHALLENGER = "gpt-5.6-sol"
# Real panel sizes: reasoning 10, extraction 29, hard_reasoning 24.
POOLED_SIZES = {"reasoning": 10, "extraction": 29, "hard_reasoning": 24}
EXPLORATORY_SIZES = {"summarization": 8, "code_generation": 5}


def _mk_run(model: str, suite: str, scores: list[float],
            errors: dict[int, str] | None = None,
            fingerprint: str = "fp-1") -> RunResult:
    errors = errors or {}
    cases = [
        CaseResult(
            case_index=i,
            input_text=f"input {i}",
            expected="x",
            output=f"output {i}",
            score=score,
            latency_ms=10.0,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
            tags=[],
            error=errors.get(i),
            provider_fingerprint=fingerprint,
        )
        for i, score in enumerate(scores)
    ]
    return RunResult(
        model=model, suite_name=suite, scoring_method="exact_match",
        cases=cases, started_at="2026-07-12T00:00:00Z",
        completed_at="2026-07-12T00:05:00Z",
        metadata={"n_errors": len(errors)},
    )


def _panel_runs(flip_challenger: int = 0, errors_on: str | None = None):
    """Build matched baseline/challenger runs over the full panel.

    ``flip_challenger`` flips that many baseline-correct cases to 0.0 on
    the challenger side (spread across suites) to manufacture a deficit.
    ``errors_on`` marks case 0 of that suite as errored on the challenger.
    """
    baseline, challenger = {}, {}
    remaining_flips = flip_challenger
    for suite, n in POOLED_SIZES.items():
        b_scores = [1.0] * n
        c_scores = list(b_scores)
        flips = min(remaining_flips, n - 1)
        for i in range(1, flips + 1):
            c_scores[i] = 0.0
        remaining_flips -= flips
        errs = {0: "timeout"} if errors_on == suite else None
        baseline[suite] = _mk_run("claude-fable-5", suite, b_scores)
        challenger[suite] = _mk_run("gpt-5.6-sol", suite, c_scores, errors=errs)
    for suite, n in EXPLORATORY_SIZES.items():
        baseline[suite] = _mk_run("claude-fable-5", suite, [0.8] * n)
        challenger[suite] = _mk_run("gpt-5.6-sol", suite, [0.7] * n)
    return baseline, challenger


class TestPoolPairs:
    def test_pools_in_declared_suite_order_with_tags(self):
        baseline, challenger = _panel_runs()
        b, c, bc, cc, tags, excluded, kept = frontier.pool_pairs(
            baseline, challenger
        )
        assert len(b) == len(c) == len(tags) == len(kept) == sum(POOLED_SIZES.values())
        assert excluded == 0
        # Tag layout must follow the declared POOLED_SUITES order.
        expect_tags = [
            f"suite:{name}"
            for name in frontier.POOLED_SUITES
            for _ in range(POOLED_SIZES[name])
        ]
        assert [t[0] for t in tags] == expect_tags
        assert bc == [0.001] * len(b)

    def test_errored_pair_excluded_either_side(self):
        baseline, challenger = _panel_runs(errors_on="extraction")
        b, c, _, _, _, excluded, kept = frontier.pool_pairs(baseline, challenger)
        assert excluded == 1
        assert len(b) == len(kept) == sum(POOLED_SIZES.values()) - 1
        assert ("extraction", 0) not in kept

    def test_partial_credit_binarized_all_or_nothing(self):
        # extraction's per-field scorer can emit fractions (e.g. 0.75).
        # The panel must binarize them, or a single partial score would
        # silently switch the pre-registered primary from McNemar to
        # paired-t depending on the data.
        from rift.comparator import compare_runs

        baseline, challenger = _panel_runs()
        challenger["extraction"].cases[3].score = 0.75
        b, c, bc, cc, _, _, _ = frontier.pool_pairs(baseline, challenger)
        assert set(b) | set(c) <= {0.0, 1.0}
        drift = compare_runs(
            baseline_scores=b, challenger_scores=c,
            baseline_model=BASELINE, challenger_model=CHALLENGER,
            suite_name=frontier.PANEL_NAME,
            baseline_costs=bc, challenger_costs=cc,
        )
        assert drift.test_used == "mcnemar_exact"

    def test_panel_score_trials_fraction_of_correct_trials(self):
        case = _mk_run("m", "reasoning", [1.0]).cases[0]
        case.trial_scores = [1.0, 0.5, 0.0]
        assert frontier._panel_score(case) == pytest.approx(1 / 3)

    def test_pooled_run_aligns_with_drift_vectors(self):
        # A pair errored on ONE side must vanish from BOTH pooled runs,
        # or the report's regressed-cases table (positional indexing)
        # attributes regressions to the wrong cases.
        baseline, challenger = _panel_runs(errors_on="reasoning")
        b, c, _, _, _, _, kept = frontier.pool_pairs(baseline, challenger)
        pooled_base = frontier._pooled_run(BASELINE, baseline, kept)
        pooled_chal = frontier._pooled_run(CHALLENGER, challenger, kept)
        assert len(pooled_base.cases) == len(pooled_chal.cases) == len(b)
        # The errored pair (reasoning case 0, challenger side) is gone
        # from the baseline's pooled run too, even though the baseline
        # case itself did not error.
        assert pooled_base.cases[0].case_index == 1
        assert pooled_base.scores == b
        assert pooled_chal.scores == c


class TestVerdictSentence:
    def _outcome(self, baseline_runs, challenger_runs):
        from rift.comparator import compare_runs
        from rift.preregistration import evaluate, load_preregistration

        prereg = load_preregistration(frontier.PREREG_PATH)
        b, c, bc, cc, _, _, _ = frontier.pool_pairs(baseline_runs, challenger_runs)
        drift = compare_runs(
            baseline_scores=b, challenger_scores=c,
            baseline_model=BASELINE, challenger_model=CHALLENGER,
            suite_name=frontier.PANEL_NAME, alpha=prereg.alpha,
            baseline_costs=bc, challenger_costs=cc,
        )
        return evaluate(prereg, drift, n_cases=drift.n_cases,
                        baseline_model=BASELINE, challenger_model=CHALLENGER)

    def test_no_difference_reads_as_null(self):
        outcome = self._outcome(*_panel_runs(flip_challenger=0))
        assert not outcome.primary_significant
        assert outcome.honored, outcome.violations
        sentence = frontier._verdict_sentence(outcome)
        assert "No significant difference" in sentence
        assert "DISHONORED" not in sentence

    def test_large_deficit_confirms_two_sided(self):
        outcome = self._outcome(*_panel_runs(flip_challenger=20))
        assert outcome.primary_significant
        sentence = frontier._verdict_sentence(outcome)
        assert "Significant difference" in sentence
        assert "baseline ahead" in sentence

    def test_alias_identity_honored_against_spec(self):
        # Spec pins canonical ids; the run uses aliases. resolve_model
        # canonicalization must keep the plan honored.
        outcome = self._outcome(*_panel_runs())
        assert all("mismatch" not in v for v in outcome.violations)

    def test_cost_primary_two_sided_names_winner_by_endpoint(self):
        # For a cost_per_correct primary, positive delta = challenger
        # MORE expensive = baseline ahead. The sentence must not reuse
        # accuracy's sign convention.
        from rift.preregistration import PreregOutcome

        outcome = PreregOutcome(
            primary="cost_per_correct", direction="two_sided", alpha=0.05,
            honored=True, violations=[], primary_delta=+0.01,
            primary_significant=True, adverse_confirmed=True,
            detail="$/correct Δ=+0.0100",
        )
        assert "baseline ahead" in frontier._verdict_sentence(outcome)
        outcome.primary_delta = -0.01
        assert "challenger ahead" in frontier._verdict_sentence(outcome)


class TestReplayEndToEnd:
    def _write_capture(self, tmp_path: Path, **panel_kwargs) -> Path:
        baseline, challenger = _panel_runs(**panel_kwargs)
        for suite, run in baseline.items():
            run.save(tmp_path / suite / f"{BASELINE}.json")
        for suite, run in challenger.items():
            run.save(tmp_path / suite / f"{CHALLENGER}.json")
        return tmp_path

    def test_replay_writes_report_with_prereg_verdict(self, tmp_path):
        capture = self._write_capture(tmp_path)
        frontier.main([
            "--mode", "replay", "--from-dir", str(capture),
            "--baseline", BASELINE, "--challenger", CHALLENGER,
        ])
        report = (capture / "report.md").read_text()
        assert "Pre-registered primary endpoint" in report
        assert frontier.PANEL_NAME in report
        assert "No significant difference" in report
        assert "Keyless replay" in report
        assert "Scorecard" in report
        # Exploratory suites made it in as exploratory, not confirmatory.
        assert "### `summarization`" in report

    def test_replay_reports_excluded_pairs(self, tmp_path):
        capture = self._write_capture(tmp_path, errors_on="reasoning")
        frontier.main([
            "--mode", "replay", "--from-dir", str(capture),
            "--baseline", BASELINE, "--challenger", CHALLENGER,
        ])
        report = (capture / "report.md").read_text()
        assert "1 case pair(s) excluded" in report

    def test_replay_missing_pooled_suite_is_operational_error(self, tmp_path):
        capture = self._write_capture(tmp_path)
        (capture / "hard_reasoning" / f"{CHALLENGER}.json").unlink()
        with pytest.raises(SystemExit) as exc_info:
            frontier.main([
                "--mode", "replay", "--from-dir", str(capture),
                "--baseline", BASELINE, "--challenger", CHALLENGER,
            ])
        assert exc_info.value.code == 2

    def test_replay_requires_from_dir(self):
        with pytest.raises(SystemExit) as exc_info:
            frontier.main(["--mode", "replay"])
        assert exc_info.value.code == 2

    def test_replay_surfaces_trials_from_metadata(self, tmp_path):
        baseline, challenger = _panel_runs()
        for runs, name in ((baseline, BASELINE), (challenger, CHALLENGER)):
            for suite, run in runs.items():
                run.metadata["trials"] = 3
                for case in run.cases:
                    case.trial_scores = [case.score] * 3
                run.save(tmp_path / suite / f"{name}.json")
        frontier.main([
            "--mode", "replay", "--from-dir", str(tmp_path),
            "--baseline", BASELINE, "--challenger", CHALLENGER,
        ])
        report = (tmp_path / "report.md").read_text()
        assert "trials per case: 3" in report
        assert "Replication / noise floor" in report

    def test_asymmetric_exploratory_suite_dropped_not_crashed(self, tmp_path, capsys):
        # summarization present for the baseline only (budget cap tripped
        # between legs, or a file missing from a replay dir): the suite
        # is dropped with a warning, the report still renders.
        capture = self._write_capture(tmp_path)
        (capture / "summarization" / f"{CHALLENGER}.json").unlink()
        frontier.main([
            "--mode", "replay", "--from-dir", str(capture),
            "--baseline", BASELINE, "--challenger", CHALLENGER,
        ])
        report = (capture / "report.md").read_text()
        assert "### `summarization`" not in report
        assert "### `code_generation`" in report
        assert "one side only" in capsys.readouterr().err

    def test_malformed_run_json_is_operational_error(self, tmp_path):
        capture = self._write_capture(tmp_path)
        (capture / "extraction" / f"{BASELINE}.json").write_text("{ truncated")
        with pytest.raises(SystemExit) as exc_info:
            frontier.main([
                "--mode", "replay", "--from-dir", str(capture),
                "--baseline", BASELINE, "--challenger", CHALLENGER,
            ])
        assert exc_info.value.code == 2

    def test_all_errored_side_refuses_verdict(self, tmp_path):
        # Total challenger outage: every pooled pair excluded. Publishing
        # "no significant difference" over a dead panel would be the
        # exact failure mode the exit-code contract exists to prevent.
        baseline, challenger = _panel_runs()
        for suite in POOLED_SIZES:
            for case in challenger[suite].cases:
                case.error = "connection reset"
        for suite, run in baseline.items():
            run.save(tmp_path / suite / f"{BASELINE}.json")
        for suite, run in challenger.items():
            run.save(tmp_path / suite / f"{CHALLENGER}.json")
        with pytest.raises(SystemExit) as exc_info:
            frontier.main([
                "--mode", "replay", "--from-dir", str(tmp_path),
                "--baseline", BASELINE, "--challenger", CHALLENGER,
            ])
        assert exc_info.value.code == 2
        assert not (tmp_path / "report.md").exists()

    def test_excluded_pair_regression_attributed_to_right_case(self, tmp_path):
        # Regression case AFTER an excluded errored pair: the regressed-
        # cases table indexes pooled cases positionally, so a misaligned
        # pooled run would display the wrong input text here.
        baseline, challenger = _panel_runs(errors_on="reasoning")
        challenger["reasoning"].cases[5].score = 0.0
        for suite, run in baseline.items():
            run.save(tmp_path / suite / f"{BASELINE}.json")
        for suite, run in challenger.items():
            run.save(tmp_path / suite / f"{CHALLENGER}.json")
        frontier.main([
            "--mode", "replay", "--from-dir", str(tmp_path),
            "--baseline", BASELINE, "--challenger", CHALLENGER,
        ])
        report = (tmp_path / "report.md").read_text()
        primary = report.split("## Scorecard")[0]
        assert "input 5" in primary
        assert "input 4" not in primary

    def test_strip_io_capture_still_replays(self, tmp_path):
        baseline, challenger = _panel_runs()
        for runs, name in ((baseline, BASELINE), (challenger, CHALLENGER)):
            for suite, run in runs.items():
                run.save(tmp_path / suite / f"{name}.json", strip_io=True)
        frontier.main([
            "--mode", "replay", "--from-dir", str(tmp_path),
            "--baseline", BASELINE, "--challenger", CHALLENGER,
        ])
        data = json.loads(
            (tmp_path / "reasoning" / f"{BASELINE}.json").read_text()
        )
        assert data["cases"][0]["input_text"] == ""
        assert (tmp_path / "report.md").exists()


class TestLiveBudgetPreflight:
    def test_tiny_cap_refuses_before_any_provider_work(self, tmp_path):
        # No API keys in the test env: reaching the key preflight would
        # raise MissingAPIKeyError, so exit-2-with-refusal proves the
        # budget check fired first.
        with pytest.raises(SystemExit) as exc_info:
            frontier.main([
                "--mode", "live", "--max-cost", "0.0001",
                "--out-dir", str(tmp_path),
            ])
        assert exc_info.value.code == 2
        assert not (tmp_path / "report.md").exists()

    def test_estimate_is_positive_for_priced_pair(self):
        from rift.config import load_suite

        suites = [load_suite(n) for n in frontier.POOLED_SUITES]
        est = frontier.estimate_total_cost(
            [BASELINE, CHALLENGER], suites, trials=1
        )
        assert est > 0
        # Trials scale the estimate linearly.
        assert frontier.estimate_total_cost(
            [BASELINE, CHALLENGER], suites, trials=3
        ) == pytest.approx(est * 3)


class TestRepoPlumbing:
    def test_results_dir_not_gitignored(self):
        # The repo-root `results/` ignore pattern must not swallow
        # benchmarks/frontier/results/ — the Actions workflow commits
        # captures there for keyless replay.
        import subprocess

        repo = Path(__file__).parent.parent
        if not (repo / ".git").exists():
            pytest.skip("not a git checkout")
        probe = "benchmarks/frontier/results/2026-01-01/reasoning/x.json"
        proc = subprocess.run(
            ["git", "check-ignore", "-q", probe], cwd=repo,
            capture_output=True,
        )
        # exit 1 = NOT ignored (what we require); 0 = ignored.
        assert proc.returncode == 1, (
            f"{probe} is gitignored — the workflow's commit step would fail"
        )
