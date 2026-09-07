"""Hygiene checks for the scheduled observatory workflow.

Same spirit as test_github_action.py: the workflow can't be executed in a
unit test, but its structure, injection-safety, and the CLI flags it
invokes can all be pinned so it never drifts from the tool it wraps.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from click.testing import CliRunner

from rift.cli import main

ROOT = Path(__file__).parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "observatory.yml"
SELFTEST_SCRIPT = ROOT / ".github" / "scripts" / "observatory_selftest.sh"
COMMIT_SCRIPT = ROOT / ".github" / "scripts" / "observatory_commit.sh"
BOOTSTRAP_SCRIPT = ROOT / ".github" / "scripts" / "observatory_bootstrap.sh"
ALERT_SCRIPT = ROOT / ".github" / "scripts" / "observatory_alert.sh"


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())


def _all_steps() -> list[dict]:
    return [s for job in _workflow()["jobs"].values()
            for s in job.get("steps", [])]


def test_workflow_files_exist():
    assert WORKFLOW.is_file()
    assert SELFTEST_SCRIPT.is_file()
    assert COMMIT_SCRIPT.is_file()
    assert BOOTSTRAP_SCRIPT.is_file()
    assert ALERT_SCRIPT.is_file()


def test_data_branch_is_bootstrapped_before_it_is_checked_out():
    # Every scheduled run from 2026-07 to 2026-09 failed at
    # `actions/checkout ref: observatory-data` because the orphan branch
    # was a manual setup step nobody performed. Both writer jobs must
    # create it themselves, BEFORE the checkout that needs it.
    jobs = _workflow()["jobs"]
    for job in ("observe", "selftest"):
        names = [s.get("name", "") for s in jobs[job]["steps"]]
        runs = [s.get("run", "") for s in jobs[job]["steps"]]
        boot = next(i for i, r in enumerate(runs)
                    if "observatory_bootstrap.sh" in r)
        ckpt = next(i for i, s in enumerate(jobs[job]["steps"])
                    if s.get("with", {}).get("ref") == "observatory-data")
        assert boot < ckpt, (
            f"{job}: bootstrap ({names[boot]}) must run before the "
            f"data-branch checkout ({names[ckpt]})"
        )
    text = BOOTSTRAP_SCRIPT.read_text()
    assert "set -euo pipefail" in text
    # An orphan root commit, never a force-push over an existing branch.
    assert "commit-tree" in text
    assert "--force" not in text and "+refs" not in text


def test_failure_opens_an_issue():
    # Sixteen silent failures in a row is the bug; a red badge is not a
    # notification. The alert job must run on failure of any producer.
    wf = _workflow()
    alert = wf["jobs"]["alert"]
    assert set(alert["needs"]) == {"observe", "selftest", "pages"}
    for job in ("observe", "selftest", "pages"):
        assert f"needs.{job}.result == 'failure'" in alert["if"]
    assert alert["permissions"]["issues"] == "write"
    assert wf["permissions"]["issues"] == "write"
    step = next(s for s in alert["steps"] if "observatory_alert.sh" in s.get("run", ""))
    # Token and run URL reach the script through env, never the run body.
    assert "GH_TOKEN" in step["env"] and "RUN_URL" in step["env"]
    text = ALERT_SCRIPT.read_text()
    assert "set -euo pipefail" in text
    # One issue per outage: append when one is already open.
    assert "gh issue list" in text and "gh issue comment" in text
    assert "gh issue create" in text


def test_schedule_and_dispatch_triggers():
    # PyYAML parses the bare `on:` key as boolean True.
    on = _workflow().get("on") or _workflow()[True]
    crons = {s["cron"] for s in on["schedule"]}
    assert "0 6 * * 1" in crons      # weekly panel
    assert "0 7 1 * *" in crons      # monthly selftest
    assert "workflow_dispatch" in on


def test_no_secret_or_input_interpolation_in_run_bodies():
    # Script-injection guard: secrets/inputs reach the shell via env:,
    # never as ${{ }} interpolated into a run: body.
    for step in _all_steps():
        run = step.get("run", "")
        for needle in ("${{ secrets.", "${{ inputs.", "${{ github.event."):
            assert needle not in run, (
                f"step '{step.get('name')}' interpolates {needle}*  into its "
                f"run body; pass it via env: instead"
            )


def test_api_keys_passed_via_env():
    panel_steps = [s for s in _all_steps()
                   if "observe " in s.get("run", "")
                   or "observatory_selftest" in s.get("run", "")]
    assert panel_steps
    for step in panel_steps:
        env = step.get("env", {})
        for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
            assert key in env, f"step '{step.get('name')}' missing {key} env"


def test_concurrency_guard_present():
    wf = _workflow()
    assert wf["concurrency"]["group"] == "observatory"
    assert wf["concurrency"]["cancel-in-progress"] is False


def test_pages_job_permissions():
    pages = _workflow()["jobs"]["pages"]
    # Deploys after either producer so the monthly selftest refresh also
    # updates the published calibration figures.
    assert pages["needs"] == ["observe", "selftest"]
    assert pages["permissions"]["pages"] == "write"
    assert pages["permissions"]["id-token"] == "write"


def test_no_completion_cache_persisted_across_runs():
    # Restoring .rift/cache between scheduled runs would turn every later
    # "observation" into a cache replay of week 1 (scores AND fingerprints
    # round-trip through the cache), structurally blinding the observatory.
    # The panel must hit the live endpoint on every pass.
    for step in _all_steps():
        assert not step.get("uses", "").startswith("actions/cache"), (
            f"step '{step.get('name')}' persists a cache across scheduled "
            f"runs — the observatory must re-query live endpoints every pass"
        )


def test_observe_step_sets_pipefail():
    # `rift observe | tee` under the default Actions shell (`bash -e {0}`,
    # no pipefail) would mask a crashed pass and commit partial data.
    observe_steps = [s for s in _all_steps() if "rift" in s.get("run", "")
                     and "observe " in s.get("run", "")]
    assert observe_steps
    for step in observe_steps:
        if "| tee" in step["run"]:
            assert "set -o pipefail" in step["run"]


def test_commit_script_fails_when_push_never_lands():
    # A green job whose push silently failed loses the week's observations.
    text = COMMIT_SCRIPT.read_text()
    assert "set -euo pipefail" in text
    assert "exit 1" in text
    # Both jobs commit through the shared script — retry semantics can't drift.
    commit_steps = [s for s in _all_steps()
                    if "observatory_commit.sh" in s.get("run", "")]
    assert len(commit_steps) == 2


def test_observe_and_selftest_jobs_are_mutually_exclusive():
    jobs = _workflow()["jobs"]
    observe_if, selftest_if = jobs["observe"]["if"], jobs["selftest"]["if"]
    # Weekly cron → observe only; monthly cron → selftest only; dispatch
    # routes on the selftest input.
    assert "0 6 * * 1" in observe_if and "0 6 * * 1" not in selftest_if
    assert "0 7 1 * *" in selftest_if and "0 7 1 * *" not in observe_if
    assert "!inputs.selftest" in observe_if
    assert "inputs.selftest" in selftest_if


def _cli_flags(command: str) -> set[str]:
    help_text = CliRunner().invoke(main, [command, "--help"]).output
    return set(re.findall(r"--[a-z][a-z0-9-]+", help_text))


def test_workflow_uses_real_observe_flags():
    used = {"--panel", "--data-dir", "--max-cost"}
    missing = used - _cli_flags("observe")
    assert not missing, f"workflow uses flags not on `rift observe`: {missing}"


def test_workflow_uses_real_site_flags():
    used = {"--data-dir", "--out"}
    missing = used - _cli_flags("observatory-site")
    assert not missing


def test_selftest_script_writes_only_the_selftest_block():
    text = SELFTEST_SCRIPT.read_text()
    assert "rift" in text and "selftest" in text
    # The stored file must be the SelfTestResult alone, not the full run
    # (which would bloat the data branch with per-trial completions).
    assert 'data["selftest"]' in text
