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


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())


def _all_steps() -> list[dict]:
    return [s for job in _workflow()["jobs"].values()
            for s in job.get("steps", [])]


def test_workflow_files_exist():
    assert WORKFLOW.is_file()
    assert SELFTEST_SCRIPT.is_file()


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
    assert pages["needs"] == "observe"
    assert pages["permissions"]["pages"] == "write"
    assert pages["permissions"]["id-token"] == "write"


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
