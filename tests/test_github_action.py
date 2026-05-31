"""Validate the bundled GitHub Action stays consistent with the CLI.

These tests don't run the action (that needs a GH runner); they parse
``action.yml`` and assert its structure, its injection-safe shape, and — most
importantly — that every ``rift compare`` flag the action invokes actually
exists on the CLI, so the action can't silently drift from the tool it wraps.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from click.testing import CliRunner

from rift.cli import main

ACTION_DIR = Path(__file__).parent.parent / ".github" / "actions" / "rift-drift-check"
ACTION_YML = ACTION_DIR / "action.yml"


def _action() -> dict:
    return yaml.safe_load(ACTION_YML.read_text())


def test_action_files_exist():
    assert ACTION_YML.is_file()
    assert (ACTION_DIR / "README.md").is_file()


def test_action_is_well_formed_composite():
    a = _action()
    assert a["name"]
    assert a["description"]
    assert a["runs"]["using"] == "composite"
    assert isinstance(a["runs"]["steps"], list) and a["runs"]["steps"]


def test_required_inputs_present():
    inputs = _action()["inputs"]
    for req in ("baseline", "challenger", "suite"):
        assert inputs[req]["required"] is True
    for opt in ("alpha", "fail-on-regression", "report-path", "metrics-format",
                "cache-dir", "install-spec", "python-version"):
        assert "default" in inputs[opt]


def test_outputs_declared():
    outputs = _action()["outputs"]
    for o in ("regression", "report-path", "metrics-path"):
        assert o in outputs


def _compare_flags() -> set[str]:
    help_text = CliRunner().invoke(main, ["compare", "--help"]).output
    return set(re.findall(r"--[a-z][a-z0-9-]+", help_text))


def test_compare_flags_used_by_action_exist():
    # Flags the action passes to `rift compare`.
    used = {"--baseline", "--challenger", "--suite", "--alpha", "--report",
            "--cache-dir", "--metrics-out", "--metrics-format"}
    available = _compare_flags()
    missing = used - available
    assert not missing, f"action uses flags not on `rift compare`: {missing}"


def test_run_step_invokes_rift_compare():
    steps = _action()["runs"]["steps"]
    run_blob = "\n".join(s.get("run", "") for s in steps)
    assert "rift " in run_blob
    assert "compare" in run_blob
    # The exit-code contract: 1 == regression is handled explicitly.
    assert "regression=true" in run_blob
    assert "regression=false" in run_blob
    assert "GITHUB_STEP_SUMMARY" in run_blob
    assert "GITHUB_OUTPUT" in run_blob


def test_metrics_format_choices_match_cli():
    help_text = CliRunner().invoke(main, ["compare", "--help"]).output
    assert "json" in help_text and "prometheus" in help_text


# --- security / robustness guards ---

def test_no_input_interpolation_in_run_bodies():
    # Script-injection guard: inputs must reach the shell via env:, never as
    # ${{ inputs.* }} interpolated directly into a run: block.
    for step in _action()["runs"]["steps"]:
        run = step.get("run", "")
        assert "${{ inputs." not in run, (
            f"step '{step.get('name')}' interpolates an input into its run "
            f"body; pass it via env: and reference $VAR instead"
        )


def test_run_steps_pass_inputs_via_env():
    steps = _action()["runs"]["steps"]
    compare = next(s for s in steps if s.get("id") == "compare")
    env = compare.get("env", {})
    for key in ("BASELINE", "CHALLENGER", "SUITE", "EXTRA_ARGS"):
        assert key in env, f"compare step env missing {key}"
        assert "inputs." in env[key]  # bound to a ${{ inputs.* }} value


def test_install_spec_default_is_version_pinned():
    # Default must guarantee a Rift new enough for the flags the action uses,
    # not a bare 'rift-eval' that could resolve to a pre-feature release.
    spec = _action()["inputs"]["install-spec"]["default"]
    assert spec != "rift-eval", "default install-spec must pin a minimum version"
    assert ">=1.0.0" in spec or "==" in spec


def test_extra_args_not_unquoted_splat():
    # The action must split extra-args with `read -ra`, not an unquoted splat.
    compare = next(s for s in _action()["runs"]["steps"]
                   if s.get("id") == "compare")
    run = compare["run"]
    assert "read -ra" in run
    assert "args+=($EXTRA_ARGS)" not in run
