"""Docs-vs-CLI conformance: every command the docs name must exist.

CLAUDE.md once documented a `rift report` command that didn't exist —
the kind of drift a reviewer finds in minutes. This guard parses the
`rift <cmd>` invocations out of CLAUDE.md and README.md and fails when
one names a command the CLI doesn't ship.
"""

from __future__ import annotations

import re
from pathlib import Path

from rift.cli import main

ROOT = Path(__file__).parent.parent

# `rift <word>` where <word> should be a subcommand. Excludes flags and
# the bare `rift` / `rift --help` forms.
_CMD_RE = re.compile(r"(?m)^\s*(?:\$ )?rift ([a-z][a-z0-9-]*)\b")

# Words that follow `rift` in prose/examples but are not subcommands.
_NOT_COMMANDS = {"eval"}  # e.g. "rift eval suites" prose


def _documented_commands(path: Path) -> set[str]:
    return {
        m.group(1)
        for m in _CMD_RE.finditer(path.read_text())
        if m.group(1) not in _NOT_COMMANDS
    }


def test_claude_md_commands_exist():
    documented = _documented_commands(ROOT / "CLAUDE.md")
    assert documented, "expected CLAUDE.md to document rift commands"
    missing = documented - set(main.commands)
    assert not missing, (
        f"CLAUDE.md documents commands the CLI doesn't have: {sorted(missing)}"
    )


def test_readme_commands_exist():
    documented = _documented_commands(ROOT / "README.md")
    assert documented, "expected README.md to document rift commands"
    missing = documented - set(main.commands)
    assert not missing, (
        f"README.md documents commands the CLI doesn't have: {sorted(missing)}"
    )


def test_lm_subcommands_exist():
    lm = main.commands["lm"]
    for sub in ("train", "sample", "suite"):
        assert sub in lm.commands  # type: ignore[attr-defined]
