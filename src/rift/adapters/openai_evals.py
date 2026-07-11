"""OpenAI evals → Rift suite adapter.

Consumes the evals sample format: JSONL where each line carries a chat
``input`` (message list or plain string) and an ``ideal`` answer
(string or list of acceptable strings). This is the ``match``-style
samples file used by the original openai/evals registry.

Chat inputs are flattened to one prompt string (warned); a list
``ideal`` keeps only the first entry (warned). Default scoring is
``exact_match`` (the evals ``match`` grader); pass ``--scoring`` for
fuzzier graders.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import ImportedSuite, SuiteImportError
from ._common import (
    build_suite,
    flatten_chat,
    provenance_description,
    sanitize_name,
)


def convert_openai_evals(
    source: Path,
    *,
    name: str | None = None,
    scoring: str | None = None,
) -> ImportedSuite:
    """Convert an OpenAI evals samples JSONL to a Rift suite."""
    rows = [
        json.loads(line)
        for line in source.read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise SuiteImportError(f"{source}: no samples found.")

    scoring = scoring or "exact_match"
    warnings: list[str] = []
    cases: list[dict] = []

    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            warnings.append(f"case {i}: not an object; skipped.")
            continue
        label = f"case {i}"
        raw_input = row.get("input")
        if raw_input is None:
            warnings.append(f"{label}: no 'input' field; skipped.")
            continue
        if isinstance(raw_input, list):
            input_text = flatten_chat(raw_input, warnings=warnings,
                                      case_label=label)
        else:
            input_text = str(raw_input)

        ideal = row.get("ideal")
        if isinstance(ideal, list):
            if len(ideal) > 1:
                warnings.append(
                    f"{label}: 'ideal' lists {len(ideal)} acceptable answers; "
                    "only the first was kept (Rift scores against one "
                    "expected value)."
                )
            ideal = ideal[0] if ideal else None
        if ideal is None:
            warnings.append(f"{label}: no 'ideal' field; skipped.")
            continue

        cases.append({"input": input_text, "expected": ideal, "tags": []})

    if not cases:
        raise SuiteImportError(f"{source}: no importable samples.")

    suite_name = sanitize_name(name or source.stem)
    desc = provenance_description(
        "openai-evals", source, len(cases), warnings,
        extra=f"Scoring: {scoring} (evals 'match' grader equivalent).",
    )
    return ImportedSuite(
        suite=build_suite(suite_name, desc, scoring, cases),
        warnings=warnings,
    )
