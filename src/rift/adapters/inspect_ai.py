"""Inspect AI → Rift suite adapter.

Consumes an Inspect *dataset* file (JSONL or a JSON list of samples with
``input`` / ``target`` / optional ``choices`` / ``metadata`` / ``id``).
The scorer in an Inspect eval lives in Python ``@task`` code, which this
importer deliberately does not execute — pass ``--scoring`` to say how
the imported cases should be graded (default ``exact_match``, matching
Inspect's ``exact()``/``match()``).

Transforms and their disclosures:

* chat-message ``input`` is flattened to one prompt string (warned);
* ``choices`` are rendered into the prompt as lettered options and the
  target letter becomes ``expected``;
* a list ``target`` (any-of grading) keeps only the first entry (warned) —
  Rift scores against a single expected value;
* ``metadata`` scalars and ``id`` become ``key:value`` tags, which drive
  ``rift compare --subgroup key:``.
"""

from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Any

from . import ImportedSuite, SuiteImportError
from ._common import (
    build_suite,
    flatten_chat,
    provenance_description,
    sanitize_name,
    stringify_tags,
)


def _load_samples(source: Path) -> list[dict]:
    text = source.read_text()
    if source.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        data = json.loads(text)
        rows = data if isinstance(data, list) else data.get("samples", [])
    if not rows:
        raise SuiteImportError(f"{source}: no samples found.")
    if not all(isinstance(r, dict) for r in rows):
        raise SuiteImportError(f"{source}: expected each sample to be an object.")
    return rows


def _letter_options(choices: list[Any]) -> str:
    letters = string.ascii_uppercase
    return "\n".join(f"{letters[i]}) {c}" for i, c in enumerate(choices))


def convert_inspect(
    source: Path,
    *,
    name: str | None = None,
    scoring: str | None = None,
) -> ImportedSuite:
    """Convert an Inspect dataset file to a Rift suite."""
    samples = _load_samples(source)
    scoring = scoring or "exact_match"
    warnings: list[str] = []
    cases: list[dict] = []

    for i, sample in enumerate(samples):
        label = f"case {i}"
        raw_input = sample.get("input")
        if raw_input is None:
            warnings.append(f"{label}: no 'input' field; skipped.")
            continue
        if isinstance(raw_input, list):
            input_text = flatten_chat(raw_input, warnings=warnings,
                                      case_label=label)
        else:
            input_text = str(raw_input)

        target = sample.get("target")
        if isinstance(target, list):
            if len(target) > 1:
                warnings.append(
                    f"{label}: target lists {len(target)} acceptable answers; "
                    "only the first was kept (Rift scores against one "
                    "expected value)."
                )
            target = target[0] if target else None
        if target is None:
            warnings.append(f"{label}: no 'target' field; skipped.")
            continue

        choices = sample.get("choices")
        if choices:
            input_text = (
                f"{input_text}\n\nOptions:\n{_letter_options(choices)}\n\n"
                "Answer with the letter of the correct option."
            )

        cases.append({
            "input": input_text,
            "expected": target,
            "tags": stringify_tags(sample.get("metadata"), sample.get("id")),
        })

    if not cases:
        raise SuiteImportError(f"{source}: no importable samples.")

    suite_name = sanitize_name(name or source.stem)
    desc = provenance_description(
        "inspect", source, len(cases), warnings,
        extra=(
            f"Scoring: {scoring} (Inspect scorers live in @task code, which "
            "the importer does not execute — pass --scoring to change)."
        ),
    )
    return ImportedSuite(
        suite=build_suite(suite_name, desc, scoring, cases),
        warnings=warnings,
    )
