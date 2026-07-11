"""lm-evaluation-harness → Rift suite adapter.

Consumes an lm-eval *task YAML* (``doc_to_text`` / ``doc_to_target`` /
optional ``doc_to_choice`` / ``output_type``) plus the documents it
templates over. Documents come from ``--dataset`` (JSONL/JSON), or from
the task's ``dataset_path`` when that is a local file — the importer
never pulls from the HuggingFace hub.

Scope (v1, disclosed loudly rather than approximated silently):

* ``generate_until`` tasks with flat ``{{field}}`` templates convert
  cleanly to ``exact_match``.
* ``multiple_choice`` tasks are **approximated**: lm-eval scores them by
  comparing answer-choice loglikelihoods, which Rift cannot reproduce
  (providers are generation-only). The importer renders the choices into
  the prompt and asks for the letter — a related but different
  measurement. This is warned at import time and recorded in the suite
  description; do not cite imported-MC numbers as the original
  benchmark.
* Jinja beyond flat variables ({% blocks %}, filters) is refused, and
  ``filter_list`` post-processing is dropped with a warning.
"""

from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Any

import yaml

from . import ImportedSuite, SuiteImportError
from ._common import (
    build_suite,
    provenance_description,
    render_template,
    sanitize_name,
)


def _load_docs(task: dict, task_path: Path, dataset: Path | None) -> list[dict]:
    if dataset is None:
        candidate = task.get("dataset_path")
        if candidate:
            p = Path(str(candidate))
            if not p.is_absolute():
                p = task_path.parent / p
            if p.is_file():
                dataset = p
        if dataset is None:
            raise SuiteImportError(
                f"{task_path}: the task's dataset is not a local file "
                f"(dataset_path={task.get('dataset_path')!r}). Export the "
                "documents to JSONL and pass --dataset docs.jsonl — the "
                "importer does not download datasets."
            )
    text = dataset.read_text()
    if dataset.suffix == ".jsonl":
        docs = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        loaded = json.loads(text)
        docs = loaded if isinstance(loaded, list) else []
    if not docs or not all(isinstance(d, dict) for d in docs):
        raise SuiteImportError(f"{dataset}: expected a JSONL/JSON list of document objects.")
    return docs


def _resolve_field(spec: Any, doc: dict, *, source_desc: str) -> Any:
    """Resolve doc_to_target/doc_to_choice: template, field name, or literal.

    lm-eval allows three spellings; disambiguate the way the harness does:
    a string containing ``{{`` is a template, a string naming a doc key is
    a field reference, anything else is a literal.
    """
    if isinstance(spec, str):
        if "{{" in spec:
            return render_template(spec, doc, source_desc=source_desc)
        if spec in doc:
            return doc[spec]
    return spec


def convert_lmeval(
    source: Path,
    *,
    dataset: str | Path | None = None,
    name: str | None = None,
    scoring_override: str | None = None,
) -> ImportedSuite:
    """Convert an lm-eval task YAML (+ documents) to a Rift suite."""
    with open(source) as f:
        task = yaml.safe_load(f)
    if not isinstance(task, dict):
        raise SuiteImportError(f"{source}: not an lm-eval task mapping.")

    doc_to_text = task.get("doc_to_text")
    doc_to_target = task.get("doc_to_target")
    if not doc_to_text or doc_to_target is None:
        raise SuiteImportError(
            f"{source}: task must define doc_to_text and doc_to_target. "
            "(Python-function tasks can't be imported — only YAML templates.)"
        )

    docs = _load_docs(task, source, Path(dataset) if dataset else None)
    output_type = task.get("output_type", "generate_until")
    doc_to_choice = task.get("doc_to_choice")
    warnings: list[str] = []

    if task.get("filter_list"):
        warnings.append(
            "task defines filter_list post-processing, which was dropped — "
            "wrap the equivalent extraction in a custom scorer if scores "
            "look depressed."
        )
    metrics = [
        m.get("metric") if isinstance(m, dict) else m
        for m in task.get("metric_list", [])
    ]
    unsupported_metrics = [
        m for m in metrics if m not in (None, "exact_match", "acc", "acc_norm")
    ]
    if unsupported_metrics:
        warnings.append(
            f"metrics {unsupported_metrics} have no Rift equivalent and were "
            "dropped; cases are scored exact_match."
        )

    is_mc = output_type == "multiple_choice" and doc_to_choice is not None
    if is_mc:
        warnings.append(
            "output_type multiple_choice is approximated by generation: "
            "lm-eval compares answer-choice loglikelihoods, Rift asks the "
            "model to emit the letter. Related but not the same measurement — "
            "do not cite imported numbers as the original benchmark's."
        )

    letters = string.ascii_uppercase
    cases: list[dict] = []
    for i, doc in enumerate(docs):
        source_desc = f"{source} doc {i}"
        input_text = render_template(str(doc_to_text), doc,
                                     source_desc=source_desc)
        if is_mc:
            choices = _resolve_field(doc_to_choice, doc, source_desc=source_desc)
            if not isinstance(choices, list) or not choices:
                warnings.append(f"doc {i}: unresolvable choices; skipped.")
                continue
            target = _resolve_field(doc_to_target, doc, source_desc=source_desc)
            # MC targets are canonically an index into choices; accept the
            # choice text itself as a fallback.
            try:
                idx = int(target)
            except (TypeError, ValueError):
                idx = choices.index(target) if target in choices else -1
            if not (0 <= idx < len(choices)):
                warnings.append(f"doc {i}: target {target!r} does not index choices; skipped.")
                continue
            options = "\n".join(f"{letters[j]}) {c}" for j, c in enumerate(choices))
            cases.append({
                "input": (f"{input_text}\n\nOptions:\n{options}\n\n"
                          "Answer with the letter of the correct option."),
                "expected": letters[idx],
                "tags": [],
            })
        else:
            expected = _resolve_field(doc_to_target, doc, source_desc=source_desc)
            cases.append({"input": input_text, "expected": expected, "tags": []})

    if not cases:
        raise SuiteImportError(f"{source}: no importable documents.")

    scoring = scoring_override or "exact_match"
    suite_name = sanitize_name(name or task.get("task") or source.stem)
    desc = provenance_description(
        "lm-eval", source, len(cases), warnings,
        extra=f"Task: {task.get('task', source.stem)}; output_type: {output_type}.",
    )
    return ImportedSuite(
        suite=build_suite(suite_name, desc, scoring, cases),
        warnings=warnings,
    )
