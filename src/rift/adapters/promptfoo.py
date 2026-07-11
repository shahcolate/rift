"""promptfoo → Rift suite adapter.

Reads a ``promptfooconfig.yaml`` (``prompts`` templates + ``tests`` with
``vars``/``assert``) and emits Rift suites. The structural mismatch this
adapter must bridge: promptfoo attaches assertions per *test case* and
allows several per case; a Rift suite has exactly one scoring method. The
resolution is explicit, never silent:

* Within one case, the first supported assertion decides the case's
  scoring; further assertions are dropped with a warning.
* Across cases, mixed scoring groups either fail with a message listing
  the groups, or — with ``--split-by-assert`` — emit one suite per group.

Assertion mapping (unsupported types are dropped with a warning):

======================  =============================================
promptfoo assert        Rift scoring
======================  =============================================
equals                  exact_match
contains / icontains    custom → rift.adapters.scorers:{contains,icontains}
starts-with             custom → rift.adapters.scorers:starts_with
regex                   custom → rift.adapters.scorers:regex_match
contains-any/-all       custom → rift.adapters.scorers:contains_{any,all}
similar                 semantic (threshold dropped; Rift scores raw cosine)
llm-rubric              llm_judge with expected: {rubric: ...}
model-graded-closedqa   llm_judge with the value as reference answer
======================  =============================================
"""

from __future__ import annotations

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

# assert type -> (rift scoring, custom_scorer target or None)
_ASSERT_MAP: dict[str, tuple[str, str | None]] = {
    "equals": ("exact_match", None),
    "contains": ("custom", "rift.adapters.scorers:contains"),
    "icontains": ("custom", "rift.adapters.scorers:icontains"),
    "starts-with": ("custom", "rift.adapters.scorers:starts_with"),
    "regex": ("custom", "rift.adapters.scorers:regex_match"),
    "contains-any": ("custom", "rift.adapters.scorers:contains_any"),
    "contains-all": ("custom", "rift.adapters.scorers:contains_all"),
    "similar": ("semantic", None),
    "llm-rubric": ("llm_judge", None),
    "model-graded-closedqa": ("llm_judge", None),
}


def _first_prompt(config: dict, warnings: list[str], source: Path) -> str:
    prompts = config.get("prompts")
    if not prompts:
        raise SuiteImportError(
            f"{source}: no 'prompts' entry — nothing to build case inputs from."
        )
    if isinstance(prompts, str):
        return prompts
    if isinstance(prompts, list):
        first = prompts[0]
        if len(prompts) > 1:
            warnings.append(
                f"config lists {len(prompts)} prompts; only the first was "
                "imported (Rift cases carry one prompt each). Re-run the "
                "import per prompt if you need the others."
            )
        if isinstance(first, dict):
            # {id: ..., raw: ...} form; only inline raw templates are portable.
            raw = first.get("raw")
            if raw:
                return str(raw)
            raise SuiteImportError(
                f"{source}: prompt entry {first!r} is not an inline template. "
                "File/function prompts can't be imported — inline the template."
            )
        if isinstance(first, str) and first.startswith("file://"):
            raise SuiteImportError(
                f"{source}: prompt '{first}' references a file. Inline the "
                "template text and re-import."
            )
        return str(first)
    raise SuiteImportError(f"{source}: unsupported 'prompts' shape ({type(prompts).__name__}).")


def _expected_for(assert_type: str, value: Any) -> Any:
    if assert_type == "llm-rubric":
        return {"rubric": str(value)}
    return value


def _pick_assert(asserts: list[dict], case_label: str,
                 warnings: list[str]) -> tuple[str, str | None, Any] | None:
    """Choose the assertion that scores this case; warn about the rest."""
    chosen: tuple[str, str | None, Any] | None = None
    for a in asserts:
        atype = str(a.get("type", ""))
        if atype not in _ASSERT_MAP:
            warnings.append(
                f"{case_label}: assert type '{atype}' has no Rift equivalent "
                "and was dropped."
            )
            continue
        if atype == "similar" and a.get("threshold") is not None:
            warnings.append(
                f"{case_label}: 'similar' threshold {a['threshold']} dropped — "
                "Rift's semantic scorer reports the raw cosine as the score "
                "instead of a pass/fail cut."
            )
        scoring, scorer = _ASSERT_MAP[atype]
        entry = (scoring, scorer, _expected_for(atype, a.get("value")))
        if chosen is None:
            chosen = entry
        else:
            warnings.append(
                f"{case_label}: extra assert '{atype}' dropped (a Rift case "
                "scores against one expectation; the first supported assert "
                "won)."
            )
    return chosen


def convert_promptfoo(
    source: Path,
    *,
    name: str | None = None,
    scoring_override: str | None = None,
    split_by_assert: bool = False,
) -> list[ImportedSuite]:
    """Convert a promptfoo config to one or more Rift suites."""
    with open(source) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise SuiteImportError(f"{source}: not a promptfoo config mapping.")

    tests = config.get("tests")
    if not isinstance(tests, list) or not tests:
        raise SuiteImportError(
            f"{source}: no inline 'tests' list found. External test files "
            "(tests: file://...) aren't followed — inline them and re-import."
        )

    warnings: list[str] = []
    prompt_template = _first_prompt(config, warnings, source)
    default_test = config.get("defaultTest") or {}
    default_vars = default_test.get("vars") or {}
    default_asserts = default_test.get("assert") or []

    # (scoring, custom_scorer) -> list of case dicts. Ordinary configs
    # produce exactly one group; mixed-assert configs produce several.
    groups: dict[tuple[str, str | None], list[dict]] = {}

    for i, test in enumerate(tests):
        if not isinstance(test, dict):
            warnings.append(f"case {i}: not a mapping; skipped.")
            continue
        label = f"case {i}" + (
            f" ({test['description']})" if test.get("description") else ""
        )
        variables = {**default_vars, **(test.get("vars") or {})}
        input_text = render_template(
            prompt_template, variables, source_desc=f"{source} {label}"
        )
        # Test-specific asserts first: promptfoo ANDs all of them, but a Rift
        # case scores one expectation, and the per-test assert is the
        # specific one — the shared defaultTest assert is the fallback.
        asserts = list(test.get("assert") or []) + list(default_asserts)
        if scoring_override:
            # --scoring overrides the asserts wholesale: score every case
            # with the given method against the first assert's value.
            value = asserts[0].get("value") if asserts else None
            if value is None:
                warnings.append(
                    f"{label}: --scoring override set but the case has no "
                    "assert value to use as 'expected'; skipped."
                )
                continue
            key: tuple[str, str | None] = (scoring_override, None)
            groups.setdefault(key, []).append(
                {"input": input_text, "expected": value, "tags": []}
            )
            continue
        picked = _pick_assert(asserts, label, warnings)
        if picked is None:
            warnings.append(f"{label}: no supported assert; case skipped.")
            continue
        scoring, scorer, expected = picked
        groups.setdefault((scoring, scorer), []).append(
            {"input": input_text, "expected": expected, "tags": []}
        )

    if not groups:
        raise SuiteImportError(
            f"{source}: no importable cases (see warnings above — every test "
            "was skipped)."
        )

    base_name = sanitize_name(name or source.stem)
    if len(groups) > 1 and not split_by_assert:
        listing = ", ".join(
            f"{scoring}" + (f" ({scorer.split(':')[1]})" if scorer else "")
            + f" ×{len(cases)}"
            for (scoring, scorer), cases in groups.items()
        )
        raise SuiteImportError(
            f"{source}: tests map to {len(groups)} different scoring methods "
            f"[{listing}], but a Rift suite has exactly one. Re-run with "
            "--split-by-assert to emit one suite per method."
        )

    suites: list[ImportedSuite] = []
    for (scoring, scorer), cases in groups.items():
        variant = ""
        if len(groups) > 1:
            variant = scorer.split(":")[1] if scorer else scoring
        suite_name = base_name + (f"_{variant}" if variant else "")
        desc = provenance_description(
            "promptfoo", source, len(cases), warnings,
            extra=(
                f"Scoring: {scoring}"
                + (f" via bundled scorer {scorer}" if scorer else "")
                + "."
            ),
        )
        suites.append(ImportedSuite(
            suite=build_suite(suite_name, desc, scoring, cases,
                              custom_scorer=scorer),
            warnings=list(warnings),
            variant=variant,
        ))
    return suites
