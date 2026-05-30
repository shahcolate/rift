"""User-overridable probe prompt templates.

Rift's probes ship with carefully-worded default prompts (the llm_judge rubric,
the faithfulness articulation judge, the answer-format instruction, the cue
templates, the CoT-perturbation templates). Those defaults are good general
choices, but a user with a specific use case — grading legal summaries, probing
faithfulness in a medical Q&A bot — often needs to tune the wording. This module
lets them do that **in the committed suite YAML** rather than by editing Rift's
source.

A suite may carry two override blocks::

    prompts:                     # key -> full template string
      judge_rubric: |
        You are grading ... {question} {target_block} {output} ...
    cues:                        # faithfulness cue name -> hint template
      authority: "Opposing counsel asserts the answer is {target}."

Overrides are validated at suite-load time: an unknown key, or a template that
drops a placeholder the probe needs to fill (e.g. a custom ``judge_rubric`` with
no ``{output}``), is a hard error — better a clear message at load than a
malformed prompt at runtime. The cache key for judge prompts already hashes the
full prompt text, so an override automatically re-scores rather than reusing a
default-prompt judgment.

Adding a new overridable prompt is a one-line :data:`PROMPT_REGISTRY` entry plus
a default in :func:`_default_for`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


@dataclass(frozen=True)
class PromptSpec:
    """Describes one overridable prompt template."""

    key: str
    required_placeholders: frozenset[str]
    description: str


# The single-string templates a suite may override. Cue-template overrides are
# handled separately (see ``cues`` / :func:`resolve_cues`) because they are a
# name->template map, not a fixed key.
PROMPT_REGISTRY: dict[str, PromptSpec] = {
    "judge_rubric": PromptSpec(
        "judge_rubric",
        frozenset({"question", "target_block", "output"}),
        "llm_judge grading prompt (0-1 rubric).",
    ),
    "faithfulness_judge": PromptSpec(
        "faithfulness_judge",
        frozenset({"cue", "reasoning", "answer", "target"}),
        "Faithfulness articulation judge (did reasoning admit the cue?).",
    ),
    "faithfulness_format_instruction": PromptSpec(
        "faithfulness_format_instruction",
        frozenset(),  # appended verbatim; must keep output parseable by Answer:
        "Answer-format instruction appended to faithfulness variants.",
    ),
    "faithfulness_wrong_answer": PromptSpec(
        "faithfulness_wrong_answer",
        frozenset({"question"}),
        "Proposer prompt that invents a plausible-wrong target per case.",
    ),
    "faithfulness_cot_early": PromptSpec(
        "faithfulness_cot_early",
        frozenset({"question", "partial"}),
        "CoT early-answering perturbation (truncated reasoning).",
    ),
    "faithfulness_cot_mistake": PromptSpec(
        "faithfulness_cot_mistake",
        frozenset({"question", "reasoning"}),
        "CoT mistake-injection perturbation (corrupted reasoning).",
    ),
}

# Placeholder every cue template must keep so a per-case wrong target can be
# injected.
CUE_REQUIRED_PLACEHOLDER = "target"


def _placeholders(template: str) -> set[str]:
    return set(_PLACEHOLDER_RE.findall(template))


def _check_formattable(label: str, template: str, allowed: frozenset[str]) -> None:
    """Ensure ``template.format(**allowed)`` won't raise at runtime.

    Catches the two footguns a present-placeholder check misses:

    * an **unknown** placeholder (``{foo}``) -> ``KeyError`` when the probe
      formats with only its own placeholders;
    * an **unescaped** literal brace (a JSON example written ``{"k": 1}``
      instead of ``{{"k": 1}}``) -> ``KeyError`` / ``ValueError`` from
      ``str.format``.

    We do a trial format with dummy values for every allowed placeholder so a
    malformed template fails at suite-load with a clear message instead of mid-
    run with a traceback after spending API calls.
    """
    dummy = {name: "x" for name in allowed}
    try:
        template.format(**dummy)
    except (KeyError, IndexError) as exc:
        raise ValueError(
            f"{label} references an undefined placeholder {exc}. Allowed "
            f"placeholders: {sorted('{' + a + '}' for a in allowed)}. If you "
            f"meant a literal brace (e.g. a JSON example), double it: "
            f"{{{{ and }}}}."
        ) from None
    except ValueError as exc:
        raise ValueError(
            f"{label} is not a valid format template ({exc}). Double any "
            f"literal braces as {{{{ and }}}}."
        ) from None


def _default_for(key: str) -> str:
    """Return the built-in default template for a registry key.

    Lazy imports keep the default text living next to the probe that uses it
    (no duplication, no circular import at module load).
    """
    if key == "judge_rubric":
        from .scoring.llm_judge import JUDGE_PROMPT_TEMPLATE
        return JUDGE_PROMPT_TEMPLATE
    if key == "faithfulness_judge":
        from .scoring.faithfulness_judge import JUDGE_PROMPT_TEMPLATE
        return JUDGE_PROMPT_TEMPLATE
    if key == "faithfulness_format_instruction":
        from .faithfulness import _FORMAT_INSTRUCTION
        return _FORMAT_INSTRUCTION
    if key == "faithfulness_wrong_answer":
        from .faithfulness import _WRONG_ANSWER_PROMPT
        return _WRONG_ANSWER_PROMPT
    if key == "faithfulness_cot_early":
        from .faithfulness import _COT_EARLY_TEMPLATE
        return _COT_EARLY_TEMPLATE
    if key == "faithfulness_cot_mistake":
        from .faithfulness import _COT_MISTAKE_TEMPLATE
        return _COT_MISTAKE_TEMPLATE
    raise KeyError(key)


def default_cues() -> dict[str, str]:
    from .faithfulness import CUES
    return dict(CUES)


def validate_overrides(
    prompts: dict[str, str] | None,
    cues: dict[str, str] | None,
) -> None:
    """Validate a suite's ``prompts`` / ``cues`` override blocks.

    Raises ``ValueError`` (surfaced as a pydantic validation error at suite
    load) on an unknown prompt key or a template missing a required
    placeholder. A no-op when both blocks are empty.
    """
    if prompts:
        for key, template in prompts.items():
            spec = PROMPT_REGISTRY.get(key)
            if spec is None:
                raise ValueError(
                    f"unknown prompt override '{key}'. Valid keys: "
                    f"{sorted(PROMPT_REGISTRY)}"
                )
            if not isinstance(template, str) or not template.strip():
                raise ValueError(f"prompt override '{key}' must be a non-empty string")
            missing = spec.required_placeholders - _placeholders(template)
            if missing:
                raise ValueError(
                    f"prompt override '{key}' is missing required "
                    f"placeholder(s) {sorted('{' + m + '}' for m in missing)}; "
                    f"it must contain {sorted('{' + p + '}' for p in spec.required_placeholders)}"
                )
            # The probe formats with exactly its declared placeholders, so any
            # extra placeholder or unescaped brace must be caught here.
            _check_formattable(f"prompt override '{key}'", template,
                               spec.required_placeholders)

    if cues:
        for name, template in cues.items():
            if not isinstance(template, str) or not template.strip():
                raise ValueError(f"cue override '{name}' must be a non-empty string")
            if CUE_REQUIRED_PLACEHOLDER not in _placeholders(template):
                raise ValueError(
                    f"cue override '{name}' must contain the "
                    f"{{{CUE_REQUIRED_PLACEHOLDER}}} placeholder so a per-case "
                    f"target can be injected"
                )
            # Cues are formatted with only {target}.
            _check_formattable(f"cue override '{name}'", template,
                               frozenset({CUE_REQUIRED_PLACEHOLDER}))


def resolve(key: str, prompts: dict[str, str] | None) -> str:
    """Return the override for ``key`` if present, else the built-in default."""
    if prompts and key in prompts:
        return prompts[key]
    return _default_for(key)


def resolve_cues(cues: dict[str, str] | None) -> dict[str, str]:
    """Merge cue overrides onto the defaults (overrides win; new names extend)."""
    merged = default_cues()
    if cues:
        merged.update(cues)
    return merged


def overridden_keys(prompts: dict[str, str] | None,
                    cues: dict[str, str] | None) -> list[str]:
    """List the override keys a suite actually sets, for report disclosure."""
    keys: list[str] = sorted(prompts) if prompts else []
    if cues:
        keys += [f"cue:{name}" for name in sorted(cues)]
    return keys
