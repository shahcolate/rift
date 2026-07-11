"""Shared plumbing for suite adapters: templating, chat flattening, emission."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from . import SuiteImportError

# ``{{ var }}`` / ``{{ doc.field }}`` — the flat-variable subset shared by
# promptfoo (nunjucks) and lm-eval (Jinja2). Filters, loops, and conditionals
# are NOT supported; _render_template refuses rather than mis-rendering.
_VAR_RE = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_.]*)\s*\}\}")
_BLOCK_RE = re.compile(r"\{%.*?%\}", re.DOTALL)
_FILTER_RE = re.compile(r"\{\{[^}]*\|[^}]*\}\}")


def render_template(template: str, variables: dict[str, Any],
                    *, source_desc: str) -> str:
    """Substitute ``{{var}}`` / ``{{doc.field}}`` placeholders.

    Rift deliberately does not depend on a template engine, so only the
    flat-variable subset both promptfoo and lm-eval commonly use is
    supported. Templates using engine features beyond that ({% blocks %},
    filters) fail loudly here — a silently mis-rendered prompt would change
    what the eval measures.
    """
    if _BLOCK_RE.search(template):
        raise SuiteImportError(
            f"{source_desc}: template uses {{% ... %}} blocks, which the "
            "importer does not evaluate. Pre-render the prompts (or simplify "
            "the template to flat {{var}} substitutions) and re-import."
        )
    if _FILTER_RE.search(template):
        raise SuiteImportError(
            f"{source_desc}: template uses filters ('|' inside {{{{...}}}}), "
            "which the importer does not evaluate. Pre-render the prompts "
            "and re-import."
        )

    missing: list[str] = []

    def _sub(m: re.Match[str]) -> str:
        path = m.group(1)
        value: Any = variables
        for part in path.split("."):
            # lm-eval templates address fields as {{doc.field}} while the
            # doc dict holds plain {field}; treat a leading 'doc.' (or
            # 'item.') as transparent when the container has no such key.
            if isinstance(value, dict) and part in value:
                value = value[part]
            elif isinstance(value, dict) and part in ("doc", "item") and value is variables:
                continue
            else:
                missing.append(path)
                return m.group(0)
        return str(value)

    rendered = _VAR_RE.sub(_sub, template)
    if missing:
        raise SuiteImportError(
            f"{source_desc}: template references undefined variable(s) "
            f"{sorted(set(missing))}; available: {sorted(variables)}"
        )
    return rendered


def flatten_chat(messages: list[Any], *, warnings: list[str],
                 case_label: str) -> str:
    """Flatten a chat-message list to the single prompt string Rift sends.

    Rift providers send one user message, so multi-turn structure cannot
    be preserved. A system message is kept as a labelled preamble; user
    turns are concatenated; assistant turns (few-shot demonstrations) are
    kept inline as labelled context. The loss is disclosed via a warning.
    """
    parts: list[str] = []
    roles_seen: set[str] = set()
    for msg in messages:
        if not isinstance(msg, dict):
            parts.append(str(msg))
            continue
        role = str(msg.get("role", "user"))
        content = msg.get("content", "")
        if isinstance(content, list):  # multi-part content blocks
            content = "\n".join(
                str(b.get("text", b)) if isinstance(b, dict) else str(b)
                for b in content
            )
        roles_seen.add(role)
        if role == "user":
            parts.append(str(content))
        else:
            parts.append(f"[{role}]\n{content}")
    if roles_seen - {"user"}:
        warnings.append(
            f"{case_label}: chat input flattened to a single prompt string "
            f"(roles: {', '.join(sorted(roles_seen))}). Rift sends one user "
            "message; system/assistant turns became labelled preamble text."
        )
    return "\n\n".join(parts).strip()


def stringify_tags(metadata: dict[str, Any] | None,
                   sample_id: Any = None) -> list[str]:
    """Map foreign metadata to Rift's ``key:value`` tag convention.

    Scalar metadata becomes ``k:v`` tags, which plugs straight into
    ``rift compare --subgroup k:`` for per-group drift tables. Nested
    values are skipped (tags are flat by design).
    """
    tags: list[str] = []
    if sample_id is not None:
        tags.append(f"id:{sample_id}")
    for k, v in (metadata or {}).items():
        if isinstance(v, (str, int, float, bool)):
            tags.append(f"{k}:{v}")
    return tags


def provenance_description(source_format: str, source: Path,
                           n_cases: int, warnings: list[str],
                           extra: str = "") -> str:
    """Build the emitted suite's ``description``: provenance + loss report.

    The description travels with the YAML wherever it is copied, so every
    lossy transform is recorded there, not only in the import-time output.
    """
    lines = [
        f"Imported by `rift import --from {source_format}` from {source.name} "
        f"({n_cases} case{'s' if n_cases != 1 else ''}).",
    ]
    if extra:
        lines.append(extra)
    if warnings:
        lines.append("")
        lines.append(
            "Import caveats (transforms that may change what the eval "
            "measures):"
        )
        # Dedupe while preserving order; per-case repeats collapse to one line.
        seen: set[str] = set()
        for w in warnings:
            key = re.sub(r"case \d+", "case N", w)
            if key not in seen:
                seen.add(key)
                lines.append(f"- {w}")
    return "\n".join(lines)


def build_suite(name: str, description: str, scoring: str,
                cases: list[dict], custom_scorer: str | None = None) -> dict:
    """Assemble the SuiteConfig-shaped dict all adapters emit."""
    suite: dict[str, Any] = {
        "name": name,
        "description": description,
        "scoring": scoring,
    }
    if custom_scorer:
        suite["custom_scorer"] = custom_scorer
    suite["cases"] = cases
    return suite


def sanitize_name(raw: str) -> str:
    """Reduce an arbitrary source name to a suite-name-safe slug."""
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", raw).strip("_")
    return slug or "imported_suite"
