"""Suite adapters: import eval suites from other harnesses.

``rift import --from promptfoo|inspect|lm-eval|openai-evals`` converts a
foreign eval definition into a Rift suite YAML, so teams keep the evals
they already own and Rift becomes the statistics layer on top.

Conversion is deliberately conservative and **loud about loss**. Rift's
suite model is narrower than some sources (one scoring method per suite,
one ``expected`` per case, a single prompt string per case), so an import
can be lossy — every dropped assertion, flattened chat transcript, or
approximated scoring mode is reported as a warning at import time and
recorded in the emitted suite's ``description``. An import that silently
changed what an eval measures would defeat the point of importing it.

Every emitted suite is round-trip validated through
:func:`rift.config.load_suite` before the importer reports success.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .._errors import OperationalError


class SuiteImportError(OperationalError):
    """A source file could not be converted to a Rift suite.

    An :class:`~rift._errors.OperationalError`, so a malformed or
    unsupported source produces a clean CLI message and exit 2, never a
    traceback.
    """


@dataclass
class ImportedSuite:
    """One converted suite plus everything the user must know about it.

    ``variant`` is non-empty when a single source file yielded several
    suites (promptfoo ``--split-by-assert``); it is appended to the
    output filename so each suite lands in its own file.
    """

    suite: dict  # SuiteConfig-shaped; yaml.safe_dump-ready
    warnings: list[str] = field(default_factory=list)
    variant: str = ""


SUPPORTED_FORMATS = ("promptfoo", "inspect", "lm-eval", "openai-evals")


def convert(
    source_format: str,
    source: str | Path,
    *,
    name: str | None = None,
    scoring: str | None = None,
    dataset: str | Path | None = None,
    split_by_assert: bool = False,
) -> list[ImportedSuite]:
    """Convert ``source`` (a file in ``source_format``) to Rift suites.

    Returns one :class:`ImportedSuite` per emitted suite — usually a
    single element; promptfoo configs with mixed assertion types return
    one per assertion group under ``split_by_assert=True``.
    """
    source = Path(source)
    if source_format == "promptfoo":
        from .promptfoo import convert_promptfoo

        return convert_promptfoo(
            source, name=name, scoring_override=scoring,
            split_by_assert=split_by_assert,
        )
    if source_format == "inspect":
        from .inspect_ai import convert_inspect

        return [convert_inspect(source, name=name, scoring=scoring)]
    if source_format == "lm-eval":
        from .lmeval import convert_lmeval

        return [convert_lmeval(source, dataset=dataset, name=name,
                               scoring_override=scoring)]
    if source_format == "openai-evals":
        from .openai_evals import convert_openai_evals

        return [convert_openai_evals(source, name=name, scoring=scoring)]
    raise SuiteImportError(
        f"Unknown source format '{source_format}'. "
        f"Supported: {', '.join(SUPPORTED_FORMATS)}."
    )
