"""Bundled scorers for imported suites.

Foreign harnesses ship assertion types Rift's built-in scorers don't
cover (substring containment, regex). Imported suites reference these via
``scoring: custom`` + ``custom_scorer: "rift.adapters.scorers:<fn>"`` so
the emitted YAML is self-contained: it runs on any machine with Rift
installed, no user-authored scorer file required.

All scorers are binary (0.0 / 1.0) so imported suites flow through the
McNemar paired-test path, and all coerce ``output``/``expected`` to
strings — foreign formats frequently carry numbers where Rift expects
text.
"""

from __future__ import annotations

import re
from typing import Any


def contains(output: str, expected: Any) -> float:
    """1.0 iff ``expected`` appears verbatim in the output."""
    return 1.0 if str(expected) in str(output) else 0.0


def icontains(output: str, expected: Any) -> float:
    """Case-insensitive :func:`contains`."""
    return 1.0 if str(expected).lower() in str(output).lower() else 0.0


def starts_with(output: str, expected: Any) -> float:
    """1.0 iff the output starts with ``expected`` (leading whitespace ignored)."""
    return 1.0 if str(output).lstrip().startswith(str(expected)) else 0.0


def regex_match(output: str, expected: Any) -> float:
    """1.0 iff the ``expected`` regular expression matches anywhere in the output."""
    return 1.0 if re.search(str(expected), str(output)) else 0.0


def contains_any(output: str, expected: Any) -> float:
    """1.0 iff any element of ``expected`` (a list) appears in the output."""
    items = expected if isinstance(expected, (list, tuple)) else [expected]
    text = str(output)
    return 1.0 if any(str(item) in text for item in items) else 0.0


def contains_all(output: str, expected: Any) -> float:
    """1.0 iff every element of ``expected`` (a list) appears in the output."""
    items = expected if isinstance(expected, (list, tuple)) else [expected]
    text = str(output)
    return 1.0 if all(str(item) in text for item in items) else 0.0
