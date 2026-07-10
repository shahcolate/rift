"""The operational-error base for Rift's exit-code contract.

Exit codes are a CI contract: 0 = no significant regression, 1 =
significant regression (the gate — ONLY the gate), 2 = operational
error. Every user-fixable failure (bad file, unknown model, missing
key, malformed plan) subclasses :class:`OperationalError` so the
contract is structural — inherited, not re-typed per class — and a CI
job can never misread an infrastructure problem as model drift.
"""

from __future__ import annotations

import click


class OperationalError(click.ClickException):
    """A user-fixable infrastructure error. Clean message, exit 2."""

    exit_code = 2
