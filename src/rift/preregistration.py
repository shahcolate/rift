"""Pre-registration: pin the primary endpoint before the run.

A single ``rift compare`` can surface many numbers — overall accuracy, each
subgroup, cost-per-correct, refusal, calibration. Reading the headline off
whichever one happened to cross p<0.05 is the garden of forking paths, and it
inflates the false-positive rate far above the nominal alpha.

A pre-registration is a tiny YAML committed *before* the run that names the ONE
primary endpoint, its direction, alpha, and the minimum sample size. Rift then:

* reports the pre-registered endpoint as the headline and the gate,
* labels everything else as **exploratory** (hypothesis-generating, not
  confirmatory),
* records any protocol violation (too few cases, wrong suite/models),

so a published drift claim can state "pre-registered primary endpoint" instead
of being a post-hoc pick. This is the cleanest defense against multiplicity:
designate one test, not correct twenty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import click
import yaml
from pydantic import BaseModel, ValidationError


class PreregError(click.ClickException):
    """A pre-registration file failed to load or validate."""

    exit_code = 1


class Preregistration(BaseModel):
    """A pre-registered analysis plan for one comparison."""

    # The one endpoint the gate and headline are bound to.
    primary: Literal["accuracy", "cost_per_correct"] = "accuracy"
    # Adverse direction we are testing for. "regression" means a worse value
    # (lower accuracy, or higher cost-per-correct); "two_sided" gates on any
    # significant move.
    direction: Literal["regression", "improvement", "two_sided"] = "regression"
    alpha: float = 0.05
    # Minimum paired cases required for the run to honor the plan.
    min_cases: int = 0
    # Optional declared effect we powered for (reported, not enforced).
    target_effect: float | None = None
    # Optional identity locks — flagged as violations if the run differs.
    suite: str | None = None
    baseline: str | None = None
    challenger: str | None = None
    # Free-text hypothesis, echoed into the report for the record.
    hypothesis: str | None = None


@dataclass
class PreregOutcome:
    """Result of judging a run against its pre-registration."""

    primary: str
    direction: str
    alpha: float
    honored: bool                 # no protocol violations
    violations: list[str]
    primary_delta: float
    primary_significant: bool     # at the pre-registered alpha
    adverse_confirmed: bool       # the pre-registered adverse outcome held
    detail: str                   # one-line human summary of the primary test
    hypothesis: str | None = None
    exploratory_notes: list[str] = field(default_factory=list)


def load_preregistration(path: str | Path) -> Preregistration:
    """Load and validate a pre-registration YAML, with clean CLI errors."""
    p = Path(path)
    if not p.exists():
        raise PreregError(f"Pre-registration file not found: {path}")
    with open(p) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise PreregError(f"Pre-registration '{path}' must be a YAML mapping.")
    try:
        return Preregistration(**data)
    except ValidationError as e:
        detail = str(e).split("For further information", 1)[0].strip()
        raise PreregError(f"Invalid pre-registration '{path}':\n{detail}") from None


def evaluate(prereg: Preregistration, drift, n_cases: int,
             baseline_model: str | None = None,
             challenger_model: str | None = None) -> PreregOutcome:
    """Judge a :class:`rift.comparator.DriftResult` against ``prereg``.

    Returns a :class:`PreregOutcome` whose ``adverse_confirmed`` is the gate:
    True iff the pre-registered adverse move on the primary endpoint is
    statistically supported at the pre-registered alpha. Identity/sample-size
    mismatches are recorded as ``violations`` (the plan is dishonored) but do
    not by themselves flip the gate — they qualify the claim.

    Operating level of directional plans: the accuracy p-value is two-sided
    (McNemar / paired t), so a plan declaring ``direction: regression`` at
    alpha rejects only when the two-sided p clears alpha AND the delta is
    adverse — an effective one-sided level of about alpha/2. This is
    conservative (it never overstates significance) and matches what
    ``rift selftest`` reports empirically as the false-regression rate
    (≈ alpha/2). See docs/methodology.md § known caveats.
    """
    # Resolve aliases on both sides so a plan pinning the full id
    # ('claude-opus-4-7') is honored by a run that used the alias ('opus-4-7'),
    # and vice versa — otherwise every aliased run is a spurious violation.
    from .config import resolve_model

    def _canon(name: str | None) -> str | None:
        if not name:
            return None
        try:
            return resolve_model(name).model
        except Exception:
            return name

    violations: list[str] = []
    if prereg.suite and drift.suite_name and prereg.suite not in drift.suite_name:
        violations.append(
            f"suite mismatch: pre-registered '{prereg.suite}', "
            f"ran '{drift.suite_name}'"
        )
    if (prereg.baseline and baseline_model
            and _canon(prereg.baseline) != _canon(baseline_model)):
        violations.append(
            f"baseline mismatch: pre-registered '{prereg.baseline}', "
            f"ran '{baseline_model}'"
        )
    if (prereg.challenger and challenger_model
            and _canon(prereg.challenger) != _canon(challenger_model)):
        violations.append(
            f"challenger mismatch: pre-registered '{prereg.challenger}', "
            f"ran '{challenger_model}'"
        )
    if n_cases < prereg.min_cases:
        violations.append(
            f"underpowered: {n_cases} cases < pre-registered min_cases "
            f"{prereg.min_cases}"
        )

    if prereg.primary == "accuracy":
        delta = drift.delta
        significant = drift.p_value < prereg.alpha
        # Adverse = worse accuracy = negative delta.
        adverse_delta = delta < 0
        detail = (f"accuracy Δ={delta:+.4f}, p={drift.p_value:.4f} "
                  f"(α={prereg.alpha})")
    else:  # cost_per_correct
        delta = drift.cost_normalized_delta_usd
        # No p-value for the cost delta; significance is the bootstrap CI
        # excluding zero. The CI's level follows the comparison's alpha
        # (compare_runs computes it at 1 − alpha, and the CLI sets that alpha
        # from this plan), so a pre-registered alpha: 0.01 really is gated on
        # a 99% interval — verify ci_level matches rather than assume.
        ci_level = getattr(drift, "ci_level", 0.95)
        expected_level = round(1.0 - prereg.alpha, 4)
        if abs(ci_level - expected_level) > 1e-9:
            violations.append(
                f"cost CI level mismatch: pre-registered α={prereg.alpha} "
                f"needs a {expected_level:.0%} CI, but the comparison was "
                f"run with a {ci_level:.0%} CI (was compare_runs called "
                "with the plan's alpha?)"
            )
        if getattr(drift, "cost_delta_ci_defined", False):
            lo, hi = drift.cost_delta_ci_lower, drift.cost_delta_ci_upper
            significant = lo > 0 or hi < 0
            detail = (f"$/correct Δ={delta:+.4f}, {ci_level:.0%} CI "
                      f"[{lo:+.4f}, {hi:+.4f}]")
        else:
            significant = False
            detail = (f"$/correct Δ={delta:+.4f}, CI undefined "
                      "(no cost CI available)")
        # Adverse = more expensive per correct = positive delta.
        adverse_delta = delta > 0

    if prereg.direction == "regression":
        adverse_confirmed = significant and adverse_delta
    elif prereg.direction == "improvement":
        adverse_confirmed = significant and not adverse_delta
    else:  # two_sided
        adverse_confirmed = significant

    return PreregOutcome(
        primary=prereg.primary,
        direction=prereg.direction,
        alpha=prereg.alpha,
        honored=not violations,
        violations=violations,
        primary_delta=round(delta, 6),
        primary_significant=significant,
        adverse_confirmed=adverse_confirmed,
        detail=detail,
        hypothesis=prereg.hypothesis,
    )
