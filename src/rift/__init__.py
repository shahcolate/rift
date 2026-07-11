"""Rift: Detect behavioral regressions between LLM model versions.

Library use — the same primitives the CLI drives::

    import rift

    suite = rift.load_suite("suites/mine.yaml")     # or a built-in name
    base = await rift.run_suite(suite, rift.resolve_model("claude-opus-4-7"))
    chal = await rift.run_suite(suite, rift.resolve_model("claude-opus-4-8"))
    drift = rift.compare_runs(
        base.scores, chal.scores, base.model, chal.model, suite.name,
        baseline_costs=[c.cost_usd for c in base.cases],
        challenger_costs=[c.cost_usd for c in chal.cases],
    )
    if drift.significant and drift.delta < 0:
        ...  # gate the deploy

Runs produced elsewhere work too: ``rift.compare_runs`` only needs two
paired score vectors, so Rift can be the statistics layer over any
harness's outputs (see also ``rift import`` for converting whole
suites).

Exports resolve lazily (PEP 562): ``import rift`` stays cheap, and heavy
dependencies load only when the symbol is first touched.
"""

from typing import TYPE_CHECKING

__version__ = "1.1.0"

# Public API surface. Everything here follows semver: breaking changes to
# these names/signatures require a major version bump.
_EXPORTS = {
    # Suites & models
    "load_suite": ("rift.config", "load_suite"),
    "SuiteConfig": ("rift.config", "SuiteConfig"),
    "EvalCase": ("rift.config", "EvalCase"),
    "resolve_model": ("rift.config", "resolve_model"),
    "ModelConfig": ("rift.config", "ModelConfig"),
    # Execution
    "run_suite": ("rift.runner", "run_suite"),
    "RunResult": ("rift.runner", "RunResult"),
    "CaseResult": ("rift.runner", "CaseResult"),
    # Statistics
    "compare_runs": ("rift.comparator", "compare_runs"),
    "compare_by_subgroup": ("rift.comparator", "compare_by_subgroup"),
    "DriftResult": ("rift.comparator", "DriftResult"),
    "benjamini_hochberg": ("rift.comparator", "benjamini_hochberg"),
    "power_analysis": ("rift.comparator", "power_analysis"),
    "variance_components": ("rift.comparator", "variance_components"),
    "cohens_kappa": ("rift.comparator", "cohens_kappa"),
    # Scoring
    "get_scorer": ("rift.scoring", "get_scorer"),
    # Reporting
    "generate_markdown_report": ("rift.reporter", "generate_markdown_report"),
}

__all__ = [
    "CaseResult",
    "DriftResult",
    "EvalCase",
    "ModelConfig",
    "RunResult",
    "SuiteConfig",
    "__version__",
    "benjamini_hochberg",
    "cohens_kappa",
    "compare_by_subgroup",
    "compare_runs",
    "generate_markdown_report",
    "get_scorer",
    "load_suite",
    "power_analysis",
    "resolve_model",
    "run_suite",
    "variance_components",
]


def __getattr__(name: str):
    try:
        module_name, attr = _EXPORTS[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None
    import importlib

    value = getattr(importlib.import_module(module_name), attr)
    globals()[name] = value  # cache so subsequent lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


if TYPE_CHECKING:  # static analyzers see real symbols; runtime stays lazy
    from rift.comparator import (
        DriftResult as DriftResult,
        benjamini_hochberg as benjamini_hochberg,
        cohens_kappa as cohens_kappa,
        compare_by_subgroup as compare_by_subgroup,
        compare_runs as compare_runs,
        power_analysis as power_analysis,
        variance_components as variance_components,
    )
    from rift.config import (
        EvalCase as EvalCase,
        ModelConfig as ModelConfig,
        SuiteConfig as SuiteConfig,
        load_suite as load_suite,
        resolve_model as resolve_model,
    )
    from rift.reporter import generate_markdown_report as generate_markdown_report
    from rift.runner import (
        CaseResult as CaseResult,
        RunResult as RunResult,
        run_suite as run_suite,
    )
    from rift.scoring import get_scorer as get_scorer
