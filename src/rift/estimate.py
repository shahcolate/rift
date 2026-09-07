"""Keyless pre-flight cost estimates: know the bill before you run.

``rift estimate`` answers "what will this cost?" for a model × suite grid
or a whole observatory panel pass, without an API key and without a
single call. It reuses the same heuristic the Observatory's budget guard
uses (:func:`rift.observatory.estimate_stage_cost`: prompt chars / 4 for
input, a flat per-case output allowance) so the number you see here is
the number the guard would compare against ``max_cost_usd``.

The estimate is deliberately an order-of-magnitude figure, not a quote.
Two systematic errors to keep in mind:

* **Thinking models run over on output.** Fable 5 / 5.1 and Opus 5 bill
  reasoning as output tokens. On the standard suites the live Fable 5
  run (benchmarks/fable5_vs_opus47) averaged ~100 output tokens/case
  — *under* the 300 default — but on ``hard_reasoning`` it averaged
  ~750/case, 2.5× over. Pass ``--output-tokens-per-case`` when you know
  the workload, or ``--calibrate-from run.json`` to reuse a saved run's
  measured token counts for the same suite.
* **List price, standard mode only.** Batch is −50%, fast mode is a
  premium, cache reads are cheaper — see :mod:`rift.pricing`.

A model with no pricing entry is estimated at the catalog maximum (the
same conservative bound the budget guard applies) and flagged, so a new
release never silently estimates at $0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import SuiteConfig, load_suite, resolve_model
from .observatory import EST_OUTPUT_TOKENS_PER_CASE, PanelConfig
from .pricing import lookup, most_expensive


@dataclass
class StageEstimate:
    """One model × suite cell of the estimate grid."""

    model: str            # as the user typed it (alias or id)
    resolved_model: str   # canonical id the price was looked up on
    suite: str
    n_cases: int
    calls: int            # n_cases × trials (× 2 with a pushback stage)
    input_tokens: int
    output_tokens: int
    cost_usd: float
    priced: bool          # False → estimated at the catalog maximum
    keyless: bool = False  # RiftLM / self-hosted: $0 API spend
    note: str = ""


@dataclass
class Estimate:
    stages: list[StageEstimate] = field(default_factory=list)

    @property
    def total_usd(self) -> float:
        return sum(s.cost_usd for s in self.stages)

    @property
    def unpriced_models(self) -> list[str]:
        return sorted({s.model for s in self.stages
                       if not s.priced and not s.keyless})


def _tokens_for(suite: SuiteConfig, output_per_case: int,
                calibration: dict[str, dict] | None,
                resolved_model: str) -> tuple[int, int, str]:
    """(input_tokens, output_tokens, source) for ONE pass over ``suite``.

    ``calibration`` maps model id → measured ``{input_tokens,
    output_tokens}`` for this suite. The estimated model's own row wins;
    otherwise the heaviest row is used (conservative — a thinking model's
    output profile is the one you don't want to under-budget).
    """
    if calibration:
        row = calibration.get(resolved_model)
        src = f"measured ({resolved_model})"
        if row is None:
            heaviest = max(calibration, key=lambda m: calibration[m]["output_tokens"])
            row, src = calibration[heaviest], f"measured on {heaviest}"
        return int(row["input_tokens"]), int(row["output_tokens"]), src
    est_in = sum(len(c.input) // 4 for c in suite.cases)
    return est_in, output_per_case * len(suite.cases), "heuristic"


def estimate_stage(model: str, suite: SuiteConfig, *, trials: int = 1,
                   output_per_case: int = EST_OUTPUT_TOKENS_PER_CASE,
                   calibration: dict[str, dict] | None = None,
                   pushback: bool = False,
                   suite_label: str | None = None) -> StageEstimate:
    """Estimate one model × suite stage.

    ``pushback=True`` adds the sycophancy follow-up pass: every case is
    re-asked with the model's own answer embedded and a challenge
    appended, so the probe roughly doubles the calls and runs ~1.5× the
    base input per call. Modelled as +1.5× input, +1× output.
    """
    cfg = resolve_model(model)
    keyless = cfg.provider in ("riftlm", "openai_compatible", "local")
    price = lookup(cfg.model)
    priced = price is not None
    note = ""
    if keyless:
        note = "no API spend (local inference)"
    elif not priced:
        price = most_expensive()
        note = "NOT in pricing catalog — estimated at the catalog maximum"

    inp, out, source = _tokens_for(suite, output_per_case, calibration, cfg.model)
    if source != "heuristic":
        note = (note + "; " if note else "") + source
    calls = len(suite.cases) * max(1, trials)
    inp, out = inp * max(1, trials), out * max(1, trials)
    if pushback:
        inp += int(inp * 1.5)
        out += out
        calls *= 2
    cost = 0.0 if keyless else price.cost(inp, out)  # type: ignore[union-attr]
    return StageEstimate(
        model=model, resolved_model=cfg.model, suite=suite_label or suite.name,
        n_cases=len(suite.cases), calls=calls, input_tokens=inp,
        output_tokens=out, cost_usd=cost, priced=priced, keyless=keyless,
        note=note,
    )


def _calibration_for(calibrations: dict[str, dict[str, dict]] | None,
                     name: str, suite: SuiteConfig) -> dict[str, dict] | None:
    if not calibrations:
        return None
    return calibrations.get(suite.name) or calibrations.get(name)


def estimate_grid(models: list[str], suites: list[str], *, trials: int = 1,
                  output_per_case: int = EST_OUTPUT_TOKENS_PER_CASE,
                  calibrations: dict[str, dict[str, dict]] | None = None) -> Estimate:
    """Estimate every model × suite pair (what ``compare``/``matrix`` would run)."""
    est = Estimate()
    loaded = {name: load_suite(name) for name in suites}
    for name, suite in loaded.items():
        cal = _calibration_for(calibrations, name, suite)
        for model in models:
            est.stages.append(estimate_stage(
                model, suite, trials=trials, output_per_case=output_per_case,
                calibration=cal, suite_label=name,
            ))
    return est


def estimate_panel(panel: PanelConfig, *,
                   output_per_case: int = EST_OUTPUT_TOKENS_PER_CASE,
                   calibrations: dict[str, dict[str, dict]] | None = None) -> Estimate:
    """Estimate one full observatory pass: every endpoint × suite, plus the
    sycophancy pushback stage on ``panel.sycophancy_on``."""
    est = Estimate()
    loaded = {name: load_suite(name) for name in panel.suites}
    for name, suite in loaded.items():
        cal = _calibration_for(calibrations, name, suite)
        for ep in panel.endpoints:
            stage = estimate_stage(
                ep.model, suite, output_per_case=output_per_case,
                calibration=cal, pushback=(name == panel.sycophancy_on),
                suite_label=name,
            )
            stage.model = ep.id
            est.stages.append(stage)
    return est


def calibration_from_run(path: str | Path) -> tuple[str, dict[str, dict]]:
    """Measured token totals from a saved ``rift run`` or ``compare`` JSON.

    Returns ``(suite_name, {model_id: {"input_tokens", "output_tokens"}})``
    — one row per side of a comparison, one for a bare run — so the
    estimate for that suite uses observed usage, the same trick the
    Observatory's budget guard plays with the prior week's record.
    """
    import json

    with open(path) as f:
        data = json.load(f)
    runs = [data[k] for k in ("baseline", "challenger") if isinstance(data.get(k), dict)]
    if not runs:
        runs = [data.get("run") or data]
    rows: dict[str, dict] = {}
    suite_name = Path(path).stem
    for run in runs:
        cases = run.get("cases", [])
        suite_name = run.get("suite_name", suite_name)
        rows[run.get("model", "?")] = {
            "input_tokens": sum(int(c.get("input_tokens", 0)) for c in cases),
            "output_tokens": sum(int(c.get("output_tokens", 0)) for c in cases),
        }
    return suite_name, rows
