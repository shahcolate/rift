"""Calibration drift: does the model's stated confidence track its correctness?

A model that says "I am 90% sure" on outputs it gets right 60% of
the time is *overconfident*. The standard summary numbers for this
are:

* **Brier score** — mean squared error between predicted confidence
  and the 0/1 correctness indicator. Lower is better. A Brier of 0
  is perfect, 0.25 is the score of always-50% on a balanced task.
* **Expected Calibration Error (ECE)** — bins predictions by
  confidence, computes |confidence − accuracy| within each bin,
  averages weighted by bin size. The interpretable "how much do you
  miss the diagonal of the reliability diagram" number.

Calibration *drift* is the difference in these between two model
versions. It is independent of accuracy: a model can get strictly
better on Brier while getting worse on raw accuracy (became less
confidently wrong), or vice versa. Production teams care about
both.

We parse confidence from the model's text output rather than from
logits because that's the only thing portable across providers. The
parser is tolerant — it accepts ``Confidence: 0.85``, ``confidence
85%``, ``I am 85% sure``, and a few other shapes. Cases where no
confidence can be parsed are counted in ``n_unparsed`` and excluded
from the metrics (rather than silently coerced to 0.5, which would
bias both runs toward looking miscalibrated together).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


_CONFIDENCE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE) for p in [
        # "Confidence: 0.85" / "confidence = 85%" / "confidence 0.85"
        r"confidence\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(%?)",
        # "I am 85% sure" / "I'm 0.85 confident"
        r"\bi(?:'m| am)\s+(\d+(?:\.\d+)?)\s*(%?)\s*(?:sure|confident|certain)",
        # Standalone trailing "(0.85)" right after a final answer line
        r"\bp\s*[:=]\s*(\d+(?:\.\d+)?)\s*(%?)",
    ]
)


@dataclass
class CalibrationStats:
    """Calibration summary for a single run."""
    n_cases: int
    n_parsed: int
    n_unparsed: int
    brier: float                 # NaN if n_parsed == 0
    ece: float                   # NaN if n_parsed == 0
    accuracy: float              # over parsed subset
    mean_confidence: float       # over parsed subset
    overconfidence: float        # mean_confidence − accuracy; >0 = overconfident
    bins: list[dict]             # reliability diagram bins (for plots / md)


@dataclass
class CalibrationComparison:
    """Calibration drift between two runs."""
    baseline: CalibrationStats
    challenger: CalibrationStats
    delta_brier: float       # challenger − baseline; negative is better
    delta_ece: float         # challenger − baseline; negative is better
    delta_overconfidence: float


def parse_confidence(text: str) -> float | None:
    """Extract a confidence value in [0, 1] from a model output.

    Returns ``None`` if no recognizable confidence is present. ``%``
    forms are divided by 100; bare floats >1 are also interpreted as
    percentages (a model that writes "Confidence: 85" almost
    certainly means 85%, not 8500%). Values are clamped to [0, 1].
    """
    if not text:
        return None
    for pat in _CONFIDENCE_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        raw = float(m.group(1))
        is_pct = m.group(2) == "%" if m.lastindex and m.lastindex >= 2 else False
        if is_pct or raw > 1.0:
            raw = raw / 100.0
        return float(np.clip(raw, 0.0, 1.0))
    return None


def _brier(confs: np.ndarray, correct: np.ndarray) -> float:
    """Mean squared error between confidence and 0/1 correctness."""
    if confs.size == 0:
        return float("nan")
    return float(np.mean((confs - correct) ** 2))


def _ece(confs: np.ndarray, correct: np.ndarray, n_bins: int = 10
         ) -> tuple[float, list[dict]]:
    """Expected calibration error with equal-width bins on [0, 1].

    Returns ``(ece, bins)`` where each bin dict carries
    ``{"lo","hi","count","mean_conf","accuracy","gap"}`` for use in
    reliability-diagram rendering downstream.
    """
    if confs.size == 0:
        return float("nan"), []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = confs.size
    ece = 0.0
    bins: list[dict] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (confs >= lo) & (confs <= hi)
        else:
            mask = (confs >= lo) & (confs < hi)
        count = int(mask.sum())
        if count == 0:
            bins.append({"lo": float(lo), "hi": float(hi), "count": 0,
                         "mean_conf": 0.0, "accuracy": 0.0, "gap": 0.0})
            continue
        mean_conf = float(confs[mask].mean())
        acc = float(correct[mask].mean())
        gap = abs(mean_conf - acc)
        ece += (count / n) * gap
        bins.append({"lo": float(lo), "hi": float(hi), "count": count,
                     "mean_conf": mean_conf, "accuracy": acc, "gap": gap})
    return float(ece), bins


def compute_calibration(run, correctness_threshold: float = 0.999
                        ) -> CalibrationStats:
    """Compute Brier, ECE, and overconfidence for one run.

    Only cases with a parseable confidence contribute; the count of
    unparseable cases is surfaced so the caller can decide whether
    the suite needs a prompt change ("answer in this format:
    ``Confidence: X``").
    """
    confs: list[float] = []
    correct: list[int] = []
    n_unparsed = 0
    for case in run.cases:
        out = getattr(case, "output", "") or ""
        conf = parse_confidence(out)
        if conf is None:
            n_unparsed += 1
            continue
        confs.append(conf)
        correct.append(1 if float(case.score) >= correctness_threshold else 0)
    confs_a = np.asarray(confs, dtype=float)
    correct_a = np.asarray(correct, dtype=float)
    if confs_a.size == 0:
        return CalibrationStats(
            n_cases=len(run.cases),
            n_parsed=0,
            n_unparsed=n_unparsed,
            brier=float("nan"),
            ece=float("nan"),
            accuracy=0.0,
            mean_confidence=0.0,
            overconfidence=0.0,
            bins=[],
        )
    brier = _brier(confs_a, correct_a)
    ece, bins = _ece(confs_a, correct_a)
    acc = float(correct_a.mean())
    mean_conf = float(confs_a.mean())
    return CalibrationStats(
        n_cases=len(run.cases),
        n_parsed=int(confs_a.size),
        n_unparsed=n_unparsed,
        brier=round(brier, 4),
        ece=round(ece, 4),
        accuracy=round(acc, 4),
        mean_confidence=round(mean_conf, 4),
        overconfidence=round(mean_conf - acc, 4),
        bins=bins,
    )


def compare_calibration(baseline_run, challenger_run
                        ) -> CalibrationComparison:
    """Calibration drift between two runs.

    Negative ``delta_brier`` and ``delta_ece`` are improvements
    (challenger is better calibrated). ``delta_overconfidence`` shows
    whether the new model talks a bigger game than its accuracy
    earns it.
    """
    b = compute_calibration(baseline_run)
    c = compute_calibration(challenger_run)

    def _diff(x: float, y: float) -> float:
        if np.isnan(x) or np.isnan(y):
            return float("nan")
        return round(y - x, 4)

    return CalibrationComparison(
        baseline=b,
        challenger=c,
        delta_brier=_diff(b.brier, c.brier),
        delta_ece=_diff(b.ece, c.ece),
        delta_overconfidence=_diff(b.overconfidence, c.overconfidence),
    )
