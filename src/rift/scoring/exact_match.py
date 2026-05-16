"""Exact match scoring for structured outputs."""

import json
import re
from typing import Any


# A confidence side-channel ("Confidence: 0.85" / "I am 85% sure" / etc.)
# is stripped from the output before comparison so that suites which
# additionally elicit a calibration probe (see ``rift.calibration``)
# remain compatible with exact-match scoring on the answer itself.
# The pattern is intentionally narrow — it only strips a *trailing*
# confidence-tag line, never one buried inside the answer.
_TRAILING_CONFIDENCE_RE = re.compile(
    r"(?im)^\s*(?:confidence\s*[:=]?\s*\d+(?:\.\d+)?\s*%?"
    r"|i(?:'m| am)\s+\d+(?:\.\d+)?\s*%?\s*(?:sure|confident|certain)\b[^\n]*"
    r"|p\s*[:=]\s*\d+(?:\.\d+)?\s*%?)\s*$"
)


def _strip_confidence(text: str) -> str:
    """Strip a trailing confidence-tag line, if present.

    Operates on the *last* line of the text. We don't run this over the
    whole body so that an output that legitimately mentions a
    probability mid-answer ("there's a 50% chance of rain") is not
    mangled. Returns the text unchanged when no confidence tag is in
    the trailing position.
    """
    if not text:
        return text
    stripped = text.rstrip()
    # Quick reject: only attempt the regex when "confidence" / "sure"
    # / a leading "p:" appears in the last ~80 chars.
    tail = stripped[-200:].lower()
    if "confidence" not in tail and "sure" not in tail \
            and "confident" not in tail and "certain" not in tail \
            and not re.search(r"^\s*p\s*[:=]", tail, re.M):
        return text
    new = _TRAILING_CONFIDENCE_RE.sub("", stripped).rstrip()
    return new if new else stripped


class ExactMatchScorer:
    """Scores 1.0 if output matches expected exactly, 0.0 otherwise.

    For dict expected values, parses JSON from output and compares field-by-field,
    returning the fraction of fields that match.

    A trailing ``Confidence: X`` line is removed from the output before
    comparison so that a suite which elicits both an answer and a
    calibration probe still scores cleanly. The full output (including
    the confidence tag) is preserved on the :class:`CaseResult`, so
    ``rift.calibration`` can still parse it.
    """

    def score(self, output: str, expected: Any) -> float:
        if isinstance(expected, dict):
            return self._score_dict(output, expected)
        clean = _strip_confidence(output)
        if isinstance(expected, str):
            return 1.0 if clean.strip() == expected.strip() else 0.0
        return 1.0 if str(clean).strip() == str(expected).strip() else 0.0

    def _score_dict(self, output: str, expected: dict) -> float:
        """Parse JSON from output and compare fields."""
        try:
            # Try to extract JSON from the output
            parsed = self._extract_json(output)
            if parsed is None:
                return 0.0

            # Field-by-field comparison
            matches = 0
            total = len(expected)
            for key, exp_val in expected.items():
                if key in parsed and str(parsed[key]).strip() == str(exp_val).strip():
                    matches += 1

            return matches / total if total > 0 else 0.0
        except Exception:
            return 0.0

    def _extract_json(self, text: str) -> dict | None:
        """Try to extract a JSON object from text."""
        text = text.strip()

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON block in markdown code fences
        if "```" in text:
            for block in text.split("```"):
                block = block.strip()
                if block.startswith("json"):
                    block = block[4:].strip()
                try:
                    return json.loads(block)
                except json.JSONDecodeError:
                    continue

        # Try to find first { ... } block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

        return None
