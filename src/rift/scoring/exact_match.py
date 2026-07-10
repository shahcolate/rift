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
    r"(?i)^\s*(?:confidence\s*[:=]?\s*\d+(?:\.\d+)?\s*%?"
    r"|i(?:'m| am)\s+\d+(?:\.\d+)?\s*%?\s*(?:sure|confident|certain)\b[^\n]*"
    r"|p\s*[:=]\s*\d+(?:\.\d+)?\s*%?)\s*$"
)


def _strip_confidence(text: str) -> str:
    """Strip a trailing confidence-tag line, if present.

    Operates on the *last* line of the text only — a confidence-shaped
    line earlier in the answer ("I am 90% sure\\nParis") is part of the
    answer, not the calibration side-channel, and must not be removed.
    Returns the text unchanged when no confidence tag is in the trailing
    position.
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
    # Peel trailing confidence lines one at a time (a model may emit
    # several stacked tags); only lines in trailing position are touched.
    current = stripped
    while True:
        head, sep, last_line = current.rpartition("\n")
        if not _TRAILING_CONFIDENCE_RE.fullmatch(last_line.strip()):
            break
        if not sep:  # the whole output is the tag — keep the original
            return stripped
        current = head.rstrip()
    return current if current != stripped else text


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

        # Greedy attempt: first "{" to last "}". Handles a single object
        # (possibly nested) wrapped in prose in one parse.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

        # Greedy span failed — e.g. prose between/around objects, an
        # emoticon brace, or multiple top-level objects. Scan for the first
        # *balanced* {...} object (string/escape-aware so braces inside JSON
        # strings don't throw off the depth count) and parse that.
        for i, ch in enumerate(text):
            if ch != "{":
                continue
            candidate = _balanced_object(text, i)
            if candidate is not None:
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue

        return None


def _balanced_object(text: str, start: int) -> str | None:
    """Return the substring of ``text`` from ``start`` ("{") through its
    matching "}", or ``None`` if the braces never balance.

    Tracks JSON string context and backslash escapes so that braces inside
    string literals (``{"note": "a } brace"}``) do not corrupt the depth count.
    """
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None
