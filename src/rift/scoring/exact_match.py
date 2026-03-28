"""Exact match scoring for structured outputs."""

import json
from typing import Any


class ExactMatchScorer:
    """Scores 1.0 if output matches expected exactly, 0.0 otherwise.

    For dict expected values, parses JSON from output and compares field-by-field,
    returning the fraction of fields that match.
    """

    def score(self, output: str, expected: Any) -> float:
        if isinstance(expected, dict):
            return self._score_dict(output, expected)
        if isinstance(expected, str):
            return 1.0 if output.strip() == expected.strip() else 0.0
        return 1.0 if str(output).strip() == str(expected).strip() else 0.0

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
