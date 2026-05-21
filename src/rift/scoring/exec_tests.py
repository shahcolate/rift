"""Exec-based scoring for code-generation suites.

Runs the model-generated function in an isolated subprocess against a
list of canonical ``(args, expected_return)`` test cases. The case
score is the fraction of tests that return a value equal to
``expected`` (so 5 tests with 3 passing → 0.6). Compilation errors,
missing function definitions, runtime exceptions, and timeouts all
yield 0.0 for the offending tests.

Threat model: this scorer exec's untrusted LLM output. It uses
subprocess isolation + a wall-clock timeout, which is enough to
contain infinite loops and accidental ``import os; rm -rf`` style
mistakes from naive prompts. It is NOT a sandbox — a determined
adversarial model could still touch the filesystem or network from
the spawned interpreter. Run code_generation suites only against
models you would otherwise trust to write code in your dev env.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


# Strip a leading ```python / ``` fence if the model added one despite
# the prompt asking for "no markdown". We only pull the *first* fenced
# block so chatty preambles after the code (rare with temperature=0
# but possible) don't pollute the namespace.
_FENCE_RE = re.compile(r"```(?:python)?\s*\n?(.*?)```", re.DOTALL)


def _extract_code(output: str) -> str:
    """Return the Python source from a model's raw output."""
    m = _FENCE_RE.search(output)
    if m:
        return m.group(1).strip()
    return output.strip()


_DRIVER = r"""
import json, sys
code_path, spec_path = sys.argv[1], sys.argv[2]
with open(code_path) as _f:
    _code = _f.read()
with open(spec_path) as _f:
    _spec = json.load(_f)
_ns = {}
try:
    exec(compile(_code, code_path, 'exec'), _ns)
except BaseException as _e:
    print(json.dumps({'error': f'compile/import: {type(_e).__name__}: {_e}',
                      'results': [False] * len(_spec['tests'])}))
    sys.exit(0)
_fn = _ns.get(_spec['function'])
if not callable(_fn):
    print(json.dumps({'error': f"function {_spec['function']!r} not defined",
                      'results': [False] * len(_spec['tests'])}))
    sys.exit(0)
_results = []
for _case in _spec['tests']:
    try:
        _actual = _fn(*_case['args'])
        _results.append(_actual == _case['expected'])
    except BaseException:
        _results.append(False)
print(json.dumps({'results': _results}))
"""


class ExecTestsScorer:
    """Score code-generation outputs by executing test cases.

    The suite's ``expected`` field must be a dict of the form::

        function: <name>
        tests:
          - args: [<positional args>]
            expected: <return value>
          ...

    The score is the fraction of test cases whose return value is
    equal (``==``) to ``expected``. Returns 0.0 on any non-dict
    expected, to fail loudly on misconfigured suites rather than
    silently scoring 1.0.
    """

    def __init__(self, timeout_s: float = 5.0) -> None:
        self.timeout_s = timeout_s

    def score(self, output: str, expected: Any) -> float:
        if not isinstance(expected, dict) or "function" not in expected \
                or "tests" not in expected:
            return 0.0
        tests = expected["tests"]
        if not tests:
            return 0.0

        code = _extract_code(output)
        with tempfile.TemporaryDirectory() as tmp:
            code_path = Path(tmp) / "model.py"
            spec_path = Path(tmp) / "spec.json"
            code_path.write_text(code)
            spec_path.write_text(json.dumps({
                "function": expected["function"],
                "tests": tests,
            }))

            try:
                proc = subprocess.run(
                    [sys.executable, "-I", "-c", _DRIVER,
                     str(code_path), str(spec_path)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                )
            except subprocess.TimeoutExpired:
                return 0.0

            if proc.returncode != 0:
                return 0.0
            try:
                payload = json.loads(proc.stdout.strip().splitlines()[-1])
            except (json.JSONDecodeError, IndexError):
                return 0.0

            results = payload.get("results", [])
            if not results:
                return 0.0
            return sum(1 for r in results if r) / len(results)
