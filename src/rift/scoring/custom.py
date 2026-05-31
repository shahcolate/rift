"""Loader for user-supplied (``scoring: custom``) scorers.

A suite selects a custom scorer with a ``custom_scorer`` reference of the form
``target:callable``, where ``target`` is either an importable module path
(``mypkg.scorers``) or a file path (``./scorer.py``). The referenced object may
be:

* a plain function ``score(output, expected) -> float``;
* an async function ``ascore(output, expected, context=None) -> float``;
* a class implementing the :class:`~rift.scoring.Scorer` /
  :class:`~rift.scoring.AsyncScorer` protocol (instantiated with no args); or
* an already-instantiated scorer object.

Plain functions are wrapped so they satisfy the runner's scorer interface.

Security note: loading a custom scorer **executes** the target module. Only run
suites you trust. Rift never imports a custom scorer except when a suite with
``scoring: custom`` is actually run.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from pathlib import Path

from . import Scorer


def _looks_like_path(target: str) -> bool:
    return target.endswith(".py") or "/" in target or target.startswith(".")


def _load_from_file(target: str, base_dir: Path | None) -> object:
    path = Path(target).expanduser()
    if not path.is_absolute():
        # Resolve against the suite file's directory first, then the CWD.
        tried = []
        if base_dir is not None:
            tried.append((base_dir / path))
        tried.append(Path.cwd() / path)
        for cand in tried:
            if cand.exists():
                path = cand.resolve()
                break
        else:
            raise ValueError(
                f"custom_scorer file not found: '{target}' (looked in "
                f"{[str(t) for t in tried]})"
            )
    elif not path.exists():
        raise ValueError(f"custom_scorer file not found: {path}")

    mod_name = f"rift_custom_scorer_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"could not load custom_scorer module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _load_object(spec: str, base_dir: Path | None) -> object:
    if ":" not in spec:
        raise ValueError(
            f"custom_scorer must be 'module.path:callable' or "
            f"'path/to/file.py:callable', got '{spec}'"
        )
    target, _, attr = spec.rpartition(":")
    target, attr = target.strip(), attr.strip()
    if not target or not attr:
        raise ValueError(
            f"custom_scorer '{spec}' is malformed; expected 'target:callable'"
        )
    if _looks_like_path(target):
        module = _load_from_file(target, base_dir)
    else:
        try:
            module = importlib.import_module(target)
        except ImportError as e:
            raise ValueError(
                f"could not import custom_scorer module '{target}': {e}"
            ) from e
    try:
        return getattr(module, attr)
    except AttributeError:
        raise ValueError(
            f"custom_scorer '{attr}' not found in '{target}'"
        ) from None


def _as_score(spec_name: str, value) -> float:
    """Coerce a custom scorer's return to float with a clear error message."""
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"custom scorer '{spec_name}' returned a non-numeric value "
            f"({value!r}); it must return a float in [0, 1]"
        ) from None


class _FunctionScorer:
    """Adapt a plain ``score(output, expected) -> float`` function."""

    def __init__(self, fn):
        self._fn = fn
        self.__name__ = getattr(fn, "__name__", "custom")

    def score(self, output: str, expected) -> float:
        return _as_score(self.__name__, self._fn(output, expected))


class _AsyncFunctionScorer:
    """Adapt an async scorer function; forwards ``context`` when accepted.

    ``context`` is forwarded if the function declares a ``context`` parameter
    or accepts arbitrary keywords (``**kwargs``); otherwise it's omitted so a
    two-argument scorer isn't handed an unexpected keyword.
    """

    def __init__(self, fn):
        self._fn = fn
        params = inspect.signature(fn).parameters
        self._accepts_context = "context" in params or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        self.__name__ = getattr(fn, "__name__", "custom")

    async def ascore(self, output: str, expected, context: str | None = None) -> float:
        if self._accepts_context:
            return _as_score(self.__name__, await self._fn(output, expected, context=context))
        return _as_score(self.__name__, await self._fn(output, expected))


def load_custom_scorer(spec: str, base_dir: Path | None = None) -> Scorer:
    """Resolve a ``custom_scorer`` spec to a Scorer the runner can call."""
    obj = _load_object(spec, base_dir)

    if inspect.isclass(obj):
        instance = obj()
        if not (hasattr(instance, "score") or hasattr(instance, "ascore")):
            raise ValueError(
                f"custom_scorer class '{spec}' must define score() or ascore()"
            )
        return instance  # type: ignore[return-value]

    # An already-instantiated scorer object.
    if hasattr(obj, "score") or hasattr(obj, "ascore"):
        return obj  # type: ignore[return-value]

    if callable(obj):
        if inspect.iscoroutinefunction(obj):
            return _AsyncFunctionScorer(obj)  # type: ignore[return-value]
        return _FunctionScorer(obj)  # type: ignore[return-value]

    raise ValueError(
        f"custom_scorer '{spec}' must be a callable, a Scorer instance, or a "
        f"Scorer class defining score()/ascore()"
    )
