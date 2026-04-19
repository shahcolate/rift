"""Eval execution engine.

The runner is the piece Rift stakes its credibility on. Every drift
claim in a report traces back to a pair of :class:`RunResult` objects
produced here, so the runner has three properties it must hold
unconditionally:

1. **Paired determinism.** Baseline and challenger must see
   byte-identical prompts, in the same order, with the same scorer.
   Randomness (distractor shuffling, bootstrap resampling) lives
   outside this module, pre-seeded.
2. **Fail-loud, retry-safe.** Transient network errors are retried
   with exponential backoff; persistent errors surface with the case
   index intact so a failed run is never silently short.
3. **Accounting is mandatory, not decorative.** Every
   :class:`CaseResult` carries input/output tokens and the USD cost
   of the completion. Dropping either breaks the cost-normalized
   drift metrics downstream.

The runner is async end-to-end with a user-configurable concurrency
cap. Case-level caching is content-addressed by
``(model, input_hash)`` so re-running a suite is free when the prompts
haven't changed — important for iterative suite development and for
the paired-comparison workflow where only one side of the pair is
new.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from .config import ModelConfig, SuiteConfig
from .pricing import cost_of
from .providers import BaseProvider, Completion
from .providers.anthropic import AnthropicProvider
from .providers.openai import OpenAIProvider
from .scoring import get_scorer


# Tunables. These are module-level (not arguments) because they are
# operational knobs, not suite-level parameters — changing them never
# changes what the eval measures, only how robustly it runs.
MAX_RETRIES = 8
BACKOFF_BASE_S = 2.0
BACKOFF_CAP_S = 90.0
PER_CASE_TIMEOUT_S = 180.0


@dataclass
class CaseResult:
    """Result of a single eval case.

    ``cost_usd`` is derived from ``input_tokens``, ``output_tokens``, and
    the model's entry in :mod:`rift.pricing`. It is stored (not
    recomputed) so historical runs remain auditable when list prices
    change.
    """

    case_index: int
    input_text: str
    expected: Any
    output: str
    score: float
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float = 0.0
    tags: list[str] = field(default_factory=list)
    error: str | None = None
    attempts: int = 1


@dataclass
class RunResult:
    """Result of running a full suite against one model."""

    model: str
    suite_name: str
    scoring_method: str
    cases: list[CaseResult]
    started_at: str = ""
    completed_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def scores(self) -> list[float]:
        return [c.score for c in self.cases]

    @property
    def mean_score(self) -> float:
        s = self.scores
        return sum(s) / len(s) if s else 0.0

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.cases)

    @property
    def total_input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.cases)

    @property
    def total_output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.cases)

    def cost_per_correct(self, correctness_threshold: float = 0.999) -> float:
        """USD spent per fully-correct case. ``inf`` if zero correct.

        The threshold is 0.999 rather than 1.0 so float-rounded
        dict-field scores (e.g. 3/3 fields = 1.0 exactly) still count
        while a 0.66 partial does not. Override it for graded rubrics.
        """
        n_correct = sum(1 for c in self.cases if c.score >= correctness_threshold)
        if n_correct == 0:
            return float("inf")
        return self.total_cost_usd / n_correct

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str | Path) -> "RunResult":
        with open(path) as f:
            data = json.load(f)
        cases = [CaseResult(**c) for c in data.pop("cases")]
        return cls(cases=cases, **data)


def _get_provider(config: ModelConfig) -> BaseProvider:
    if config.provider == "anthropic":
        return AnthropicProvider(model=config.model, **config.params)
    elif config.provider == "openai":
        return OpenAIProvider(model=config.model, **config.params)
    else:
        raise ValueError(f"Unknown provider: {config.provider}")


def _cache_key(model: str, input_text: str, model_params: dict) -> str:
    """Cache key for a completion.

    Includes ``model_params`` so changing ``temperature`` or
    ``max_tokens`` does not silently return stale completions.
    """
    payload = f"{model}:{json.dumps(model_params, sort_keys=True)}:{input_text}"
    h = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"{model}_{h}"


def _is_transient(exc: BaseException) -> bool:
    """Decide if an exception is worth retrying."""
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, asyncio.TimeoutError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        # 429 rate limit + 5xx server — retry. 4xx client errors — don't.
        code = exc.response.status_code
        return code == 429 or 500 <= code < 600
    return False


def _retry_after_s(exc: BaseException) -> float | None:
    """Extract a retry-after hint from a 429/5xx HTTP error, if present.

    Honors both the ``Retry-After`` header (seconds or HTTP-date) and
    Anthropic's ``anthropic-ratelimit-*-reset`` timestamps. Returns
    ``None`` when no authoritative hint is available so the caller
    falls back to exponential backoff.
    """
    if not isinstance(exc, httpx.HTTPStatusError):
        return None
    h = exc.response.headers
    ra = h.get("retry-after")
    if ra:
        try:
            return max(0.0, float(ra))
        except ValueError:
            # HTTP-date form — fall through; jitter-backoff is fine.
            pass
    # Anthropic: anthropic-ratelimit-tokens-reset is an ISO timestamp.
    import datetime as _dt
    for key in ("anthropic-ratelimit-input-tokens-reset",
                "anthropic-ratelimit-tokens-reset",
                "anthropic-ratelimit-requests-reset"):
        v = h.get(key)
        if not v:
            continue
        try:
            reset = _dt.datetime.fromisoformat(v.replace("Z", "+00:00"))
            now = _dt.datetime.now(_dt.timezone.utc)
            return max(0.0, (reset - now).total_seconds())
        except ValueError:
            continue
    return None


async def _complete_with_retry(
    provider: BaseProvider,
    prompt: str,
    params: dict,
) -> tuple[Completion, int]:
    """Call the provider with exponential backoff on transient failures.

    When the server sends a ``Retry-After`` (or Anthropic's
    per-window reset timestamps), we wait exactly that long rather
    than use backoff — guessing under-estimates rate-limit windows
    and wastes the retry budget.

    Returns the completion and the number of attempts used. Raises
    the last exception if all retries are exhausted or an error is
    judged non-transient.
    """
    import random as _r

    last_exc: BaseException | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            completion = await asyncio.wait_for(
                provider.complete(prompt, **params),
                timeout=PER_CASE_TIMEOUT_S,
            )
            return completion, attempt
        except BaseException as exc:
            last_exc = exc
            if not _is_transient(exc) or attempt == MAX_RETRIES:
                raise
            server_hint = _retry_after_s(exc)
            if server_hint is not None:
                # Cap server hints so a misconfigured header can't
                # stall the whole run; add small jitter to avoid
                # thundering-herd on concurrent retries.
                delay = min(server_hint, BACKOFF_CAP_S) + _r.random()
            else:
                delay = min(BACKOFF_BASE_S * (2 ** (attempt - 1)), BACKOFF_CAP_S)
                delay *= 0.8 + 0.4 * _r.random()
            await asyncio.sleep(delay)
    # Unreachable: loop either returns or raises.
    assert last_exc is not None
    raise last_exc


async def run_suite(
    suite: SuiteConfig,
    model_config: ModelConfig,
    concurrency: int = 5,
    cache_dir: str | None = None,
    enterprise_multiplier: float = 1.0,
) -> RunResult:
    """Run every case in ``suite`` against ``model_config``.

    Parameters
    ----------
    suite : SuiteConfig
        The parsed suite. ``suite.model_params`` (e.g. ``temperature``)
        are threaded through to every completion *and* into the cache
        key, so changing them invalidates the cache.
    model_config : ModelConfig
        Provider + model identifier. Provider instantiation is
        deferred until this call so ``--dry-run``-style flows can
        validate suites without API keys.
    concurrency : int
        Max simultaneous in-flight completions. Caller is responsible
        for staying within provider rate limits; the runner does not
        inspect `429` headers beyond retrying them.
    cache_dir : str | None
        Path to the completion cache. Defaults to ``.rift/cache``.
        Cache entries are JSON blobs keyed by a sha256 of
        ``(model, model_params, prompt)``.
    enterprise_multiplier : float
        Applied to list price when computing ``cost_usd``. Use e.g.
        ``0.65`` to model a 35%-discount Enterprise contract.

    Returns
    -------
    RunResult
        Cases are returned in their original suite order regardless of
        completion order. Failed cases carry ``score=0.0`` and a
        populated ``error`` field so a partial run is still analyzable.
    """
    # Provider is instantiated lazily on first cache miss so fully-cached
    # runs (including benchmark replays from recorded outcomes) work
    # without API keys configured.
    provider_holder: dict[str, BaseProvider] = {}

    def _provider() -> BaseProvider:
        if "p" not in provider_holder:
            provider_holder["p"] = _get_provider(model_config)
        return provider_holder["p"]

    scorer = get_scorer(suite.scoring)
    semaphore = asyncio.Semaphore(concurrency)

    cache_path = Path(cache_dir) if cache_dir else Path(".rift/cache")
    cache_path.mkdir(parents=True, exist_ok=True)

    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    async def run_case(idx: int, case) -> CaseResult:
        async with semaphore:
            ck = _cache_key(model_config.model, case.input, suite.model_params)
            cached = cache_path / f"{ck}.json"
            attempts = 0
            error: str | None = None

            if cached.exists():
                try:
                    with open(cached) as f:
                        completion = Completion(**json.load(f))
                except Exception:
                    # Corrupted cache entry — fall through and refetch.
                    cached.unlink(missing_ok=True)
                    completion = None  # type: ignore[assignment]
                else:
                    attempts = 0  # served from cache
            else:
                completion = None  # type: ignore[assignment]

            if completion is None:
                try:
                    completion, attempts = await _complete_with_retry(
                        _provider(), case.input, dict(suite.model_params)
                    )
                except Exception as exc:
                    return CaseResult(
                        case_index=idx,
                        input_text=case.input,
                        expected=case.expected,
                        output="",
                        score=0.0,
                        latency_ms=0.0,
                        input_tokens=0,
                        output_tokens=0,
                        cost_usd=0.0,
                        tags=list(case.tags),
                        error=f"{type(exc).__name__}: {exc}",
                        attempts=MAX_RETRIES,
                    )
                # Write cache atomically (tmp + rename) so a crashed
                # writer never leaves a half-written JSON.
                tmp = cached.with_suffix(".tmp")
                with open(tmp, "w") as f:
                    json.dump(asdict(completion), f, default=str)
                tmp.replace(cached)

            sc = scorer.score(completion.output_text, case.expected)
            cost = cost_of(
                model_config.model,
                completion.input_tokens,
                completion.output_tokens,
                enterprise_multiplier=enterprise_multiplier,
            )

            return CaseResult(
                case_index=idx,
                input_text=case.input,
                expected=case.expected,
                output=completion.output_text,
                score=sc,
                latency_ms=completion.latency_ms,
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
                cost_usd=cost,
                tags=list(case.tags),
                error=error,
                attempts=attempts,
            )

    results: list[CaseResult] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
    ) as progress:
        task = progress.add_task(
            f"Running {suite.name} on {model_config.model}",
            total=len(suite.cases),
        )
        tasks = [run_case(i, case) for i, case in enumerate(suite.cases)]
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            progress.advance(task)

    results.sort(key=lambda r: r.case_index)
    if "p" in provider_holder:
        await provider_holder["p"].close()

    completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    n_errors = sum(1 for r in results if r.error)
    metadata = {
        "concurrency": concurrency,
        "enterprise_multiplier": enterprise_multiplier,
        "n_errors": n_errors,
    }

    return RunResult(
        model=model_config.model,
        suite_name=suite.name,
        scoring_method=suite.scoring,
        cases=results,
        started_at=started_at,
        completed_at=completed_at,
        metadata=metadata,
    )
