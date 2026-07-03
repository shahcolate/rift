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
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import click
import httpx
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

from .config import ModelConfig, SuiteConfig
from .pricing import cost_of
from .providers import BaseProvider, Completion
from .providers.anthropic import AnthropicProvider
from .providers.google import GoogleProvider
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
    # Server-reported model version / fingerprint for this completion (see
    # ``Completion.provider_fingerprint``). Persisted so a saved run records
    # exactly which served snapshot produced each score.
    provider_fingerprint: str | None = None
    # Per-trial scores when the case was run with replication (``trials>1``).
    # Empty for a single-trial run; ``score`` is then the lone observation.
    # When populated, ``score`` is the mean over these trials and the spread
    # captures run-to-run (generation) noise — the variance a single-trial
    # paired test cannot see. ``cost_usd``/token counts are per-trial means so
    # they still read as "the cost of one production call".
    trial_scores: list[float] = field(default_factory=list)


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

    def to_dict(self, strip_io: bool = False) -> dict:
        """Serialise the run to a dict.

        ``strip_io=True`` removes per-case ``input_text`` and ``output``
        fields — useful for sharing a results file from a proprietary suite
        without leaking the prompts or completions. Scores, costs, tokens,
        tags, and errors are preserved.
        """
        d = asdict(self)
        if strip_io:
            for case in d.get("cases", []):
                case["input_text"] = ""
                case["output"] = ""
        return d

    def save(self, path: str | Path, strip_io: bool = False) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(strip_io=strip_io), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str | Path) -> "RunResult":
        with open(path) as f:
            data = json.load(f)
        # Filter each case dict to known fields so a run saved by an older or
        # newer Rift (different CaseResult schema) still loads instead of
        # raising TypeError on a missing/extra key.
        case_fields = CaseResult.__dataclass_fields__  # type: ignore[attr-defined]
        cases = [
            CaseResult(**{k: v for k, v in c.items() if k in case_fields})
            for c in data.pop("cases")
        ]
        return cls(cases=cases, **data)


def _get_provider(config: ModelConfig) -> BaseProvider:
    if config.provider == "anthropic":
        return AnthropicProvider(model=config.model, **config.params)
    elif config.provider == "openai":
        return OpenAIProvider(model=config.model, **config.params)
    elif config.provider == "google":
        return GoogleProvider(model=config.model, **config.params)
    elif config.provider == "riftlm":
        # Local import keeps rift.lm (numpy model code) off the hot path
        # for the overwhelmingly common hosted-provider runs.
        from .providers.riftlm import RiftLMProvider

        return RiftLMProvider(model=config.model, **config.params)
    else:
        raise ValueError(f"Unknown provider: {config.provider}")


def _cache_key(model: str, input_text: str, model_params: dict,
               trial: int = 0) -> str:
    """Cache key for a completion.

    Includes ``model_params`` so changing ``temperature`` or
    ``max_tokens`` does not silently return stale completions. ``trial``
    distinguishes replicates: trial 0 keeps the legacy suffix-free key so
    existing caches still hit, while trials 1..k-1 get their own entries so
    replication actually re-samples the model instead of replaying one cached
    completion k times.
    """
    payload = f"{model}:{json.dumps(model_params, sort_keys=True)}:{input_text}"
    h = hashlib.sha256(payload.encode()).hexdigest()[:16]
    suffix = "" if trial == 0 else f"_t{trial}"
    # The key doubles as a cache *filename*. Hosted model ids are already
    # safe, but riftlm checkpoint paths ("riftlm:models/a.npz@<digest>")
    # carry slashes/colons; replace anything unsafe and bound the length so
    # a deep checkpoint path can't push the filename past the filesystem's
    # 255-byte limit (uniqueness comes from the hash, the prefix is only
    # for human-browsable cache dirs). The hash is computed from the raw
    # string first, and every shipped hosted-model id is already within
    # [A-Za-z0-9._-], so their keys are byte-identical to what older Rifts
    # wrote.
    safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", model)[-120:]
    return f"{safe_model}_{h}{suffix}"


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
                # Stamp the attempt count on the exception so the caller can
                # record it accurately in the audit trail (a non-transient
                # 4xx that fails on attempt 1 must not report MAX_RETRIES).
                try:
                    exc.rift_attempts = attempt  # type: ignore[attr-defined]
                except Exception:
                    pass  # some C-level exceptions reject attribute writes
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
    show_progress: bool = True,
    trials: int = 1,
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
    show_progress : bool
        Render a per-case Rich progress bar (default). Set ``False`` to
        run silently — used by the demo, which owns the screen with its
        own ``console.status`` spinner and cannot have a second live
        display active at the same time.
    trials : int
        Replicates per case (default 1). With ``trials>1`` each case is
        sent ``trials`` times with distinct cache keys, so re-sampling
        actually re-queries the model rather than replaying one cached
        completion. The resulting ``CaseResult.score`` is the mean over
        trials and ``trial_scores`` holds the per-trial values — the raw
        material for the run-to-run noise floor a single-trial paired test
        cannot estimate. Cost/token fields are per-trial means. Note the
        downstream consequence: trial-mean scores are continuous, so
        ``compare_runs`` selects the paired t-test rather than McNemar, and
        ``cost_per_correct`` (threshold 0.999) counts only all-trials-correct
        cases. A case that succeeds on some trials but fails others still
        produces a valid mean score and is NOT counted as an error.

    Returns
    -------
    RunResult
        Cases are returned in their original suite order regardless of
        completion order. Failed cases carry ``score=0.0`` and a
        populated ``error`` field so a partial run is still analyzable.
    """
    # A riftlm config built by hand (bypassing resolve_model) carries no
    # weight digest in its model string, which would key the cache on the
    # path alone — an in-place retrain would then replay the old weights'
    # completions. Normalize here so every entry point gets the digest.
    if model_config.provider == "riftlm":
        from .providers.riftlm import _DIGEST_RE

        if not _DIGEST_RE.search(model_config.model):
            from .config import _resolve_riftlm

            model_config = _resolve_riftlm(model_config.model)

    # Provider is instantiated lazily on first cache miss so fully-cached
    # runs (including benchmark replays from recorded outcomes) work
    # without API keys configured.
    provider_holder: dict[str, BaseProvider] = {}

    def _provider() -> BaseProvider:
        if "p" not in provider_holder:
            provider_holder["p"] = _get_provider(model_config)
        return provider_holder["p"]

    cache_path = Path(cache_dir) if cache_dir else Path(".rift/cache")
    cache_path.mkdir(parents=True, exist_ok=True)

    # For llm_judge scoring, plumb the judge model + cache dir into
    # the scorer at construction time. Other scorers ignore kwargs.
    scorer_kwargs: dict = {}
    if suite.scoring == "llm_judge":
        scorer_kwargs = {
            "judge_model": suite.judge_model,
            "cache_dir": str(cache_path),
            "prompt_template": suite.prompts.get("judge_rubric"),
        }
    elif suite.scoring == "semantic":
        scorer_kwargs = {
            "embedding_model": suite.embedding_model,
            "cache_dir": str(cache_path),
        }
    elif suite.scoring == "custom":
        scorer_kwargs = {
            "custom_scorer": suite.custom_scorer,
            "base_dir": suite._source_dir,
        }
    scorer = get_scorer(suite.scoring, **scorer_kwargs)
    is_async_scorer = hasattr(scorer, "ascore")
    semaphore = asyncio.Semaphore(concurrency)

    # Every distinct server fingerprint seen this run, across all cases AND all
    # trials. Accumulated here (not re-derived from per-case results) so a
    # within-case rollout under --trials — where a single case's trials straddle
    # a server-side switch — is still counted; the per-case CaseResult keeps only
    # its first fingerprint for the report. Updates happen synchronously inside
    # run_case between awaits, so the shared set is safe under asyncio.
    all_fingerprints: set[str] = set()

    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    async def _fetch_trial(case, trial: int) -> tuple[Completion, int]:
        """Return ``(completion, attempts)`` for ``case``'s ``trial``, caching it.

        ``attempts`` is 0 when served from cache, else the real retry count from
        :func:`_complete_with_retry` — preserving the attempt-count audit trail.
        Raises on a missing key (fatal, user-fixable) or an exhausted retry —
        the caller decides how a failed trial folds into the aggregate.
        """
        ck = _cache_key(model_config.model, case.input, suite.model_params, trial)
        cached = cache_path / f"{ck}.json"
        if cached.exists():
            try:
                with open(cached) as f:
                    return Completion.from_cache(json.load(f)), 0  # cache hit
            except Exception:
                cached.unlink(missing_ok=True)  # corrupted — refetch
        completion, attempts = await _complete_with_retry(
            _provider(), case.input, dict(suite.model_params)
        )
        # Write cache atomically (tmp + rename) so a crashed writer never
        # leaves a half-written JSON.
        tmp = cached.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(asdict(completion), f, default=str)
        tmp.replace(cached)
        return completion, attempts

    async def _score_completion(case, completion: Completion) -> float:
        if is_async_scorer:
            return await scorer.ascore(  # type: ignore[attr-defined]
                completion.output_text, case.expected, context=case.input,
            )
        return scorer.score(completion.output_text, case.expected)

    async def run_case(idx: int, case) -> CaseResult:
        async with semaphore:
            trial_scores: list[float] = []
            costs: list[float] = []
            in_toks: list[int] = []
            out_toks: list[int] = []
            latencies: list[float] = []
            fingerprints: list[str] = []
            first_output = ""
            first_error: str | None = None
            failed_attempts = 0  # attempts of the first failing trial, if any
            success_attempts = 0  # max attempts among successful trials

            for trial in range(max(1, trials)):
                try:
                    completion, attempts = await _fetch_trial(case, trial)
                except click.ClickException:
                    # Fatal + user-fixable (missing API key, unreadable
                    # RiftLM checkpoint, ...) — surface as the clean
                    # ClickException rather than burying it as score 0.0
                    # on every case and computing drift over garbage.
                    raise
                except Exception as exc:
                    if first_error is None:
                        first_error = f"{type(exc).__name__}: {exc}"
                        failed_attempts = getattr(exc, "rift_attempts", MAX_RETRIES)
                    continue  # a failed trial drops out of the aggregate
                success_attempts = max(success_attempts, attempts)
                sc = await _score_completion(case, completion)
                trial_scores.append(sc)
                costs.append(cost_of(
                    model_config.model, completion.input_tokens,
                    completion.output_tokens,
                    enterprise_multiplier=enterprise_multiplier,
                ))
                in_toks.append(completion.input_tokens)
                out_toks.append(completion.output_tokens)
                latencies.append(completion.latency_ms)
                if completion.provider_fingerprint:
                    fingerprints.append(completion.provider_fingerprint)
                if not first_output:
                    first_output = completion.output_text

            if not trial_scores:
                # Every trial failed — this case genuinely errored.
                return CaseResult(
                    case_index=idx, input_text=case.input, expected=case.expected,
                    output="", score=0.0, latency_ms=0.0, input_tokens=0,
                    output_tokens=0, cost_usd=0.0, tags=list(case.tags),
                    error=first_error, attempts=failed_attempts,
                )

            all_fingerprints.update(fingerprints)
            mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
            return CaseResult(
                case_index=idx,
                input_text=case.input,
                expected=case.expected,
                output=first_output,
                score=mean(trial_scores),
                latency_ms=mean(latencies),
                input_tokens=round(mean(in_toks)),
                output_tokens=round(mean(out_toks)),
                cost_usd=mean(costs),
                tags=list(case.tags),
                # At least one trial succeeded, so this case produced a real
                # score — it must NOT be counted as an error even if an earlier
                # trial failed (that would inflate n_errors and flag a scored
                # case). attempts is the real retry count of the successful work.
                error=None,
                attempts=success_attempts,
                # First distinct fingerprint is stored per-case; the run-level
                # set above sees every trial's so a mid-run rollout still shows.
                provider_fingerprint=fingerprints[0] if fingerprints else None,
                trial_scores=trial_scores if trials > 1 else [],
            )

    results: list[CaseResult] = []
    tasks = [run_case(i, case) for i, case in enumerate(suite.cases)]
    if show_progress:
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
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                progress.advance(task)
    else:
        for coro in asyncio.as_completed(tasks):
            results.append(await coro)

    results.sort(key=lambda r: r.case_index)
    if "p" in provider_holder:
        await provider_holder["p"].close()
    if is_async_scorer and hasattr(scorer, "close"):
        await scorer.close()  # type: ignore[attr-defined]

    completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    n_errors = sum(1 for r in results if r.error)
    metadata: dict[str, Any] = {
        "concurrency": concurrency,
        "enterprise_multiplier": enterprise_multiplier,
        "n_errors": n_errors,
    }
    if trials > 1:
        metadata["trials"] = trials
    # Stamp the distinct server-reported fingerprints seen this run. A drift
    # detector that caches on the request alone is blind to a silent
    # server-side weight swap behind a stable alias; recording the fingerprint
    # closes that hole. More than one distinct value means the served snapshot
    # changed *during* the run (a rollout) — surfaced loudly because it makes
    # the run internally non-comparable.
    fingerprints = sorted(all_fingerprints)
    if fingerprints:
        metadata["fingerprints"] = fingerprints
        if len(fingerprints) > 1:
            metadata["fingerprint_rollout"] = True
            if show_progress:
                from rich.panel import Panel
                from rich.console import Console as _Console
                from rich.markup import escape as _escape
                _Console().print(Panel(
                    f"  Model [cyan]{_escape(model_config.model)}[/cyan] returned "
                    f"{len(fingerprints)} distinct server fingerprints during "
                    "this run:\n"
                    # Fingerprints are provider-supplied — escape Rich markup.
                    + "\n".join(f"    • {_escape(fp)}" for fp in fingerprints)
                    + "\n\n  The served snapshot changed mid-run (a rollout). "
                    "Scores\n  from before and after the switch are not strictly "
                    "comparable;\n  re-run once the rollout settles.",
                    title="[bold yellow]⚠ Model fingerprint rollout[/bold yellow]",
                    border_style="yellow",
                ))
    # Stamp the judge / embedding model into metadata so a saved
    # RunResult carries who graded it. Methodology, not decoration.
    if is_async_scorer and hasattr(scorer, "judge_model"):
        metadata["judge_model"] = scorer.judge_model  # type: ignore[attr-defined]
    if is_async_scorer and hasattr(scorer, "embedding_model"):
        metadata["embedding_model"] = scorer.embedding_model  # type: ignore[attr-defined]
    # Disclose a custom scorer so a saved RunResult records how it was scored.
    if suite.scoring == "custom" and suite.custom_scorer:
        metadata["custom_scorer"] = suite.custom_scorer
    # Disclose any custom probe prompts so a published drift report can't
    # hide a non-default rubric. Methodology, not decoration.
    if suite.prompts or suite.cues:
        from .prompts import overridden_keys
        metadata["custom_prompts"] = overridden_keys(suite.prompts, suite.cues)

    return RunResult(
        model=model_config.model,
        suite_name=suite.name,
        scoring_method=suite.scoring,
        cases=results,
        started_at=started_at,
        completed_at=completed_at,
        metadata=metadata,
    )
