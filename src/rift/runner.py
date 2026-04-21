"""Eval execution engine."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from .config import SuiteConfig, ModelConfig
from .providers import BaseProvider, Completion
from .providers.anthropic import AnthropicProvider
from .providers.openai import OpenAIProvider
from .scoring import get_scorer


@dataclass
class CaseResult:
    """Result of a single eval case."""

    case_index: int
    input_text: str
    expected: Any
    output: str
    score: float
    latency_ms: float
    input_tokens: int
    output_tokens: int


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
    """Instantiate a provider from config."""
    if config.provider == "anthropic":
        return AnthropicProvider(model=config.model, **config.params)
    elif config.provider == "openai":
        return OpenAIProvider(model=config.model, **config.params)
    else:
        raise ValueError(f"Unknown provider: {config.provider}")


def _cache_key(model: str, input_text: str, params: dict[str, Any]) -> str:
    """Generate a cache key for a completion.

    Includes sampling params so that changing temperature, max_tokens, etc.
    in a suite invalidates stale completions instead of silently serving them.
    """
    params_blob = json.dumps(params or {}, sort_keys=True, default=str)
    payload = f"{model}:{input_text}:{params_blob}"
    h = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"{model}_{h}"


async def run_suite(
    suite: SuiteConfig,
    model_config: ModelConfig,
    concurrency: int = 5,
    cache_dir: str | None = None,
) -> RunResult:
    """Run all eval cases in a suite against a model."""
    provider = _get_provider(model_config)
    scorer = get_scorer(suite.scoring)
    semaphore = asyncio.Semaphore(concurrency)

    cache_path = Path(cache_dir) if cache_dir else Path(".rift/cache")
    cache_path.mkdir(parents=True, exist_ok=True)

    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    async def run_case(idx: int, case) -> CaseResult:
        async with semaphore:
            # Check cache
            ck = _cache_key(model_config.model, case.input, suite.model_params)
            cached = cache_path / f"{ck}.json"
            if cached.exists():
                with open(cached) as f:
                    cached_data = json.load(f)
                completion = Completion(**cached_data)
            else:
                completion = await provider.complete(case.input, **suite.model_params)
                # Save to cache
                with open(cached, "w") as f:
                    json.dump(asdict(completion), f, default=str)

            sc = scorer.score(completion.output_text, case.expected)

            return CaseResult(
                case_index=idx,
                input_text=case.input,
                expected=case.expected,
                output=completion.output_text,
                score=sc,
                latency_ms=completion.latency_ms,
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
            )

    # Run with progress bar
    results: list[CaseResult] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
    ) as progress:
        task = progress.add_task(
            f"Running {suite.name} on {model_config.model}", total=len(suite.cases)
        )

        tasks = [run_case(i, case) for i, case in enumerate(suite.cases)]
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            progress.advance(task)

    # Sort by case index
    results.sort(key=lambda r: r.case_index)

    await provider.close()

    completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    return RunResult(
        model=model_config.model,
        suite_name=suite.name,
        scoring_method=suite.scoring,
        cases=results,
        started_at=started_at,
        completed_at=completed_at,
    )
