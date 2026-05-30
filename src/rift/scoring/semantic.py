"""Embedding-based semantic scoring.

``exact_match`` and ``fuzzy_match`` compare surface forms; an LLM judge is
powerful but expensive and biased (length, family, self-preference). For the
middle ground — "is this answer about the same thing as the reference?" —
embedding cosine similarity is cheap, deterministic, and bias-light. This is
the meaning-level comparison the :mod:`rift.scoring.fuzzy_match` docstring
points at.

The scorer embeds the model output and the reference answer, then scores by
cosine similarity clamped to ``[0, 1]``:

    score = max(0.0, cosine(embed(output), embed(expected)))

Negatives are clamped to 0 rather than rescaled: modern sentence embeddings put
unrelated text near 0, not -1, so ``max(0, cos)`` keeps the graded signal where
it is informative without inflating unrelated pairs to ~0.5 (which ``(1+cos)/2``
would do).

Like the LLM judge, three properties keep this defensible in a report:

1. **The embedding model is named** — it is part of the methodology. The scorer
   carries ``embedding_model``; the runner stamps it into the run metadata.
2. **Embeddings are cached** by ``(embedding_model, text)`` so re-running a
   comparison is free, and the reference answer is reused across every case and
   across the paired baseline/challenger runs. Concurrent embeds of the same
   text (cases sharing a reference, run in parallel) are coalesced into a single
   API call via an in-flight future map, so a shared reference is embedded once
   even on a cold cache.
3. **The mapping is fixed** — ``max(0, cosine)``, no per-run tuning — so two runs
   over identical outputs produce identical scores.

Backends mirror the completion providers: OpenAI (``text-embedding-3-*``) and
Google (``text-embedding-004`` / ``gemini-embedding-*``), selected by the
embedding-model id. httpx only — no new dependencies.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..providers import MissingAPIKeyError

# Default embedding model. Resolved at scorer-construction time (not import
# time) so test code can mutate the env var.
DEFAULT_EMBEDDING_MODEL_ENV = "RIFT_EMBEDDING_MODEL"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors.

    Returns 0.0 if either vector is empty or has zero norm (undefined cosine);
    callers treat that as "no signal", which is the safe score.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Embedding backends
# ---------------------------------------------------------------------------


@dataclass
class EmbeddingResult:
    """A single text embedding plus the model that produced it."""

    model: str
    text: str
    vector: list[float]


class Embedder(ABC):
    """Abstract embedding backend. One HTTP call per ``embed``."""

    model: str

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Return the embedding vector for ``text``."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""
        ...


class OpenAIEmbedder(Embedder):
    """OpenAI embeddings (``POST /v1/embeddings``)."""

    def __init__(self, model: str, api_key: str | None = None,
                 api_base: str | None = None, **kwargs: Any) -> None:
        import httpx

        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise MissingAPIKeyError("openai")
        self.client = httpx.AsyncClient(
            base_url=api_base or "https://api.openai.com",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )

    async def embed(self, text: str) -> list[float]:
        resp = await self.client.post(
            "/v1/embeddings", json={"model": self.model, "input": text}
        )
        resp.raise_for_status()
        data = resp.json()
        return list(data["data"][0]["embedding"])

    async def close(self) -> None:
        await self.client.aclose()


class GoogleEmbedder(Embedder):
    """Google Gemini embeddings (``POST /v1beta/models/{m}:embedContent``)."""

    def __init__(self, model: str, api_key: str | None = None,
                 api_base: str | None = None, **kwargs: Any) -> None:
        import httpx

        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise MissingAPIKeyError("google")
        self.client = httpx.AsyncClient(
            base_url=api_base or "https://generativelanguage.googleapis.com",
            headers={
                "x-goog-api-key": self.api_key,
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )

    async def embed(self, text: str) -> list[float]:
        # The model id lives in the URL path; the body carries the content.
        url = f"/v1beta/models/{self.model}:embedContent"
        body = {
            "model": f"models/{self.model}",
            "content": {"parts": [{"text": text}]},
        }
        resp = await self.client.post(url, json=body)
        resp.raise_for_status()
        data = resp.json()
        return list(data["embedding"]["values"])

    async def close(self) -> None:
        await self.client.aclose()


EmbedderFactory = Callable[[str], Embedder]


def resolve_embedder(model_id: str) -> Embedder:
    """Build an :class:`Embedder` for an embedding-model identifier.

    Routing mirrors ``rift.config.resolve_model``: OpenAI ``text-embedding-*``
    / ``*ada*`` ids go to OpenAI; Google ``text-embedding-004`` /
    ``gemini-embedding-*`` ids go to Google. An unrecognized id is an explicit
    error rather than a silent wrong-provider call.
    """
    m = model_id.lower()
    if m.startswith("text-embedding-3") or "ada" in m or m.startswith("text-embedding-ada"):
        return OpenAIEmbedder(model_id)
    if m.startswith("gemini-embedding") or m.startswith("text-embedding-00") \
            or m.startswith("embedding-00") or m.startswith("models/text-embedding"):
        return GoogleEmbedder(model_id)
    raise ValueError(
        f"Unknown embedding model '{model_id}'. Use an OpenAI "
        f"(text-embedding-3-small/large) or Google (text-embedding-004, "
        f"gemini-embedding-001) model, or set RIFT_EMBEDDING_MODEL."
    )


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class SemanticScorer:
    """Score by embedding cosine similarity, clamped to ``[0, 1]``.

    Async by design (embedding is an HTTP call). The synchronous :meth:`score`
    is only valid outside a running event loop; the runner calls :meth:`ascore`
    directly. Mirrors :class:`rift.scoring.llm_judge.LLMJudgeScorer`.

    Parameters
    ----------
    embedding_model
        Embedding model id. Defaults to ``$RIFT_EMBEDDING_MODEL`` or
        :data:`DEFAULT_EMBEDDING_MODEL`.
    embedder_factory
        ``model_id -> Embedder``. Defaults to :func:`resolve_embedder`; tests
        inject a stub.
    cache_dir
        Where to persist embeddings. Defaults to ``$RIFT_CACHE_DIR`` or
        ``.rift/cache``. Keys include the model so swapping models never
        collides.
    threshold
        If set, the cosine is thresholded to a binary 1.0/0.0 score (so the
        binary McNemar drift test applies). When ``None`` (default), the graded
        ``max(0, cosine)`` score is returned. Must be in ``[0, 1]``.
    """

    def __init__(
        self,
        embedding_model: str | None = None,
        embedder_factory: EmbedderFactory | None = None,
        cache_dir: str | Path | None = None,
        threshold: float | None = None,
    ) -> None:
        self.embedding_model = (
            embedding_model
            or os.environ.get(DEFAULT_EMBEDDING_MODEL_ENV)
            or DEFAULT_EMBEDDING_MODEL
        )
        self._embedder_factory = embedder_factory or resolve_embedder
        self._embedder: Embedder | None = None
        if cache_dir is None:
            cache_dir = os.environ.get("RIFT_CACHE_DIR") or ".rift/cache"
        self.cache_dir = Path(cache_dir)
        if threshold is not None and not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"threshold must be in [0, 1], got {threshold!r}. The cosine "
                f"score is clamped to [0, 1], so a threshold outside it is "
                f"either always- or never-passing."
            )
        self.threshold = threshold
        # In-flight embedding coalescing: under the concurrent runner, many
        # cases share a reference answer and would all miss the not-yet-written
        # cache and embed the same text in parallel. A per-text future shared
        # across awaiters collapses those into one call (and one charge).
        self._inflight: dict[str, asyncio.Future[list[float]]] = {}
        # Per-case audit log (cosine before clamp/threshold), keyed by the
        # output text hash; useful for surfacing similarities in a report.
        self.last_similarity: dict[str, float] = {}

    # ----- sync entry point (Scorer protocol compatibility) -----

    def score(self, output: str, expected: Any) -> float:
        """Sync entry point. Only valid outside a running event loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            raise RuntimeError(
                "SemanticScorer.score() called inside a running event loop; "
                "use `await scorer.ascore(...)` instead."
            )
        return asyncio.run(self.ascore(output, expected))

    # ----- async entry point (real implementation) -----

    async def ascore(
        self,
        output: str,
        expected: Any,
        context: str | None = None,  # noqa: ARG002 - unused; protocol parity
    ) -> float:
        """Score ``output`` against ``expected`` by embedding cosine.

        Empty output or empty expected scores 0.0 without an API call (an empty
        answer has no meaning to match).
        """
        expected_str = "" if expected is None else str(expected)
        if not output.strip() or not expected_str.strip():
            return 0.0

        out_vec = await self._embed_cached(output)
        exp_vec = await self._embed_cached(expected_str)
        cos = _cosine(out_vec, exp_vec)
        # Key the audit log by (output, expected): two cases with the same
        # output but different references must not clobber each other.
        self.last_similarity[self._pair_hash(output, expected_str)] = cos

        if self.threshold is not None:
            return 1.0 if cos >= self.threshold else 0.0
        return max(0.0, cos)

    async def close(self) -> None:
        if self._embedder is not None:
            await self._embedder.close()
            self._embedder = None

    # ----- internals -----

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = self._embedder_factory(self.embedding_model)
        return self._embedder

    async def _embed_cached(self, text: str) -> list[float]:
        key = self._cache_key(text)
        cached = self._read_cache(key)
        if cached is not None:
            return cached
        # Coalesce concurrent embeds of the same text: the first caller owns
        # the API call; later callers (running cases in parallel) await the
        # same future instead of issuing duplicate, billable calls.
        inflight = self._inflight.get(key)
        if inflight is not None:
            return await inflight
        loop = asyncio.get_event_loop()
        fut: asyncio.Future[list[float]] = loop.create_future()
        self._inflight[key] = fut
        try:
            vector = await self._get_embedder().embed(text)
        except Exception as exc:
            fut.set_exception(exc)
            raise
        else:
            self._write_cache(key, vector)
            fut.set_result(vector)
            return vector
        finally:
            self._inflight.pop(key, None)

    def _text_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _pair_hash(self, output: str, expected: str) -> str:
        return hashlib.sha256(
            f"{output}\x00{expected}".encode()
        ).hexdigest()[:16]

    def _cache_key(self, text: str) -> str:
        # embed_ prefix so embedding entries never collide with completion
        # entries or judge entries in the same cache dir.
        return f"embed_{self.embedding_model}_{self._text_hash(text)}"

    def _cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    def _read_cache(self, cache_key: str) -> list[float] | None:
        path = self._cache_path(cache_key)
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            return list(data["vector"])
        except Exception:
            # Treat corruption as a miss; the next write overwrites it.
            path.unlink(missing_ok=True)
            return None

    def _write_cache(self, cache_key: str, vector: list[float]) -> None:
        path = self._cache_path(cache_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump({"model": self.embedding_model, "vector": vector}, f)
        tmp.replace(path)
