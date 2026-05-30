"""Tests for embedding-based semantic scoring (rift.scoring.semantic)."""

from __future__ import annotations

import asyncio

import pytest

from rift.scoring import get_scorer
from rift.scoring.semantic import (
    DEFAULT_EMBEDDING_MODEL,
    GoogleEmbedder,
    OpenAIEmbedder,
    SemanticScorer,
    _cosine,
    resolve_embedder,
)


class _StubEmbedder:
    """Deterministic offline embedder. Maps known phrases to fixed vectors;
    unknown text gets a hashed-but-stable vector. Counts calls."""

    VECTORS = {
        "the cat sat on the mat": [1.0, 0.0, 0.0],
        "a feline rested on the rug": [0.9, 0.1, 0.0],   # ~same meaning
        "stock prices fell sharply": [0.0, 1.0, 0.0],    # unrelated
        "opposite": [-1.0, 0.0, 0.0],                    # negative cosine
    }

    def __init__(self, model="stub"):
        self.model = model
        self.calls = 0

    async def embed(self, text: str) -> list[float]:
        self.calls += 1
        if text in self.VECTORS:
            return list(self.VECTORS[text])
        # Stable fallback so repeat calls on the same text are identical.
        h = sum(ord(c) for c in text)
        return [float(h % 7), float(h % 5), float(h % 3)]

    async def close(self):
        pass


class TestCosine:
    def test_identical_is_one(self):
        assert _cosine([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)

    def test_orthogonal_is_zero(self):
        assert _cosine([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_opposite_is_negative_one(self):
        assert _cosine([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_empty_or_mismatched_is_zero(self):
        assert _cosine([], [1]) == 0.0
        assert _cosine([1, 2], [1]) == 0.0

    def test_zero_norm_is_zero(self):
        assert _cosine([0, 0], [1, 1]) == 0.0


def _scorer(tmp_path, **kw):
    stub = _StubEmbedder()
    s = SemanticScorer(embedding_model="stub",
                       embedder_factory=lambda m: stub,
                       cache_dir=str(tmp_path), **kw)
    return s, stub


class TestAscore:
    def test_equivalent_meaning_scores_high(self, tmp_path):
        s, _ = _scorer(tmp_path)
        score = asyncio.run(s.ascore("the cat sat on the mat",
                                     "a feline rested on the rug"))
        assert score > 0.95

    def test_unrelated_scores_low(self, tmp_path):
        s, _ = _scorer(tmp_path)
        score = asyncio.run(s.ascore("the cat sat on the mat",
                                     "stock prices fell sharply"))
        assert score < 0.1

    def test_negative_cosine_clamped_to_zero(self, tmp_path):
        s, _ = _scorer(tmp_path)
        score = asyncio.run(s.ascore("opposite", "the cat sat on the mat"))
        assert score == 0.0

    def test_empty_output_scores_zero_without_calling_embedder(self, tmp_path):
        s, stub = _scorer(tmp_path)
        assert asyncio.run(s.ascore("   ", "anything")) == 0.0
        assert stub.calls == 0

    def test_empty_expected_scores_zero_without_calling_embedder(self, tmp_path):
        s, stub = _scorer(tmp_path)
        assert asyncio.run(s.ascore("anything", "")) == 0.0
        assert asyncio.run(s.ascore("anything", None)) == 0.0
        assert stub.calls == 0

    def test_records_similarity_audit(self, tmp_path):
        s, _ = _scorer(tmp_path)
        asyncio.run(s.ascore("the cat sat on the mat", "a feline rested on the rug"))
        assert s.last_similarity  # populated for the report path


class TestThreshold:
    def test_threshold_makes_binary(self, tmp_path):
        s, _ = _scorer(tmp_path, threshold=0.85)
        high = asyncio.run(s.ascore("the cat sat on the mat",
                                    "a feline rested on the rug"))
        low = asyncio.run(s.ascore("the cat sat on the mat",
                                   "stock prices fell sharply"))
        assert high == 1.0 and low == 0.0


class TestEmbeddingCache:
    def test_repeat_text_not_re_embedded(self, tmp_path):
        s, stub = _scorer(tmp_path)
        asyncio.run(s.ascore("the cat sat on the mat", "a feline rested on the rug"))
        calls_after_first = stub.calls  # 2 (output + expected)
        # Same pair again -> both served from cache, no new embed calls.
        asyncio.run(s.ascore("the cat sat on the mat", "a feline rested on the rug"))
        assert stub.calls == calls_after_first == 2

    def test_expected_vector_reused_across_outputs(self, tmp_path):
        s, stub = _scorer(tmp_path)
        # Two different outputs vs the SAME expected: expected embedded once.
        asyncio.run(s.ascore("the cat sat on the mat", "stock prices fell sharply"))
        asyncio.run(s.ascore("opposite", "stock prices fell sharply"))
        # output1, expected, output2 = 3 embeds (expected not re-embedded).
        assert stub.calls == 3

    def test_new_scorer_reads_persisted_cache(self, tmp_path):
        s1, stub1 = _scorer(tmp_path)
        asyncio.run(s1.ascore("the cat sat on the mat", "a feline rested on the rug"))
        # A fresh scorer over the same cache dir reuses persisted vectors.
        stub2 = _StubEmbedder()
        s2 = SemanticScorer(embedding_model="stub",
                            embedder_factory=lambda m: stub2,
                            cache_dir=str(tmp_path))
        asyncio.run(s2.ascore("the cat sat on the mat", "a feline rested on the rug"))
        assert stub2.calls == 0  # everything from disk


class TestSyncEntryPoint:
    def test_score_works_outside_loop(self, tmp_path):
        s, _ = _scorer(tmp_path)
        score = s.score("the cat sat on the mat", "a feline rested on the rug")
        assert score > 0.95

    def test_score_raises_inside_loop(self, tmp_path):
        s, _ = _scorer(tmp_path)

        async def inner():
            with pytest.raises(RuntimeError, match="running event loop"):
                s.score("a", "b")

        asyncio.run(inner())


class TestResolveEmbedder:
    def test_routing_by_id(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "y")
        assert isinstance(resolve_embedder("text-embedding-3-small"), OpenAIEmbedder)
        assert isinstance(resolve_embedder("text-embedding-3-large"), OpenAIEmbedder)
        assert isinstance(resolve_embedder("text-embedding-004"), GoogleEmbedder)
        assert isinstance(resolve_embedder("gemini-embedding-001"), GoogleEmbedder)

    def test_unknown_id_raises(self):
        with pytest.raises(ValueError, match="Unknown embedding model"):
            resolve_embedder("totally-made-up-model")


class TestRegistration:
    def test_get_scorer_returns_semantic(self):
        s = get_scorer("semantic")
        assert isinstance(s, SemanticScorer)
        assert s.embedding_model == DEFAULT_EMBEDDING_MODEL

    def test_validate_scoring_accepts_semantic(self):
        from rift.config import SuiteConfig
        cfg = SuiteConfig(name="x", scoring="semantic",
                          cases=[{"input": "a", "expected": "b"}])
        assert cfg.scoring == "semantic"

    def test_embedding_model_field_carried(self):
        from rift.config import SuiteConfig
        cfg = SuiteConfig(name="x", scoring="semantic",
                          embedding_model="text-embedding-3-large",
                          cases=[{"input": "a", "expected": "b"}])
        assert cfg.embedding_model == "text-embedding-3-large"
