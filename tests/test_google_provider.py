"""Tests for the Google Gemini provider.

Uses ``httpx.MockTransport`` to intercept the HTTP layer so no real
network call is exercised. Coverage:

* request shape — URL path includes the model id; auth header set;
  body has ``contents/parts/text`` shape, generationConfig with
  temperature / maxOutputTokens / thinkingConfig.
* response parsing — text extracted from candidates[0].content.parts;
  ``thought=true`` parts excluded; usageMetadata populated correctly
  including the thinking-token roll-up.
* config knob plumbing — ``temperature``, ``max_tokens``, and
  ``thinking_level`` thread through to the right places.
* error handling — invalid thinking_level rejected at the client.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from rift.providers import MissingAPIKeyError
from rift.providers.google import GoogleProvider


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _Recorded:
    """Records the requests the provider sends through the mock transport."""

    def __init__(self) -> None:
        self.url: str | None = None
        self.method: str | None = None
        self.headers: dict | None = None
        self.body: dict | None = None


def _provider_with_mock(responder, recorder: _Recorded | None = None) -> GoogleProvider:
    """Build a GoogleProvider whose httpx client is bound to a MockTransport."""
    rec = recorder or _Recorded()

    def handler(request: httpx.Request) -> httpx.Response:
        rec.url = str(request.url)
        rec.method = request.method
        rec.headers = dict(request.headers)
        try:
            rec.body = json.loads(request.content.decode())
        except json.JSONDecodeError:
            rec.body = None
        return responder(request)

    provider = GoogleProvider(model="gemini-3.5-flash", api_key="test-key")
    # Swap the httpx client for one bound to the mock transport. The
    # base_url and headers must match production so URL-routing logic
    # in the provider still gets exercised.
    provider.client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://generativelanguage.googleapis.com",
        headers={
            "x-goog-api-key": "test-key",
            "Content-Type": "application/json",
        },
    )
    return provider


def _ok_response(text: str = "Paris.", input_tokens: int = 5,
                 output_tokens: int = 2, thoughts_tokens: int = 0) -> httpx.Response:
    """Build a well-formed Gemini generateContent response."""
    return httpx.Response(
        200,
        json={
            "candidates": [
                {"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}
            ],
            "usageMetadata": {
                "promptTokenCount": input_tokens,
                "candidatesTokenCount": output_tokens,
                "thoughtsTokenCount": thoughts_tokens,
                "totalTokenCount": input_tokens + output_tokens + thoughts_tokens,
            },
        },
    )


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------


class TestRequestShape:
    def test_url_includes_model_path_and_method(self):
        rec = _Recorded()
        provider = _provider_with_mock(lambda _r: _ok_response(), rec)
        asyncio.run(provider.complete("What is the capital of France?"))
        # generateContent goes to /v1beta/models/{model}:generateContent
        assert "/v1beta/models/gemini-3.5-flash:generateContent" in rec.url
        assert rec.method == "POST"
        asyncio.run(provider.close())

    def test_api_key_header_set(self):
        rec = _Recorded()
        provider = _provider_with_mock(lambda _r: _ok_response(), rec)
        asyncio.run(provider.complete("hi"))
        assert rec.headers["x-goog-api-key"] == "test-key"
        asyncio.run(provider.close())

    def test_body_uses_contents_parts_text(self):
        rec = _Recorded()
        provider = _provider_with_mock(lambda _r: _ok_response(), rec)
        asyncio.run(provider.complete("hello there"))
        assert rec.body["contents"] == [{"parts": [{"text": "hello there"}]}]
        asyncio.run(provider.close())

    def test_temperature_routes_to_generation_config(self):
        rec = _Recorded()
        provider = _provider_with_mock(lambda _r: _ok_response(), rec)
        asyncio.run(provider.complete("x", temperature=0.0))
        assert rec.body["generationConfig"]["temperature"] == 0.0
        asyncio.run(provider.close())

    def test_max_tokens_aliased_to_max_output_tokens(self):
        # Suites in this codebase use ``max_tokens`` (Anthropic/OpenAI
        # convention); the Google provider must translate it.
        rec = _Recorded()
        provider = _provider_with_mock(lambda _r: _ok_response(), rec)
        asyncio.run(provider.complete("x", max_tokens=128))
        assert rec.body["generationConfig"]["maxOutputTokens"] == 128
        asyncio.run(provider.close())

    def test_top_p_top_k_routed_correctly(self):
        rec = _Recorded()
        provider = _provider_with_mock(lambda _r: _ok_response(), rec)
        asyncio.run(provider.complete("x", top_p=0.9, top_k=40))
        assert rec.body["generationConfig"]["topP"] == 0.9
        assert rec.body["generationConfig"]["topK"] == 40
        asyncio.run(provider.close())

    def test_thinking_level_defaults_to_medium(self):
        rec = _Recorded()
        provider = _provider_with_mock(lambda _r: _ok_response(), rec)
        asyncio.run(provider.complete("x"))
        thinking = rec.body["generationConfig"]["thinkingConfig"]
        assert thinking["thinkingLevel"] == "MEDIUM"
        asyncio.run(provider.close())

    def test_thinking_level_override(self):
        rec = _Recorded()
        provider = _provider_with_mock(lambda _r: _ok_response(), rec)
        asyncio.run(provider.complete("x", thinking_level="high"))
        assert rec.body["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "HIGH"
        asyncio.run(provider.close())

    def test_invalid_thinking_level_rejected_client_side(self):
        provider = _provider_with_mock(lambda _r: _ok_response())
        with pytest.raises(ValueError, match="thinking_level"):
            asyncio.run(provider.complete("x", thinking_level="ultra"))
        asyncio.run(provider.close())

    def test_unknown_kwargs_passed_through(self):
        # Gemini-specific knobs like safetySettings should reach the body.
        rec = _Recorded()
        provider = _provider_with_mock(lambda _r: _ok_response(), rec)
        asyncio.run(provider.complete("x", safetySettings=[{"category": "X"}]))
        assert rec.body["safetySettings"] == [{"category": "X"}]
        asyncio.run(provider.close())


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class TestResponseParsing:
    def test_basic_text_extracted(self):
        provider = _provider_with_mock(lambda _r: _ok_response(text="42."))
        completion = asyncio.run(provider.complete("the answer?"))
        assert completion.output_text == "42."
        assert completion.input_tokens == 5
        assert completion.output_tokens == 2
        asyncio.run(provider.close())

    def test_multipart_text_concatenated(self):
        body = httpx.Response(200, json={
            "candidates": [{"content": {"parts": [
                {"text": "Hello, "},
                {"text": "world."},
            ]}}],
            "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 4},
        })
        provider = _provider_with_mock(lambda _r: body)
        c = asyncio.run(provider.complete("hi"))
        assert c.output_text == "Hello, world."
        asyncio.run(provider.close())

    def test_thought_parts_excluded_from_output(self):
        # When ``includeThoughts=true`` is requested, Gemini interleaves
        # thought parts with answer parts. We want only the answer.
        body = httpx.Response(200, json={
            "candidates": [{"content": {"parts": [
                {"text": "Step 1: think...", "thought": True},
                {"text": "Step 2: also thinking...", "thought": True},
                {"text": "The answer is 4."},
            ]}}],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "thoughtsTokenCount": 50,
            },
        })
        provider = _provider_with_mock(lambda _r: body)
        c = asyncio.run(provider.complete("2+2?"))
        assert c.output_text == "The answer is 4."
        # Thought tokens roll up into output_tokens for cost accounting.
        assert c.output_tokens == 5 + 50
        asyncio.run(provider.close())

    def test_empty_candidates_handled(self):
        # A safety refusal can come back with candidates=[]. We must
        # not crash; output is "" and tokens are still recorded.
        body = httpx.Response(200, json={
            "candidates": [],
            "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 0},
        })
        provider = _provider_with_mock(lambda _r: body)
        c = asyncio.run(provider.complete("x"))
        assert c.output_text == ""
        assert c.input_tokens == 7
        asyncio.run(provider.close())

    def test_missing_usage_metadata_defaults_to_zero(self):
        # If the API ever omits usageMetadata we should report 0 rather
        # than crash. The cost math will then be 0 — caller's problem.
        body = httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
        })
        provider = _provider_with_mock(lambda _r: body)
        c = asyncio.run(provider.complete("x"))
        assert c.input_tokens == 0
        assert c.output_tokens == 0
        asyncio.run(provider.close())

    def test_http_error_raised(self):
        body = httpx.Response(429, json={"error": {"message": "rate limited"}})
        provider = _provider_with_mock(lambda _r: body)
        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(provider.complete("x"))
        asyncio.run(provider.close())

    def test_completion_carries_model_and_raw_response(self):
        provider = _provider_with_mock(lambda _r: _ok_response())
        c = asyncio.run(provider.complete("hi"))
        assert c.model == "gemini-3.5-flash"
        assert "candidates" in c.raw_response
        asyncio.run(provider.close())


# ---------------------------------------------------------------------------
# Config / env integration
# ---------------------------------------------------------------------------


class TestEnvIntegration:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(MissingAPIKeyError, match="GEMINI_API_KEY"):
            GoogleProvider(model="gemini-3.5-flash")

    def test_env_var_picked_up(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "from-env")
        provider = GoogleProvider(model="gemini-3.5-flash")
        assert provider.api_key == "from-env"
        asyncio.run(provider.close())

    def test_explicit_api_key_beats_env(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "from-env")
        provider = GoogleProvider(model="gemini-3.5-flash", api_key="explicit")
        assert provider.api_key == "explicit"
        asyncio.run(provider.close())


# ---------------------------------------------------------------------------
# resolve_model + pricing integration
# ---------------------------------------------------------------------------


class TestModelResolution:
    def test_resolve_gemini_routes_to_google_provider(self):
        from rift.config import resolve_model
        cfg = resolve_model("gemini-3.5-flash")
        assert cfg.provider == "google"
        assert cfg.model == "gemini-3.5-flash"

    def test_resolve_alias_normalizes_punctuation(self):
        from rift.config import resolve_model
        # The "3-5" punctuation alias maps to the canonical "3.5".
        cfg = resolve_model("gemini-3-5-flash")
        assert cfg.model == "gemini-3.5-flash"

    def test_resolve_short_alias(self):
        from rift.config import resolve_model
        cfg = resolve_model("gemini-flash")
        assert cfg.model == "gemini-3.5-flash"

    def test_pricing_lookup_for_gemini_35_flash(self):
        from rift.pricing import lookup
        p = lookup("gemini-3.5-flash")
        assert p is not None
        assert p.input_per_mtok == 1.50
        assert p.output_per_mtok == 9.00

    def test_pricing_family_prefix_for_dated_gemini(self):
        # A dated snapshot like gemini-3.5-flash-05-2026 should inherit
        # the family's published price via the prefix-match fallback.
        from rift.pricing import lookup
        p = lookup("gemini-3.5-flash-05-2026")
        assert p is not None
        assert p.input_per_mtok == 1.50
