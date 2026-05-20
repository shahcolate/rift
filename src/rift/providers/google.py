"""Google Gemini API provider.

Targets the Generative Language API (AI Studio key path) rather than
Vertex AI. The Vertex path uses GCP auth (ADC / service accounts) and
a different URL shape; if you need that, run via OpenRouter or wire
a separate provider.

The Gemini API differs from the Anthropic / OpenAI shape in three
ways the rest of Rift needs to know about:

1. The model identifier lives in the URL path (``/v1beta/models/
   {model}:generateContent``), not in the request body.
2. Messages are ``contents[].parts[].text``, not ``messages[].content``.
3. Per-call generation knobs (temperature, max_tokens, thinking level)
   live under a ``generationConfig`` object, not at the body root.

Authentication is via the ``x-goog-api-key`` header. We read
``GEMINI_API_KEY`` from the environment by default to match Google's
own examples; pass ``api_key=...`` explicitly to override.

Thinking budget
---------------
Gemini 3+ ships with reasoning ("thinking") on by default. Google
exposes this as a four-level enum — ``minimal``, ``low``,
``medium``, ``high`` — under
``generationConfig.thinkingConfig.thinkingLevel``. For paired
determinism in Rift we **pin** the default to ``medium`` (Google's
own default) so two runs of the same prompt against the same model
behave identically. Override per call by passing
``thinking_level="high"`` to ``complete()`` — it threads through
into the generationConfig.

Thinking tokens are billed as output tokens by Google, so for cost
accounting we include ``thoughtsTokenCount`` in ``output_tokens``
when the API returns it. Without this the per-correct-answer cost
would underestimate spend on reasoning models.
"""

from __future__ import annotations

import os
import time

import httpx

from . import BaseProvider, Completion


# Valid Gemini thinking levels. The API rejects anything else with a
# 400, so we surface that as an explicit error rather than letting an
# obscure server reply propagate.
_VALID_THINKING_LEVELS = {"minimal", "low", "medium", "high"}


class GoogleProvider(BaseProvider):
    """Provider for Google's Gemini API on Generative Language endpoint."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set")
        self.client = httpx.AsyncClient(
            base_url=api_base or "https://generativelanguage.googleapis.com",
            headers={
                "x-goog-api-key": self.api_key,
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )
        self.extra_params = kwargs

    async def complete(self, prompt: str, **kwargs) -> Completion:
        # Merge per-call overrides on top of constructor defaults. Per-call
        # wins so the runner's ``suite.model_params`` reach the API.
        merged = {**self.extra_params, **kwargs}

        generation_config: dict = {}
        if "temperature" in merged:
            generation_config["temperature"] = merged.pop("temperature")
        if "max_tokens" in merged:
            generation_config["maxOutputTokens"] = merged.pop("max_tokens")
        elif "maxOutputTokens" in merged:
            generation_config["maxOutputTokens"] = merged.pop("maxOutputTokens")
        else:
            generation_config["maxOutputTokens"] = 4096
        if "top_p" in merged:
            generation_config["topP"] = merged.pop("top_p")
        if "top_k" in merged:
            generation_config["topK"] = merged.pop("top_k")

        # Thinking config (Gemini 3+). Default to "medium" to match
        # Google's own default and keep paired determinism predictable.
        thinking_level = merged.pop("thinking_level", "medium")
        if thinking_level not in _VALID_THINKING_LEVELS:
            raise ValueError(
                f"thinking_level must be one of {sorted(_VALID_THINKING_LEVELS)}, "
                f"got {thinking_level!r}"
            )
        generation_config["thinkingConfig"] = {
            "thinkingLevel": thinking_level.upper(),
        }

        # Anything left in merged after pop()s is forwarded verbatim —
        # generation_config has the well-known knobs, but Gemini-specific
        # parameters like safetySettings can be passed through.
        body: dict = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        }
        for k, v in merged.items():
            body[k] = v

        url = f"/v1beta/models/{self.model}:generateContent"
        start = time.perf_counter()
        resp = await self.client.post(url, json=body)
        latency = (time.perf_counter() - start) * 1000

        resp.raise_for_status()
        data = resp.json()

        # Output text: walk parts[] of the first candidate's content,
        # collect every part whose ``text`` is non-empty and whose
        # ``thought`` flag is not set. Gemini interleaves "thought"
        # parts (when ``includeThoughts=true`` is requested) and final
        # answer parts; we only want the latter in ``output_text``.
        output = ""
        candidates = data.get("candidates", [])
        if candidates:
            for part in candidates[0].get("content", {}).get("parts", []) or []:
                if part.get("thought"):
                    continue
                text = part.get("text")
                if text:
                    output += text

        usage = data.get("usageMetadata", {})
        # Google bills thinking as output tokens; include them so the
        # cost-per-correct math reflects actual spend.
        output_tokens = int(usage.get("candidatesTokenCount", 0)) + int(
            usage.get("thoughtsTokenCount", 0)
        )

        return Completion(
            model=self.model,
            input_text=prompt,
            output_text=output,
            latency_ms=latency,
            input_tokens=int(usage.get("promptTokenCount", 0)),
            output_tokens=output_tokens,
            raw_response=data,
        )

    async def close(self) -> None:
        await self.client.aclose()
