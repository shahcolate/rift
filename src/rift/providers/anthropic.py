"""Anthropic API provider."""

import os
import time

import httpx

from . import BaseProvider, Completion, MissingAPIKeyError


# Per-model parameter compatibility. Newer models deprecate knobs the
# Messages API used to accept (e.g. opus-4-7 fixes the sampler and
# rejects ``temperature``). Listing them here lets a single suite YAML
# target several model generations without per-model branches at the
# call site, while still preserving paired determinism: the dropped
# param wasn't honored by the model anyway, so the comparison is fair.
DEPRECATED_PARAMS: dict[str, set[str]] = {
    # Fable 5 additionally rejects any explicit `thinking` config
    # (thinking is always on); we never send one, so only the sampler
    # knobs need stripping.
    "claude-fable-5":  {"temperature", "top_p", "top_k"},
    "claude-opus-4-8": {"temperature", "top_p", "top_k"},
    "claude-opus-4-7": {"temperature", "top_p", "top_k"},
}

# Models whose reasoning tokens are always on and billed against
# ``max_tokens``: a 4096 default that fits Opus answers can truncate a
# Fable answer after the (invisible) thinking spend. Floor, don't cap —
# an explicit larger suite value still wins.
MIN_MAX_TOKENS: dict[str, int] = {
    "claude-fable-5": 16000,
}


class AnthropicProvider(BaseProvider):
    """Provider for Anthropic's Messages API."""

    def __init__(self, model: str, api_key: str | None = None, **kwargs):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise MissingAPIKeyError("anthropic")
        self.client = httpx.AsyncClient(
            base_url="https://api.anthropic.com",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=120.0,
        )
        self.extra_params = kwargs

    async def complete(self, prompt: str, **kwargs) -> Completion:
        params = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "messages": [{"role": "user", "content": prompt}],
            **self.extra_params,
            **kwargs,
        }
        # Remove non-API params
        params.pop("max_tokens_override", None)
        for dropped in DEPRECATED_PARAMS.get(self.model, ()):
            params.pop(dropped, None)
        floor = MIN_MAX_TOKENS.get(self.model)
        if floor is not None:
            params["max_tokens"] = max(params["max_tokens"], floor)

        start = time.perf_counter()
        resp = await self.client.post("/v1/messages", json=params)
        latency = (time.perf_counter() - start) * 1000

        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            # raise_for_status drops the response body, which is where the
            # API explains *why* (invalid param, retention policy, ...).
            # Re-raise with the body attached so a 4xx in a saved run is
            # diagnosable after the fact.
            raise httpx.HTTPStatusError(
                f"{e}\nResponse body: {resp.text[:500]}",
                request=e.request,
                response=e.response,
            ) from None
        data = resp.json()

        output = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                output += block["text"]

        return Completion(
            model=self.model,
            input_text=prompt,
            output_text=output,
            latency_ms=latency,
            input_tokens=data.get("usage", {}).get("input_tokens", 0),
            output_tokens=data.get("usage", {}).get("output_tokens", 0),
            raw_response=data,
            # Anthropic exposes no system_fingerprint, but the response
            # echoes the *resolved* dated model id (e.g. a request for
            # "claude-opus-4-8" comes back as "claude-opus-4-8-2025..."),
            # which changes when the served snapshot behind an alias moves.
            # Best available drift signal on this API.
            provider_fingerprint=data.get("model"),
        )

    async def close(self) -> None:
        await self.client.aclose()
