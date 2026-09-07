"""OpenAI API provider."""

import os
import time

import httpx

from . import BaseProvider, Completion, MissingAPIKeyError, raise_for_status_with_body


class OpenAIProvider(BaseProvider):
    """Provider for OpenAI's Chat Completions API."""

    def __init__(self, model: str, api_key: str | None = None, api_base: str | None = None, **kwargs):
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
        self.extra_params = kwargs

    async def complete(self, prompt: str, **kwargs) -> Completion:
        params = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            **self.extra_params,
            **kwargs,
        }
        max_tokens = params.pop("max_tokens", 4096)

        # gpt-5 / o-series use `max_completion_tokens` and reject any
        # non-default `temperature`. Older chat models still take
        # `max_tokens` and arbitrary temperature.
        if self.model.startswith(("gpt-5", "o1", "o3", "o4")):
            params["max_completion_tokens"] = max_tokens
            params.pop("temperature", None)
        else:
            params["max_tokens"] = max_tokens

        start = time.perf_counter()
        resp = await self.client.post("/v1/chat/completions", json=params)
        latency = (time.perf_counter() - start) * 1000

        raise_for_status_with_body(resp)
        data = resp.json()

        message = data["choices"][0]["message"]
        # Structured refusals arrive as {"content": null, "refusal": "..."}.
        # Keep the refusal text as the output — an empty string would make
        # the refusal-drift classifier count a declined answer as "not
        # refused", and the reason would be lost from saved runs.
        output = message.get("content") or message.get("refusal") or ""

        return Completion(
            model=self.model,
            input_text=prompt,
            output_text=output,
            latency_ms=latency,
            input_tokens=data.get("usage", {}).get("prompt_tokens", 0),
            output_tokens=data.get("usage", {}).get("completion_tokens", 0),
            raw_response=data,
            # OpenAI's system_fingerprint is the canonical backend-version
            # signal; fall back to the resolved dated model id when the
            # endpoint (e.g. o-series) omits the fingerprint.
            provider_fingerprint=data.get("system_fingerprint") or data.get("model"),
            stop_reason=data["choices"][0].get("finish_reason"),
        )

    async def close(self) -> None:
        await self.client.aclose()
