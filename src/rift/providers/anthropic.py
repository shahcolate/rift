"""Anthropic API provider."""

import os
import time

import httpx

from . import BaseProvider, Completion


class AnthropicProvider(BaseProvider):
    """Provider for Anthropic's Messages API."""

    def __init__(self, model: str, api_key: str | None = None, **kwargs):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
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

        start = time.perf_counter()
        resp = await self.client.post("/v1/messages", json=params)
        latency = (time.perf_counter() - start) * 1000

        resp.raise_for_status()
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
        )

    async def close(self) -> None:
        await self.client.aclose()
