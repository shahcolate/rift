"""OpenAI API provider."""

import os
import time

import httpx

from . import BaseProvider, Completion


class OpenAIProvider(BaseProvider):
    """Provider for OpenAI's Chat Completions API."""

    def __init__(self, model: str, api_key: str | None = None, api_base: str | None = None, **kwargs):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")
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
            "max_tokens": kwargs.get("max_tokens", 4096),
            "messages": [{"role": "user", "content": prompt}],
            **self.extra_params,
            **kwargs,
        }

        start = time.perf_counter()
        resp = await self.client.post("/v1/chat/completions", json=params)
        latency = (time.perf_counter() - start) * 1000

        resp.raise_for_status()
        data = resp.json()

        output = data["choices"][0]["message"]["content"] or ""

        return Completion(
            model=self.model,
            input_text=prompt,
            output_text=output,
            latency_ms=latency,
            input_tokens=data.get("usage", {}).get("prompt_tokens", 0),
            output_tokens=data.get("usage", {}).get("completion_tokens", 0),
            raw_response=data,
        )

    async def close(self) -> None:
        await self.client.aclose()
