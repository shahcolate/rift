"""Anthropic API provider."""

import os
import re
import time

import httpx

from . import BaseProvider, Completion, MissingAPIKeyError, raise_for_status_with_body


# Per-model parameter compatibility. Newer models deprecate knobs the
# Messages API used to accept (e.g. opus-4-7 fixes the sampler and
# rejects ``temperature``). Listing them here lets a single suite YAML
# target several model generations without per-model branches at the
# call site, while still preserving paired determinism: the dropped
# param wasn't honored by the model anyway, so the comparison is fair.
_SAMPLER_KNOBS = {"temperature", "top_p", "top_k"}
DEPRECATED_PARAMS: dict[str, set[str]] = {
    # The Mythos-class tier and the Claude 5 family reject any explicit
    # `thinking` config (thinking is always on / adaptive by default) and
    # the classic sampler knobs; we never send `thinking`, so only the
    # sampler knobs need stripping. Fable 5.1 / Mythos 5.1 also reject
    # forced `tool_choice`, which Rift never sends either.
    "claude-fable-5-1":  set(_SAMPLER_KNOBS),
    "claude-mythos-5-1": set(_SAMPLER_KNOBS),
    "claude-fable-5":    set(_SAMPLER_KNOBS),
    "claude-opus-5":     set(_SAMPLER_KNOBS),
    "claude-sonnet-5":   set(_SAMPLER_KNOBS),
    "claude-opus-4-8":   set(_SAMPLER_KNOBS),
    "claude-opus-4-7":   set(_SAMPLER_KNOBS),
}

# Models whose reasoning tokens are on by default and billed against
# ``max_tokens``: a 4096 default that fits Opus-4.7 answers can truncate
# a Fable answer after the (invisible) thinking spend. Floor, don't cap —
# an explicit larger suite value still wins. Like DEPRECATED_PARAMS,
# this is wire-level normalization: the completion cache stays keyed on
# the *requested* params, so changing this constant does not invalidate
# existing cache entries — bump the cache dir if you change a floor.
# Opus 5 is here (thinking on by default, unlike 4.7/4.8); Sonnet 5 and
# the 4.x family run without thinking unless a suite asks, so they keep
# the plain default.
MIN_MAX_TOKENS: dict[str, int] = {
    "claude-fable-5-1":  16000,
    "claude-mythos-5-1": 16000,
    "claude-fable-5":    16000,
    "claude-opus-5":     16000,
}


def _family(model: str, table: dict) -> str | None:
    """Longest ``table`` key that ``model`` is a dated variant of.

    ``claude-opus-5-20261101`` inherits ``claude-opus-5``'s entry; a
    named submodel (``-mini``, ``-fast``) does not — same rule as
    :func:`rift.pricing.lookup`.
    """
    if model in table:
        return model
    best = None
    for key in table:
        if model.startswith(key) and re.fullmatch(
                r"(-\d[\w.]*)+", model[len(key):]):
            if best is None or len(key) > len(best):
                best = key
    return best


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
        dep_key = _family(self.model, DEPRECATED_PARAMS)
        for dropped in (DEPRECATED_PARAMS[dep_key] if dep_key else ()):
            params.pop(dropped, None)
        floor_key = _family(self.model, MIN_MAX_TOKENS)
        floor = MIN_MAX_TOKENS[floor_key] if floor_key else None
        if floor is not None:
            # `or floor` guards an explicit ``max_tokens: null`` in a
            # suite's model_params, which would otherwise crash max().
            params["max_tokens"] = max(params["max_tokens"] or floor, floor)

        start = time.perf_counter()
        resp = await self.client.post("/v1/messages", json=params)
        latency = (time.perf_counter() - start) * 1000

        raise_for_status_with_body(resp)
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
            # "refusal" arrives as HTTP 200 + empty content on Fable 5/5.1
            # and Opus 5 (safety classifiers). We deliberately do NOT opt
            # into the API's server-side `fallbacks`: an eval tool must
            # MEASURE the refusal, not quietly re-route it to another
            # model — that would swap the model under the comparison.
            stop_reason=data.get("stop_reason"),
        )

    async def close(self) -> None:
        await self.client.aclose()
