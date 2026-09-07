"""Abstract base provider interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import click
import httpx

from .._errors import OperationalError


def raise_for_status_with_body(resp: httpx.Response) -> None:
    """``raise_for_status``, but keep the response body in the message.

    ``httpx.Response.raise_for_status`` drops the body, which is where
    every provider explains *why* a 4xx happened (invalid param,
    exhausted credit, retention policy, ...). A 4xx recorded in a saved
    run is undiagnosable without it. The re-raised error keeps the same
    type, request, and response, so retry/transient classification in
    the runner is unaffected.
    """
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise httpx.HTTPStatusError(
            f"{e}\nResponse body: {resp.text[:2000]}",
            request=e.request,
            response=e.response,
        ) from None


# provider name -> (env var that holds its key, where to get a key).
# Single source of truth for both the missing-key error and `rift setup`.
PROVIDER_KEYS: dict[str, tuple[str, str]] = {
    "anthropic": ("ANTHROPIC_API_KEY", "https://console.anthropic.com/settings/keys"),
    "openai": ("OPENAI_API_KEY", "https://platform.openai.com/api-keys"),
    "google": ("GEMINI_API_KEY", "https://aistudio.google.com/apikey"),
}


class MissingAPIKeyError(OperationalError):
    """A provider needs an API key that isn't configured.

    Subclasses ``click.ClickException`` so that no matter how deep in the
    stack it surfaces — including lazily, on the first live API call — the
    CLI prints a short, actionable message and exits 2 (operational error,
    distinct from the gate's exit 1) instead of dumping a Python traceback
    at a non-engineer. The interactive ``rift setup`` / on-demand prompt
    paths normally handle keys before we ever reach here.
    """

    exit_code = 2

    def __init__(self, provider: str) -> None:
        self.provider = provider
        self.env_var, self.signup_url = PROVIDER_KEYS.get(
            provider, (f"{provider.upper()}_API_KEY", "")
        )
        super().__init__(f"{self.env_var} is not set.")

    def show(self, file=None) -> None:  # noqa: ARG002 - Click passes a file arg
        lines = [
            "",
            f"Rift needs your {self.env_var} to call the {self.provider} API.",
            "",
            "  Add it in 10 seconds:  rift setup",
            f"  Or set it yourself:    export {self.env_var}=...",
        ]
        if self.signup_url:
            lines.append(f"  Get a key:             {self.signup_url}")
        lines.append("")
        click.echo("\n".join(lines), err=True)


@dataclass
class Completion:
    """A single model completion result.

    ``provider_fingerprint`` is the server-reported model-version /
    fingerprint string, when the API exposes one (OpenAI's
    ``system_fingerprint``, Gemini's ``modelVersion``, the resolved
    dated ``model`` Anthropic/OpenAI echo back). It is the only reliable
    signal that the weights behind a *stable* model alias changed
    server-side — the silent drift a cache keyed on the request alone
    would otherwise mask. ``None`` when the provider exposes nothing
    usable.

    ``stop_reason`` is the provider's normalized end-of-generation reason
    (Anthropic ``stop_reason``, OpenAI ``finish_reason``, Gemini
    ``finishReason``), kept verbatim. The value that matters for drift
    honesty is ``"refusal"``: Fable 5 / 5.1 and Opus 5 answer a
    safety-classifier decline with HTTP 200, an EMPTY content list, and
    ``stop_reason="refusal"`` — a scorer sees "" and marks the case wrong,
    so an over-refusal regression would publish as a *capability*
    regression unless the reason travels with the output. ``None`` when
    the provider reports nothing (RiftLM, older cache blobs).
    """

    model: str
    input_text: str
    output_text: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    raw_response: dict
    provider_fingerprint: str | None = None
    stop_reason: str | None = None

    @classmethod
    def from_cache(cls, data: dict) -> "Completion":
        """Build a Completion from a cached JSON dict, tolerating schema drift.

        Cache blobs written by older Rift versions lack newer fields, and a
        future version may add fields this one doesn't know. Filtering to the
        current field set keeps both directions loadable instead of raising
        ``TypeError`` on an unexpected key.
        """
        fields = cls.__dataclass_fields__  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in fields})


class BaseProvider(ABC):
    """Abstract LLM provider."""

    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> Completion:
        """Send a prompt and return a completion."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""
        ...
