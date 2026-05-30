"""Abstract base provider interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import click


# provider name -> (env var that holds its key, where to get a key).
# Single source of truth for both the missing-key error and `rift setup`.
PROVIDER_KEYS: dict[str, tuple[str, str]] = {
    "anthropic": ("ANTHROPIC_API_KEY", "https://console.anthropic.com/settings/keys"),
    "openai": ("OPENAI_API_KEY", "https://platform.openai.com/api-keys"),
    "google": ("GEMINI_API_KEY", "https://aistudio.google.com/apikey"),
}


class MissingAPIKeyError(click.ClickException):
    """A provider needs an API key that isn't configured.

    Subclasses ``click.ClickException`` so that no matter how deep in the
    stack it surfaces — including lazily, on the first live API call — the
    CLI prints a short, actionable message and exits 1 instead of dumping a
    Python traceback at a non-engineer. The interactive ``rift setup`` /
    on-demand prompt paths normally handle keys before we ever reach here.
    """

    exit_code = 1

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
    """A single model completion result."""

    model: str
    input_text: str
    output_text: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    raw_response: dict


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
