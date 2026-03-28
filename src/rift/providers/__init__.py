"""Abstract base provider interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


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
