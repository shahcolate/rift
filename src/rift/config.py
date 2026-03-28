"""Suite configuration parsing and validation."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, field_validator


class EvalCase(BaseModel):
    """A single evaluation case."""

    input: str
    expected: Any
    tags: list[str] = []


class SuiteConfig(BaseModel):
    """An eval suite definition."""

    name: str
    description: str = ""
    scoring: str = "exact_match"
    model_params: dict[str, Any] = {}
    cases: list[EvalCase]

    @field_validator("scoring")
    @classmethod
    def validate_scoring(cls, v: str) -> str:
        valid = {"exact_match", "semantic", "llm_judge", "custom"}
        if v not in valid:
            raise ValueError(f"scoring must be one of {valid}, got '{v}'")
        return v


class ModelConfig(BaseModel):
    """Model endpoint configuration."""

    provider: str  # anthropic, openai, local
    model: str  # model identifier string
    api_base: str | None = None
    params: dict[str, Any] = {}


BUILTIN_SUITES_DIR = Path(__file__).parent.parent.parent / "suites"


def load_suite(path_or_name: str) -> SuiteConfig:
    """Load a suite from a file path or built-in suite name."""
    path = Path(path_or_name)

    # Check if it's a built-in suite name
    if not path.exists() and not path.suffix:
        builtin = BUILTIN_SUITES_DIR / f"{path_or_name}.yaml"
        if builtin.exists():
            path = builtin
        else:
            available = [f.stem for f in BUILTIN_SUITES_DIR.glob("*.yaml")]
            raise FileNotFoundError(
                f"Suite '{path_or_name}' not found. Available built-in suites: {available}"
            )

    with open(path) as f:
        data = yaml.safe_load(f)

    return SuiteConfig(**data)


def resolve_model(model_str: str) -> ModelConfig:
    """Resolve a model string like 'claude-3-5-sonnet' to a ModelConfig."""
    # Anthropic models
    if any(model_str.startswith(p) for p in ["claude", "claude-"]):
        return ModelConfig(provider="anthropic", model=model_str)

    # OpenAI models
    if any(model_str.startswith(p) for p in ["gpt-", "o1", "o3"]):
        return ModelConfig(provider="openai", model=model_str)

    # Assume local/custom endpoint
    return ModelConfig(provider="local", model=model_str)
