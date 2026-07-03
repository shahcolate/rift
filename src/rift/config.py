"""Suite configuration parsing and validation."""

from pathlib import Path
from typing import Any

import click
import yaml
from pydantic import BaseModel, PrivateAttr, ValidationError, field_validator, model_validator


class SuiteNotFoundError(click.ClickException):
    """A suite path or built-in name could not be found.

    Subclasses ``click.ClickException`` so an unknown suite produces a clean
    CLI message and exit 1 rather than a raw ``FileNotFoundError`` traceback.
    """

    exit_code = 1

    def __init__(self, name: str, available: list[str] | None) -> None:
        self.name = name
        self.available = available
        msg = f"Suite '{name}' not found."
        if available:
            msg += f" Available built-in suites: {available}"
        super().__init__(msg)


class SuiteValidationError(click.ClickException):
    """A suite YAML failed validation.

    Subclasses ``click.ClickException`` so a malformed suite produces a short,
    actionable CLI message and exit 1 — never a raw pydantic traceback — no
    matter which command loaded it. Carries the underlying pydantic error for
    callers that want the detail.
    """

    exit_code = 1

    def __init__(self, path: "Path", error: Exception) -> None:
        self.path = path
        self.original = error
        # pydantic's str() lists each failing field with its message; trim the
        # noisy URL footer it appends.
        detail = str(error).split("For further information", 1)[0].strip()
        super().__init__(f"Invalid suite '{path}':\n{detail}")


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
    # Optional judge-model override for llm_judge scoring. When set,
    # this locks the judge to a specific model in the suite file so
    # the suite is reproducible without env-var coordination. When
    # unset, the runner falls back to $RIFT_JUDGE_MODEL or the
    # built-in default. Ignored for non-llm_judge scoring.
    judge_model: str | None = None
    # Optional embedding-model override for semantic scoring. Locks the
    # embedding model in the suite file for reproducibility, mirroring
    # judge_model. When unset, the runner falls back to
    # $RIFT_EMBEDDING_MODEL or the built-in default. Ignored for
    # non-semantic scoring.
    embedding_model: str | None = None
    # Optional per-suite overrides of Rift's built-in probe prompts. `prompts`
    # maps a registry key (e.g. "judge_rubric") to a full template string;
    # `cues` maps a faithfulness cue name to a hint template (overriding an
    # existing cue or adding a new one). Both are validated against
    # rift.prompts at load time. Empty = use the built-in defaults.
    prompts: dict[str, str] = {}
    cues: dict[str, str] | None = None
    # Required when scoring == "custom": a reference to a user-supplied scorer,
    # either "package.module:callable" (importable) or "path/to/file.py:callable"
    # (a file, resolved relative to the suite file's directory then the CWD).
    # The target may be a function score(output, expected) -> float, an async
    # function ascore(output, expected, context=None) -> float, or a class /
    # instance implementing the Scorer protocol. Loading executes the target
    # module, so only run suites you trust. Ignored for non-custom scoring.
    custom_scorer: str | None = None
    cases: list[EvalCase]

    # Directory of the suite file (set by load_suite), used to resolve a
    # relative custom_scorer file path. Not part of the serialized model.
    _source_dir: Path | None = PrivateAttr(default=None)

    @field_validator("scoring")
    @classmethod
    def validate_scoring(cls, v: str) -> str:
        valid = {"exact_match", "fuzzy_match", "exec_tests", "llm_judge",
                 "semantic", "custom"}
        if v not in valid:
            raise ValueError(f"scoring must be one of {valid}, got '{v}'")
        return v

    @model_validator(mode="after")
    def _validate_prompt_overrides(self) -> "SuiteConfig":
        # Lazy import to avoid a circular import (rift.prompts pulls default
        # templates from the scoring / faithfulness modules).
        from .prompts import validate_overrides

        validate_overrides(self.prompts, self.cues)
        return self

    @model_validator(mode="after")
    def _validate_custom_scorer(self) -> "SuiteConfig":
        if self.scoring == "custom":
            if not self.custom_scorer:
                raise ValueError(
                    "scoring 'custom' requires a 'custom_scorer' field, e.g. "
                    "'mypkg.scorers:score' or './scorer.py:score'"
                )
            if ":" not in self.custom_scorer:
                raise ValueError(
                    f"custom_scorer must be 'target:callable', got "
                    f"'{self.custom_scorer}'"
                )
        elif self.custom_scorer is not None:
            raise ValueError(
                "custom_scorer is only valid when scoring: custom"
            )
        return self


class ModelConfig(BaseModel):
    """Model endpoint configuration."""

    provider: str  # anthropic, openai, local
    model: str  # model identifier string
    api_base: str | None = None
    params: dict[str, Any] = {}


# Built-in suites are bundled into the wheel under rift/_data/suites (see
# pyproject force-include). When running from a source checkout or an editable
# install that directory doesn't exist, so fall back to the repo-root suites/.
_BUNDLED_SUITES_DIR = Path(__file__).parent / "_data" / "suites"
_REPO_SUITES_DIR = Path(__file__).parent.parent.parent / "suites"
BUILTIN_SUITES_DIR = (
    _BUNDLED_SUITES_DIR if _BUNDLED_SUITES_DIR.is_dir() else _REPO_SUITES_DIR
)


def load_suite(path_or_name: str) -> SuiteConfig:
    """Load a suite from a file path or built-in suite name."""
    path = Path(path_or_name)

    # Check if it's a built-in suite name
    if not path.exists() and not path.suffix:
        builtin = BUILTIN_SUITES_DIR / f"{path_or_name}.yaml"
        if builtin.exists():
            path = builtin
        else:
            available = sorted(f.stem for f in BUILTIN_SUITES_DIR.glob("*.yaml"))
            raise SuiteNotFoundError(path_or_name, available)
    elif not path.exists():
        # An explicit path (carries a suffix) that isn't on disk.
        raise SuiteNotFoundError(path_or_name, None)

    with open(path) as f:
        data = yaml.safe_load(f)

    try:
        cfg = SuiteConfig(**data)
    except ValidationError as e:
        # Surface a malformed suite as a clean CLI message + exit 1 instead of a
        # raw pydantic traceback. Programmatic callers can catch
        # SuiteValidationError (or inspect .original) for the detail.
        raise SuiteValidationError(path, e) from None
    # Record the suite file's directory so a relative custom_scorer file path
    # resolves against it (not just the CWD).
    cfg._source_dir = path.resolve().parent
    return cfg


# Short aliases for convenience on the command line. Kept tiny on
# purpose — we want `rift compare --baseline opus-4-6 --challenger
# opus-4-7 ...` to just work without making users memorize dated
# variants, but we don't want a sprawling nickname registry.
MODEL_ALIASES: dict[str, str] = {
    "fable-5":    "claude-fable-5",
    "fable":      "claude-fable-5",
    "opus-4-8":   "claude-opus-4-8",
    "opus-4-7":   "claude-opus-4-7",
    "opus-4-6":   "claude-opus-4-6",
    "opus-4":     "claude-opus-4-20250514",
    "sonnet-4-6": "claude-sonnet-4-6",
    "sonnet-4":   "claude-sonnet-4-20250514",
    "haiku-4-5":  "claude-haiku-4-5-20251001",
    "sonnet-3-5": "claude-3-5-sonnet-20241022",
    # Google Gemini — keep both punctuations so users can type the
    # "3.5" they see in marketing or the "3-5" that's filename-safe.
    "gemini-3-5-flash": "gemini-3.5-flash",
    "gemini-3.5-flash": "gemini-3.5-flash",
    "gemini-flash":     "gemini-3.5-flash",
    # OpenAI — keep "5.5" and "5-5" both routable for the same reason.
    "gpt-5-5":          "gpt-5.5",
    "gpt-5.5":          "gpt-5.5",
}


def resolve_model(model_str: str) -> ModelConfig:
    """Resolve a model string like 'opus-4-7' to a :class:`ModelConfig`.

    Accepts short aliases (see :data:`MODEL_ALIASES`), canonical dated
    identifiers, or any unrecognized string which is treated as a
    local/custom endpoint so that self-hosted models work without
    configuration.
    """
    model_str = MODEL_ALIASES.get(model_str, model_str)

    if model_str.startswith("riftlm:"):
        return _resolve_riftlm(model_str)

    if model_str.startswith("claude"):
        return ModelConfig(provider="anthropic", model=model_str)

    if model_str.startswith(("gpt-", "o1", "o3", "o4")):
        return ModelConfig(provider="openai", model=model_str)

    if model_str.startswith("gemini"):
        return ModelConfig(provider="google", model=model_str)

    return ModelConfig(provider="local", model=model_str)


def _resolve_riftlm(model_str: str) -> ModelConfig:
    """Resolve ``riftlm:<checkpoint.npz>`` to a keyless in-process model.

    The checkpoint file's content digest is appended to the model string
    (``riftlm:models/riftlm-a.npz@3fa9c02b1d44``). The runner's completion
    cache is keyed on the model string, so retraining a checkpoint *in
    place* — same path, new weights — invalidates the cache instead of
    silently replaying the previous weights' completions. The digest also
    lands in reports and run metadata as weight-level provenance, the same
    role ``provider_fingerprint`` plays for hosted models.

    Missing checkpoints fail here, at resolve time, with a clean message —
    not per-case inside the runner.
    """
    # Local import: rift.providers pulls in click/httpx machinery that
    # config-only callers (e.g. suite validation) shouldn't need eagerly.
    from .providers.riftlm import (
        RiftLMCheckpointError,
        checkpoint_digest,
        checkpoint_path,
    )

    path = checkpoint_path(model_str)
    if not path.is_file():
        raise RiftLMCheckpointError(str(path))
    return ModelConfig(
        provider="riftlm", model=f"riftlm:{path}@{checkpoint_digest(path)}"
    )
