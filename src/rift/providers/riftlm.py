"""In-process provider for RiftLM checkpoints.

``riftlm:<path>.npz`` model strings route here (see
``rift.config.resolve_model``). Inference runs in-process on numpy —
no HTTP, no API key — so every Rift command that accepts a model
accepts a RiftLM checkpoint, keylessly.

Tokens are characters (that is the model's actual tokenizer), cost is
$0 (pricing has no entry for riftlm models), and the provider
fingerprint is the sha256 digest of the checkpoint file — the same
digest ``resolve_model`` bakes into the model string so that
retraining a checkpoint in place invalidates the completion cache
instead of silently replaying the old weights' outputs.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from pathlib import Path

import click

from ..lm.data import NEWLINE_ID, decode, encode
from ..lm.model import TinyGPT
from . import BaseProvider, Completion

# Trailing content-digest suffix appended by resolve_model, e.g.
# "riftlm:models/riftlm-a.npz@3fa9c02b1d44".
_DIGEST_RE = re.compile(r"@[0-9a-f]{8,64}$")


class RiftLMCheckpointError(click.ClickException):
    """A riftlm: model string points at a missing/unreadable checkpoint.

    ClickException so the CLI prints one actionable line and exits 1 —
    the runner re-raises ClickExceptions instead of folding them into
    per-case errors, so a bad checkpoint aborts cleanly rather than
    producing an all-errored "run".
    """

    exit_code = 1

    def __init__(self, path: str, detail: str = "not found") -> None:
        super().__init__(
            f"RiftLM checkpoint '{path}' {detail}.\n"
            f"Train one with:  rift lm train"
        )


def checkpoint_path(model_str: str) -> Path:
    """Extract the checkpoint path from a ``riftlm:<path>[@digest]`` string.

    A file whose real name happens to end in an ``@<hex>`` tail is honoured:
    the raw path wins whenever it exists on disk, and the digest suffix is
    only stripped otherwise.
    """
    spec = model_str.removeprefix("riftlm:")
    raw = Path(spec)
    if raw.is_file():
        return raw
    return Path(_DIGEST_RE.sub("", spec))


def checkpoint_digest(path: Path) -> str:
    """Content digest of a checkpoint file.

    The single definition of the digest convention (algorithm + length):
    ``resolve_model`` bakes this into the model string / cache key and the
    provider reports it as the fingerprint, so the two can never drift.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


class RiftLMProvider(BaseProvider):
    """Greedy, deterministic completion from a local TinyGPT checkpoint."""

    def __init__(self, model: str, **kwargs) -> None:
        self.model = model
        path = checkpoint_path(model)
        if not path.is_file():
            raise RiftLMCheckpointError(str(path))
        try:
            self.gpt = TinyGPT.load(path)
        except Exception as exc:  # corrupt/incompatible npz
            raise RiftLMCheckpointError(str(path), f"could not be loaded ({exc})")
        # The fingerprint is the digest resolve_model already computed and
        # baked into the model string — reuse it so the fingerprint always
        # matches the cache key's identity. Hash the file only when the
        # string carries no digest (a hand-built ModelConfig).
        spec = model.removeprefix("riftlm:")
        tail = _DIGEST_RE.search(spec)
        if tail and not Path(spec).is_file():
            self._digest = tail.group(0)[1:]
        else:
            self._digest = checkpoint_digest(path)
        self.extra_params = kwargs

    async def complete(self, prompt: str, **kwargs) -> Completion:
        params = {**self.extra_params, **kwargs}
        max_new = int(params.get("max_tokens", 32))

        start = time.perf_counter()
        ids = encode(prompt)
        # Decode in a worker thread: the loop stays responsive, cases can
        # actually overlap under the runner's semaphore (numpy releases the
        # GIL inside BLAS), and the per-case asyncio.wait_for timeout can
        # fire mid-generation.
        out_ids = await asyncio.to_thread(
            self.gpt.generate, ids, max_new, NEWLINE_ID
        )
        latency = (time.perf_counter() - start) * 1000

        return Completion(
            model=self.model,
            input_text=prompt,
            output_text=decode(out_ids),
            latency_ms=latency,
            # Characters ARE this model's tokens.
            input_tokens=len(ids),
            output_tokens=len(out_ids),
            raw_response={"engine": "riftlm", "greedy": True},
            provider_fingerprint=f"riftlm-{self._digest}",
        )

    async def close(self) -> None:
        pass  # nothing to release — no network client
