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

    ClickException so the CLI prints one actionable line and exits 1.
    """

    exit_code = 1

    def __init__(self, path: str, detail: str = "not found") -> None:
        super().__init__(
            f"RiftLM checkpoint '{path}' {detail}.\n"
            f"Train one with:  rift lm train"
        )


def checkpoint_path(model_str: str) -> Path:
    """Extract the checkpoint path from a ``riftlm:<path>[@digest]`` string."""
    spec = model_str.removeprefix("riftlm:")
    return Path(_DIGEST_RE.sub("", spec))


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
        import hashlib

        self._digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        self.extra_params = kwargs

    async def complete(self, prompt: str, **kwargs) -> Completion:
        params = {**self.extra_params, **kwargs}
        max_new = int(params.get("max_tokens", 32))

        start = time.perf_counter()
        ids = encode(prompt)
        # Inference is pure CPU numpy; yield once so the runner's event loop
        # stays responsive under concurrency.
        await asyncio.sleep(0)
        out_ids = self.gpt.generate(
            ids, max_new_tokens=max_new, stop_id=NEWLINE_ID
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
