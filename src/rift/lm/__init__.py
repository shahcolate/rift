"""RiftLM: Rift's own tiny built-in language model.

A character-level GPT implemented entirely in numpy — forward pass,
backprop, and Adam by hand — trained from scratch on synthetic string
tasks (`cpy`, `rev`, `srt`, `max`) whose answers are exact-match
scoreable. Training deliberately shifts the task mix partway through
and saves a checkpoint on each side of the shift, so one `rift lm
train` produces a genuine baseline/challenger pair with a real
subgroup regression for `rift compare` to catch — no API key, no
network, no third-party ML framework.

This is not a capable model and never will be; it exists so that Rift
can demonstrate (and test) its entire drift-detection pipeline against
a model whose weights, data, and failure modes are fully controlled.
"""

from .data import TASKS, VOCAB, decode, encode, gen_eval_cases, is_eval_line
from .model import TinyGPT, TinyGPTConfig
from .train import TrainResult, train_riftlm

__all__ = [
    "TASKS",
    "VOCAB",
    "TinyGPT",
    "TinyGPTConfig",
    "TrainResult",
    "decode",
    "encode",
    "gen_eval_cases",
    "is_eval_line",
    "train_riftlm",
]
