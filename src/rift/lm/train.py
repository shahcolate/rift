"""RiftLM training loop.

One ``rift lm train`` run manufactures a *model upgrade*, not just a
model: partway through training (``switch``, default 60%) the task mix
shifts — ``rev`` is dropped entirely and its probability mass moves to
the other tasks — and a checkpoint is saved on each side of the shift:

- ``riftlm-a.npz``  end of phase 1, trained on all four tasks
- ``riftlm-b.npz``  final weights after continued phase-2 training

Checkpoint B is the "newer, more trained" model and typically improves
on ``cpy``/``srt``/``max`` — while catastrophically forgetting ``rev``.
That is exactly the shape of regression Rift exists to catch, and
``rift compare --baseline riftlm:models/riftlm-a.npz --challenger
riftlm:models/riftlm-b.npz --suite riftlm --subgroup task:`` catches it
with a real paired test on a real model, no API key involved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .data import MAX_LINE_LEN, TASKS, batch, encode, gen_eval_cases
from .model import TinyGPT, TinyGPTConfig

PHASE1_MIX = {"cpy": 0.25, "rev": 0.25, "srt": 0.25, "max": 0.25}
# Phase 2 = the "upgrade": rev vanishes from the training data, the rest
# get its probability mass (srt slightly favoured so something visibly
# *improves* alongside the regression).
PHASE2_MIX = {"cpy": 0.3, "rev": 0.0, "srt": 0.4, "max": 0.3}


@dataclass
class TrainResult:
    checkpoint_a: Path
    checkpoint_b: Path
    steps: int
    switch_step: int
    final_loss: float
    # task -> accuracy on held-out prompts, measured at each checkpoint.
    accuracy_a: dict[str, float] = field(default_factory=dict)
    accuracy_b: dict[str, float] = field(default_factory=dict)


def _lr_at(step: int, total: int, lr: float) -> float:
    """5% linear warmup, then cosine decay to 10% of peak.

    Applied *per phase* (step/total are phase-relative): phase 2 re-warms
    to the full peak rate, as real continued-pretraining runs do. This is
    load-bearing for the drift story — if the phase-1 cosine were left to
    decay across the whole run, phase 2's updates would be too small to
    displace ``rev`` and both checkpoints would behave identically.
    Empirically, ~300 re-warmed steps on the rev-free mix drive held-out
    rev accuracy from 100% toward 0% while the other tasks hold.
    """
    warmup = max(1, int(0.05 * total))
    if step < warmup:
        return lr * (step + 1) / warmup
    frac = (step - warmup) / max(1, total - warmup)
    return lr * (0.1 + 0.45 * (1.0 + np.cos(np.pi * frac)))


def quick_accuracy(
    model: TinyGPT, per_task: int = 25, seed: int = 20260703
) -> dict[str, float]:
    """Greedy exact-match accuracy per task on held-out prompts.

    A coarse training-time gauge; the statistically honest number comes
    from ``rift compare`` on the committed suite.
    """
    cases = gen_eval_cases(per_task, seed=seed)
    acc: dict[str, list[float]] = {t: [] for t in TASKS}
    for case in cases:
        task = str(case["tags"][0]).split(":", 1)[1]  # type: ignore[index]
        out = model.generate(encode(str(case["input"])), max_new_tokens=8)
        from .data import decode

        acc[task].append(1.0 if decode(out) == case["expected"] else 0.0)
    return {t: float(np.mean(v)) if v else 0.0 for t, v in acc.items()}


def train_riftlm(
    out_dir: str | Path = "models",
    steps: int = 3000,
    switch: float = 0.6,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
    cfg: TinyGPTConfig | None = None,
    log: Callable[[str], None] | None = None,
    eval_at_checkpoints: bool = True,
) -> TrainResult:
    """Train RiftLM from scratch and save the baseline/challenger pair.

    Deterministic for a given ``seed`` (data sampling and init both flow
    from it), so two people running the same command get the same drift
    story to within BLAS reduction noise.
    """
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)
    cfg = cfg or TinyGPTConfig(block_size=MAX_LINE_LEN + 4)
    model = TinyGPT(cfg, seed=seed)
    say = log or (lambda _msg: None)

    n_params = sum(int(v.size) for v in model.params.values())
    switch_step = int(steps * switch)
    say(
        f"RiftLM: {n_params:,} params, {steps} steps "
        f"(task-mix shift at step {switch_step}: rev dropped)"
    )

    path_a = out_dir / "riftlm-a.npz"
    path_b = out_dir / "riftlm-b.npz"
    result = TrainResult(
        checkpoint_a=path_a, checkpoint_b=path_b, steps=steps,
        switch_step=switch_step, final_loss=float("nan"),
    )

    loss = float("nan")
    for step in range(steps):
        in_phase1 = step < switch_step
        mix = PHASE1_MIX if in_phase1 else PHASE2_MIX
        # Phase-relative schedule: phase 2 re-warms to the peak rate.
        if in_phase1:
            step_lr = _lr_at(step, switch_step, lr)
        else:
            step_lr = _lr_at(step - switch_step, steps - switch_step, lr)
        x, y, mask = batch(mix, batch_size, cfg.block_size, rng)
        loss, grads = model.loss_and_grads(x, y, mask)
        model.adam_step(grads, lr=step_lr)

        if step % 200 == 0:
            phase = 1 if step < switch_step else 2
            say(f"step {step:>5}  loss {loss:.4f}  (phase {phase})")

        if step + 1 == switch_step:
            model.save(path_a)
            if eval_at_checkpoints:
                result.accuracy_a = quick_accuracy(model)
                acc = "  ".join(f"{t}={v:.0%}" for t, v in result.accuracy_a.items())
                say(f"saved {path_a}  held-out: {acc}")
            else:
                say(f"saved {path_a}")

    model.save(path_b)
    result.final_loss = float(loss)
    if eval_at_checkpoints:
        result.accuracy_b = quick_accuracy(model)
        acc = "  ".join(f"{t}={v:.0%}" for t, v in result.accuracy_b.items())
        say(f"saved {path_b}  held-out: {acc}")
    else:
        say(f"saved {path_b}")
    return result
