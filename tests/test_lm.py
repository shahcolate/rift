"""Tests for RiftLM: the pure-numpy tiny GPT and its provider wiring."""

import numpy as np
import pytest

from rift.config import resolve_model
from rift.lm import data
from rift.lm.data import (
    TASKS,
    batch,
    encode,
    decode,
    gen_eval_cases,
    is_eval_line,
    sample_line,
)
from rift.lm.model import TinyGPT, TinyGPTConfig
from rift.providers.riftlm import (
    RiftLMCheckpointError,
    RiftLMProvider,
    checkpoint_path,
)

MIX = {"cpy": 0.25, "rev": 0.25, "srt": 0.25, "max": 0.25}


def _micro_model(dtype=np.float64, seed=3) -> TinyGPT:
    cfg = TinyGPTConfig(
        vocab_size=data.VOCAB_SIZE, block_size=8, n_layer=1, n_head=2,
        d_model=16, d_ff=32,
    )
    return TinyGPT(cfg, seed=seed, dtype=dtype)


# --------------------------------------------------------------------- model


def test_backward_matches_finite_differences():
    """The hand-written backprop must agree with central differences.

    This is the guardrail for model.py: any wrong term in _backward shifts
    gradients by orders of magnitude, far beyond the tolerance here. The
    tolerance floor absorbs cancellation noise on near-zero gradients.
    """
    m = _micro_model()
    rng = np.random.default_rng(0)
    x, y, mask = batch(MIX, 3, 8, rng)
    _, grads = m.loss_and_grads(x, y, mask)

    eps = 1e-6
    for name, g in grads.items():
        flat = m.params[name].reshape(-1)
        gf = g.reshape(-1)
        for i in rng.choice(flat.size, size=min(5, flat.size), replace=False):
            old = flat[i]
            flat[i] = old + eps
            lp, _ = m.loss(x, y, mask)
            flat[i] = old - eps
            lm, _ = m.loss(x, y, mask)
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            rel = abs(num - gf[i]) / max(1e-4, abs(num) + abs(gf[i]))
            assert rel < 1e-4, f"{name}[{i}]: analytic {gf[i]} vs numeric {num}"


def test_training_reduces_loss():
    """A short Adam run on real task data must cut the loss sharply."""
    cfg = TinyGPTConfig(block_size=24, n_layer=1, d_model=48, d_ff=96, n_head=2)
    m = TinyGPT(cfg, seed=0)
    rng = np.random.default_rng(1)
    x, y, mask = batch(MIX, 32, 24, rng)  # fixed batch: overfit it
    first, _ = m.loss(x, y, mask)
    for _ in range(150):
        loss, grads = m.loss_and_grads(x, y, mask)
        m.adam_step(grads, lr=3e-3)
    final, _ = m.loss(x, y, mask)
    assert final < first * 0.5, f"loss barely moved: {first:.3f} -> {final:.3f}"


def test_generate_is_deterministic_and_stops_at_newline():
    m = _micro_model(dtype=np.float32)
    ids = encode("cpy abc = ")
    a = m.generate(ids, max_new_tokens=6)
    b = m.generate(ids, max_new_tokens=6)
    assert a == b
    assert data.NEWLINE_ID not in a
    assert len(a) <= 6


def test_save_load_roundtrip(tmp_path):
    m = _micro_model(dtype=np.float32)
    p = tmp_path / "m.npz"
    m.save(p)
    m2 = TinyGPT.load(p)
    assert m2.cfg == m.cfg
    for k, v in m.params.items():
        np.testing.assert_array_equal(v, m2.params[k])
    prompt = encode("rev ab = ")
    assert m.generate(prompt, 5) == m2.generate(prompt, 5)


# ---------------------------------------------------------------------- data


def test_vocab_roundtrip_and_unknown_chars_degrade():
    assert decode(encode("rev abc = cba\n")) == "rev abc = cba\n"
    # Out-of-vocab characters become spaces (after lowercasing), never a crash.
    assert decode(encode("REV a!c")) == "rev a c"


def test_train_eval_split_is_disjoint():
    """The training sampler must never emit an eval-split line."""
    rng = np.random.default_rng(7)
    for _ in range(500):
        assert not is_eval_line(sample_line(MIX, rng))
    # ...and the suite generator must only emit eval-split lines.
    for case in gen_eval_cases(5, seed=99):
        line = f"{case['input']}{case['expected']}\n"
        assert is_eval_line(line)


def test_gen_eval_cases_deterministic_and_answers_correct():
    a = gen_eval_cases(4, seed=42)
    b = gen_eval_cases(4, seed=42)
    assert a == b
    assert len(a) == 4 * len(TASKS)
    for case in a:
        task, arg = str(case["input"]).split()[:2]
        expected = str(case["expected"])
        if task == "cpy":
            assert expected == arg
        elif task == "rev":
            assert expected == arg[::-1]
        elif task == "srt":
            assert expected == "".join(sorted(arg))
        elif task == "max":
            assert expected == max(arg)
        assert case["tags"] == [f"task:{task}"]


def test_batch_mask_covers_line_and_newline_only():
    rng = np.random.default_rng(0)
    x, y, mask = batch(MIX, 4, 24, rng)
    for b in range(4):
        n = int(mask[b].sum())
        assert n > 0
        # The last masked target is the terminating newline...
        assert y[b, n - 1] == data.NEWLINE_ID
        # ...and everything beyond the mask is padding.
        assert (mask[b, n:] == 0).all()


# ------------------------------------------------------------ provider/config


def _saved_checkpoint(tmp_path):
    m = _micro_model(dtype=np.float32)
    p = tmp_path / "ck.npz"
    m.save(p)
    return p


def test_resolve_model_riftlm_appends_weight_digest(tmp_path):
    p = _saved_checkpoint(tmp_path)
    cfg = resolve_model(f"riftlm:{p}")
    assert cfg.provider == "riftlm"
    assert cfg.model.startswith(f"riftlm:{p}@")
    digest1 = cfg.model.rsplit("@", 1)[1]
    assert len(digest1) == 12

    # Resolving an already-digested string is stable (idempotent).
    assert resolve_model(cfg.model).model == cfg.model

    # Retraining in place (different bytes, same path) must change the
    # digest — that is what invalidates the runner's completion cache.
    _micro_model(dtype=np.float32, seed=99).save(p)
    digest2 = resolve_model(f"riftlm:{p}").model.rsplit("@", 1)[1]
    assert digest1 != digest2


def test_resolve_model_riftlm_missing_checkpoint_is_clean_error(tmp_path):
    with pytest.raises(RiftLMCheckpointError):
        resolve_model(f"riftlm:{tmp_path}/nope.npz")


def test_checkpoint_path_strips_prefix_and_digest():
    assert checkpoint_path("riftlm:models/a.npz").name == "a.npz"
    assert str(checkpoint_path("riftlm:models/a.npz@0123456789ab")) == "models/a.npz"


async def test_provider_completion_is_deterministic_and_keyless(tmp_path):
    p = _saved_checkpoint(tmp_path)
    provider = RiftLMProvider(model=f"riftlm:{p}")
    c1 = await provider.complete("cpy abc = ")
    c2 = await provider.complete("cpy abc = ")
    assert c1.output_text == c2.output_text
    assert c1.input_tokens == len("cpy abc = ")
    assert c1.output_tokens == len(c1.output_text)
    assert c1.provider_fingerprint is not None
    assert c1.provider_fingerprint.startswith("riftlm-")
    await provider.close()


async def test_provider_missing_checkpoint(tmp_path):
    with pytest.raises(RiftLMCheckpointError):
        RiftLMProvider(model=f"riftlm:{tmp_path}/gone.npz")


async def test_run_suite_end_to_end_with_riftlm(tmp_path):
    """The full runner path — scoring, cost, fingerprint — on a checkpoint."""
    from rift.config import SuiteConfig
    from rift.runner import run_suite

    p = _saved_checkpoint(tmp_path)
    model_cfg = resolve_model(f"riftlm:{p}")
    suite = SuiteConfig(
        name="riftlm_smoke",
        scoring="exact_match",
        cases=[
            {"input": "cpy abc = ", "expected": "abc", "tags": ["task:cpy"]},
            {"input": "max 391 = ", "expected": "9", "tags": ["task:max"]},
        ],
    )
    result = await run_suite(
        suite, model_cfg, cache_dir=str(tmp_path / "cache"), show_progress=False
    )
    assert len(result.cases) == 2
    for case in result.cases:
        assert case.error is None
        assert case.cost_usd == 0.0  # riftlm has no price: it's your model
        assert case.provider_fingerprint.startswith("riftlm-")
        assert case.score in (0.0, 1.0)  # untrained weights may miss; must score
