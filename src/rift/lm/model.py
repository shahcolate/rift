"""A character-level GPT in pure numpy — forward, backprop, and Adam by hand.

Rift's dependency footprint is click/httpx/yaml/numpy/scipy/rich; adding a
deep-learning framework to train a ~400K-parameter toy would multiply the
install size by an order of magnitude. numpy is already here for the
statistics, and at this scale hand-written backprop is a page of math, not
an engineering project — in the same spirit as the hand-rolled SVG
observatory dashboard.

Architecture: standard pre-norm decoder-only transformer (learned
positional embeddings, causal self-attention, ReLU MLP, untied LM head).
Correctness of every gradient is enforced by a finite-difference check in
``tests/test_lm.py`` — if you touch ``_backward``, run it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .data import NEWLINE_ID, VOCAB_SIZE


@dataclass
class TinyGPTConfig:
    vocab_size: int = VOCAB_SIZE
    block_size: int = 24
    n_layer: int = 2
    n_head: int = 4
    d_model: int = 128
    d_ff: int = 512

    def __post_init__(self) -> None:
        if self.d_model % self.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")


def _relu(x: np.ndarray) -> np.ndarray:
    # ReLU, not GELU: tanh/exp are single-threaded transcendentals and were
    # ~70% of the measured step time; the original-transformer activation
    # trains this toy just as well at a third of the wall clock.
    return np.maximum(x, 0.0)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(x.dtype)


def _layernorm_fwd(
    x: np.ndarray, g: np.ndarray, b: np.ndarray, eps: float = 1e-5
) -> tuple[np.ndarray, tuple]:
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    istd = 1.0 / np.sqrt(var + eps)
    xhat = (x - mu) * istd
    return g * xhat + b, (xhat, istd, g)


def _layernorm_bwd(dy: np.ndarray, cache: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xhat, istd, g = cache
    dg = (dy * xhat).sum(axis=tuple(range(dy.ndim - 1)))
    db = dy.sum(axis=tuple(range(dy.ndim - 1)))
    dxhat = dy * g
    dx = istd * (
        dxhat
        - dxhat.mean(axis=-1, keepdims=True)
        - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True)
    )
    return dx, dg, db


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class TinyGPT:
    """The model: a flat dict of numpy weights plus the math to move them."""

    def __init__(
        self,
        cfg: TinyGPTConfig | None = None,
        seed: int = 0,
        dtype: type = np.float32,
    ) -> None:
        self.cfg = cfg or TinyGPTConfig()
        rng = np.random.default_rng(seed)
        c = self.cfg
        std = 0.02
        # Residual-branch output projections get the GPT-2 depth-scaled
        # init so activations don't grow with n_layer.
        res_std = std / np.sqrt(2 * c.n_layer)

        def w(*shape: int, s: float = std) -> np.ndarray:
            return (rng.normal(0.0, s, size=shape)).astype(dtype)

        p: dict[str, np.ndarray] = {
            "tok_emb": w(c.vocab_size, c.d_model),
            "pos_emb": w(c.block_size, c.d_model),
            "lnf.g": np.ones(c.d_model, dtype=dtype),
            "lnf.b": np.zeros(c.d_model, dtype=dtype),
            "head": w(c.d_model, c.vocab_size),
        }
        for i in range(c.n_layer):
            p[f"l{i}.ln1.g"] = np.ones(c.d_model, dtype=dtype)
            p[f"l{i}.ln1.b"] = np.zeros(c.d_model, dtype=dtype)
            p[f"l{i}.wqkv"] = w(c.d_model, 3 * c.d_model)
            p[f"l{i}.wo"] = w(c.d_model, c.d_model, s=res_std)
            p[f"l{i}.ln2.g"] = np.ones(c.d_model, dtype=dtype)
            p[f"l{i}.ln2.b"] = np.zeros(c.d_model, dtype=dtype)
            p[f"l{i}.w1"] = w(c.d_model, c.d_ff)
            p[f"l{i}.b1"] = np.zeros(c.d_ff, dtype=dtype)
            p[f"l{i}.w2"] = w(c.d_ff, c.d_model, s=res_std)
            p[f"l{i}.b2"] = np.zeros(c.d_model, dtype=dtype)
        self.params = p
        # Adam state, allocated lazily on the first step.
        self._adam_m: dict[str, np.ndarray] | None = None
        self._adam_v: dict[str, np.ndarray] | None = None
        self._adam_t = 0

    # ---------------------------------------------------------------- forward

    def _forward(self, idx: np.ndarray) -> tuple[np.ndarray, dict]:
        """Logits for a batch of token ids ``idx`` of shape (B, T).

        Returns ``(logits, cache)``; the cache holds every intermediate
        ``_backward`` needs.
        """
        p, c = self.params, self.cfg
        B, T = idx.shape
        if T > c.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {c.block_size}")
        H, hd = c.n_head, c.d_model // c.n_head
        scale = 1.0 / np.sqrt(hd)

        x = p["tok_emb"][idx] + p["pos_emb"][:T]
        # Causal mask: large negative (not -inf) so float32 softmax never NaNs.
        causal = np.triu(np.full((T, T), -1e9, dtype=x.dtype), k=1)

        cache: dict = {"idx": idx, "T": T, "layers": []}
        for i in range(c.n_layer):
            x0 = x
            y1, ln1c = _layernorm_fwd(x0, p[f"l{i}.ln1.g"], p[f"l{i}.ln1.b"])
            qkv = y1 @ p[f"l{i}.wqkv"]  # (B, T, 3D)
            q, k, v = np.split(qkv, 3, axis=-1)
            # (B, T, D) -> (B, H, T, hd)
            q = q.reshape(B, T, H, hd).transpose(0, 2, 1, 3)
            k = k.reshape(B, T, H, hd).transpose(0, 2, 1, 3)
            v = v.reshape(B, T, H, hd).transpose(0, 2, 1, 3)
            scores = q @ k.transpose(0, 1, 3, 2) * scale + causal
            att = _softmax(scores)  # (B, H, T, T)
            ctx = att @ v  # (B, H, T, hd)
            merged = ctx.transpose(0, 2, 1, 3).reshape(B, T, c.d_model)
            o = merged @ p[f"l{i}.wo"]
            x1 = x0 + o

            y2, ln2c = _layernorm_fwd(x1, p[f"l{i}.ln2.g"], p[f"l{i}.ln2.b"])
            hpre = y2 @ p[f"l{i}.w1"] + p[f"l{i}.b1"]
            h = _relu(hpre)
            x = x1 + h @ p[f"l{i}.w2"] + p[f"l{i}.b2"]

            cache["layers"].append(
                dict(y1=y1, ln1c=ln1c, q=q, k=k, v=v, att=att, merged=merged,
                     y2=y2, ln2c=ln2c, hpre=hpre, h=h, scale=scale)
            )
        yf, lnfc = _layernorm_fwd(x, p["lnf.g"], p["lnf.b"])
        cache["yf"], cache["lnfc"] = yf, lnfc
        logits = yf @ p["head"]
        return logits, cache

    def loss(
        self, x: np.ndarray, y: np.ndarray, mask: np.ndarray
    ) -> tuple[float, dict]:
        """Masked mean cross-entropy; also returns the forward/backward cache."""
        logits, cache = self._forward(x)
        probs = _softmax(logits.astype(np.float64))
        B, T = x.shape
        n = max(mask.sum(), 1.0)
        picked = probs[np.arange(B)[:, None], np.arange(T)[None, :], y]
        loss = float(-(np.log(np.maximum(picked, 1e-12)) * mask).sum() / n)
        cache["probs"], cache["y"], cache["mask"], cache["n"] = probs, y, mask, n
        return loss, cache

    # --------------------------------------------------------------- backward

    def _backward(self, cache: dict) -> dict[str, np.ndarray]:
        """Gradients of the masked cross-entropy w.r.t. every parameter."""
        p, c = self.params, self.cfg
        idx, T = cache["idx"], cache["T"]
        B = idx.shape[0]
        H, hd = c.n_head, c.d_model // c.n_head
        probs, y, mask, n = cache["probs"], cache["y"], cache["mask"], cache["n"]

        grads: dict[str, np.ndarray] = {}

        dlogits = probs.copy()
        dlogits[np.arange(B)[:, None], np.arange(T)[None, :], y] -= 1.0
        dlogits *= (mask / n)[..., None]
        dlogits = dlogits.astype(p["head"].dtype)

        yf = cache["yf"]
        grads["head"] = yf.reshape(-1, c.d_model).T @ dlogits.reshape(-1, c.vocab_size)
        dyf = dlogits @ p["head"].T
        dx, grads["lnf.g"], grads["lnf.b"] = _layernorm_bwd(dyf, cache["lnfc"])

        for i in reversed(range(c.n_layer)):
            L = cache["layers"][i]
            # MLP block: x_out = x1 + gelu(y2 @ w1 + b1) @ w2 + b2
            dmo = dx  # grad into the MLP residual branch
            grads[f"l{i}.w2"] = L["h"].reshape(-1, c.d_ff).T @ dmo.reshape(-1, c.d_model)
            grads[f"l{i}.b2"] = dmo.sum(axis=(0, 1))
            dh = dmo @ p[f"l{i}.w2"].T
            dhpre = dh * _relu_grad(L["hpre"])
            grads[f"l{i}.w1"] = L["y2"].reshape(-1, c.d_model).T @ dhpre.reshape(-1, c.d_ff)
            grads[f"l{i}.b1"] = dhpre.sum(axis=(0, 1))
            dy2 = dhpre @ p[f"l{i}.w1"].T
            dln2, grads[f"l{i}.ln2.g"], grads[f"l{i}.ln2.b"] = _layernorm_bwd(dy2, L["ln2c"])
            dx1 = dx + dln2  # residual + normed path

            # Attention block: x1 = x0 + merge(att @ v) @ wo
            do = dx1
            grads[f"l{i}.wo"] = (
                L["merged"].reshape(-1, c.d_model).T @ do.reshape(-1, c.d_model)
            )
            dmerged = do @ p[f"l{i}.wo"].T
            dctx = dmerged.reshape(B, T, H, hd).transpose(0, 2, 1, 3)
            datt = dctx @ L["v"].transpose(0, 1, 3, 2)
            dv = L["att"].transpose(0, 1, 3, 2) @ dctx
            # softmax backward; masked entries have att==0 hence gradient 0.
            att = L["att"]
            dscores = att * (datt - (datt * att).sum(axis=-1, keepdims=True))
            dq = dscores @ L["k"] * L["scale"]
            dk = dscores.transpose(0, 1, 3, 2) @ L["q"] * L["scale"]
            dqkv = np.concatenate(
                [
                    g.transpose(0, 2, 1, 3).reshape(B, T, c.d_model)
                    for g in (dq, dk, dv)
                ],
                axis=-1,
            )
            grads[f"l{i}.wqkv"] = (
                L["y1"].reshape(-1, c.d_model).T @ dqkv.reshape(-1, 3 * c.d_model)
            )
            dy1 = dqkv @ p[f"l{i}.wqkv"].T
            dln1, grads[f"l{i}.ln1.g"], grads[f"l{i}.ln1.b"] = _layernorm_bwd(dy1, L["ln1c"])
            dx = dx1 + dln1

        grads["pos_emb"] = np.zeros_like(p["pos_emb"])
        grads["pos_emb"][:T] = dx.sum(axis=0)
        grads["tok_emb"] = np.zeros_like(p["tok_emb"])
        np.add.at(grads["tok_emb"], idx, dx)
        return grads

    def loss_and_grads(
        self, x: np.ndarray, y: np.ndarray, mask: np.ndarray
    ) -> tuple[float, dict[str, np.ndarray]]:
        loss, cache = self.loss(x, y, mask)
        return loss, self._backward(cache)

    # -------------------------------------------------------------- optimizer

    def adam_step(
        self,
        grads: dict[str, np.ndarray],
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.99,
        eps: float = 1e-8,
    ) -> None:
        if self._adam_m is None or self._adam_v is None:
            self._adam_m = {k: np.zeros_like(v) for k, v in self.params.items()}
            self._adam_v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._adam_t += 1
        t = self._adam_t
        for k, g in grads.items():
            g = g.astype(self.params[k].dtype)
            m = self._adam_m[k]
            v = self._adam_v[k]
            m += (1 - beta1) * (g - m)
            v += (1 - beta2) * (g * g - v)
            mhat = m / (1 - beta1**t)
            vhat = v / (1 - beta2**t)
            self.params[k] -= (lr * mhat / (np.sqrt(vhat) + eps)).astype(
                self.params[k].dtype
            )

    # -------------------------------------------------------------- inference

    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 32,
        stop_id: int | None = NEWLINE_ID,
    ) -> list[int]:
        """Greedy decode (temperature 0 — deterministic, cache-friendly).

        Returns only the newly generated ids, excluding the stop token.
        A prompt longer than the context window is cropped to its tail,
        the standard sliding-window behaviour.
        """
        ids = list(prompt_ids)
        out: list[int] = []
        if not ids:
            # An empty prompt has no last position to read logits from;
            # return an empty completion rather than crash on a zero-size
            # argmax.
            return out
        for _ in range(max_new_tokens):
            window = ids[-self.cfg.block_size :]
            logits, _ = self._forward(np.array([window], dtype=np.int64))
            nxt = int(np.argmax(logits[0, -1]))
            if stop_id is not None and nxt == stop_id:
                break
            ids.append(nxt)
            out.append(nxt)
        return out

    # ------------------------------------------------------------ persistence

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic (tmp + rename), matching the runner's cache discipline.
        tmp = path.with_suffix(path.suffix + ".tmp")
        # NOTE: no allow_pickle kwarg — np.savez_compressed only grew it in
        # numpy 2.1, and on the 1.26–2.0 range pyproject permits, the kwarg
        # would be silently swallowed into **kwds and written into the
        # checkpoint as a stray array. We only store plain ndarrays, so
        # pickle never engages on save regardless.
        arrays = {"__config__": np.array(json.dumps(asdict(self.cfg))), **self.params}
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **arrays)  # type: ignore[arg-type]
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "TinyGPT":
        with np.load(path) as z:
            cfg = TinyGPTConfig(**json.loads(str(z["__config__"])))
            model = cls(cfg)
            # Keep only real parameter tensors: skip the config blob and any
            # stray scalar (e.g. an 'allow_pickle' array that checkpoints
            # written by older Rift under numpy<2.1 may carry).
            model.params = {
                k: z[k].copy() for k in z.files
                if k in model.params
            }
        return model
