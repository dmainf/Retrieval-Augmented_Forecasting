"""In-context Time Series Predictor: one token per forecasting task.

Reimplementation of Lu, Sun and Yang, "In-context Time Series Predictor"
(ICLR 2025, arXiv:2405.14982) -- not the authors' code, and not aimed at
reproducing their benchmark table. The point here is the token format.

Chronos is a Temporal-wise Transformer: a token is 16 consecutive points of one
series. A retrieved example's context and its future therefore land in
different tokens, and the product

    dW = -(eta/N) sum_i (W x_i - y_i) x_i^T

that one attention layer would need in order to learn from the example cannot be
formed inside any single token. That is the structural reason the ablations keep
finding the context-to-future pairing unused (+0.2% on Bolt, +2.8% on Chronos-2
when the futures are permuted among the retrieved windows).

ICTSP puts a whole (lookback, future) pair in one token:

    token i   = [ pos | x^(i .. i+L_b) | y^(i+L_b .. i+L_b+L_P) ]
    target    = [ pos | x^(L_I-L_b .. L_I) |        0            ]

which is exactly the [[p],[x],[y]] layout the ICL constructions assume, so the
product is available. Whether the model then uses it is an empirical question --
`--example-ablation shuffle-future` is what answers it.

Deviations from the paper, all deliberate:

  * L_b = 96, L_P = 64 so that one token is exactly the 160-point window this
    repo's retrieval database is built from. The paper uses L_b = 512 and
    L_P in {96, 192, 336, 720}.
  * The output head emits 9 quantiles instead of a point forecast, so the
    existing `evaluate.py` (channel-normalized MSE and quantile loss) applies
    unchanged and the numbers sit beside the Chronos runs.
  * The loss is on target tokens only. Context tokens carry their own futures,
    so supervising them under bidirectional attention would be a copy task.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ABLATIONS = ("none", "shuffle-future", "random-future", "drop-future",
             "shuffle-time", "no-context")


def context_starts(hist_len: int, lookback: int, pred_len: int, step: int) -> np.ndarray:
    """Start offsets of the context tokens inside a history of `hist_len` points.

    The most recent context token ends exactly at the forecast origin, so its
    future half is the last `pred_len` observed points -- adjacent to, but never
    overlapping, what the target token has to predict. Older tokens are spaced
    `step` apart; a larger step trades context examples for compute.
    """
    last = hist_len - lookback - pred_len
    if last < 0:
        raise ValueError(
            f"history {hist_len} is shorter than one token ({lookback} + {pred_len})"
        )
    return np.arange(last, 0, -step)[::-1].copy()


def build_tokens(hist: torch.Tensor, lookback: int, pred_len: int,
                 starts: np.ndarray) -> torch.Tensor:
    """hist (B, C, hist_len) -> tokens (B, C, T, lookback + pred_len).

    T = len(starts) + 1; the last token along that axis is the target, whose
    future half is zero-filled. Everything is laid out per channel and only
    flattened when it reaches the transformer, so an ablation can still address
    "the context tokens of this series".
    """
    b, c, hist_len = hist.shape
    win = lookback + pred_len
    idx = torch.as_tensor(np.asarray(starts, dtype=np.int64), device=hist.device)
    offs = torch.arange(win, device=hist.device)
    gather = (idx[:, None] + offs[None, :]).reshape(-1)          # (T-1 * win)
    ctx = hist.index_select(-1, gather).view(b, c, len(starts), win)

    target = torch.zeros(b, c, 1, win, dtype=hist.dtype, device=hist.device)
    target[..., :lookback] = hist[:, :, hist_len - lookback:].unsqueeze(2)
    return torch.cat([ctx, target], dim=2)


def ablate(tokens: torch.Tensor, kind: str, lookback: int,
           generator: torch.Generator | None = None) -> torch.Tensor:
    """Damage the context tokens to test whether the pairing is being used.

    The target token is never touched. `shuffle-future` is the decisive one: it
    permutes the future halves among the context tokens, so the same futures are
    all still present and only the context-to-future correspondence is broken.
    If accuracy does not move, no in-context learning is happening no matter
    what the token format allows.
    """
    if kind == "none":
        return tokens
    if kind == "no-context":
        return tokens[:, :, -1:, :]

    out = tokens.clone()
    ctx, fut = out[:, :, :-1, :lookback], out[:, :, :-1, lookback:]
    b, c, t, _ = fut.shape

    if kind == "shuffle-future":
        perm = torch.argsort(torch.rand(b, c, t, device=fut.device), dim=-1)
        fut.copy_(torch.gather(fut, 2, perm[..., None].expand_as(fut)))
    elif kind == "random-future":
        fut.copy_(fut[torch.randperm(b, device=fut.device)].flip(2))
    elif kind == "drop-future":
        fut.zero_()
    elif kind == "shuffle-time":
        perm = torch.argsort(torch.rand_like(fut), dim=-1)
        fut.copy_(torch.gather(fut, 3, perm))
    else:
        raise ValueError(f"unknown ablation {kind}; choose from {ABLATIONS}")
    del ctx
    return out


class Block(nn.Module):
    """One pre-norm layer, following equation (1) of the paper.

        a   = z + Attn(LN(z))
        out = a + LN(FFN(a))
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout))
        self.ff_norm = nn.LayerNorm(d_model)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.norm(z)
        a = z + self.attn(h, h, h, need_weights=False)[0]
        return a + self.ff_norm(self.ff(a))


class ICTSP(nn.Module):
    """Tokens are forecasting tasks; attention is over tasks, not timesteps.

    Positional embeddings index a token's recency rank, not its channel, so the
    same weights apply to a dataset with a different number of columns -- that
    is what makes zero-shot transfer across datasets possible at all.
    """

    def __init__(self, lookback: int = 96, pred_len: int = 64, d_model: int = 128,
                 n_heads: int = 8, n_layers: int = 3, d_ff: int = 256,
                 dropout: float = 0.1, max_tokens: int = 512,
                 quantiles: list[float] | None = None):
        super().__init__()
        self.lookback = lookback
        self.pred_len = pred_len
        self.quantiles = list(quantiles or QUANTILES)
        self.win = lookback + pred_len

        self.proj_in = nn.Linear(self.win, d_model)
        # index 0 is the target token; 1.. are context tokens, most recent first
        self.pos = nn.Embedding(max_tokens + 1, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.norm_out = nn.LayerNorm(d_model)
        self.proj_out = nn.Linear(d_model, pred_len * len(self.quantiles))
        self.max_tokens = max_tokens

    @property
    def config(self) -> dict:
        return {"lookback": self.lookback, "pred_len": self.pred_len,
                "d_model": self.proj_in.out_features,
                "n_heads": self.blocks[0].attn.num_heads,
                "n_layers": len(self.blocks),
                "d_ff": self.blocks[0].ff[0].out_features,
                "max_tokens": self.max_tokens, "quantiles": self.quantiles}

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens (B, C, T, win) -> quantile forecasts (B, C, n_quantiles, pred_len).

        All C * T tokens attend to each other in one sequence: that is how a
        series can borrow an example from a different column, which is the only
        cross-channel path in the model.
        """
        b, c, t, win = tokens.shape
        if win != self.win:
            raise ValueError(f"token width {win} != lookback + pred_len ({self.win})")
        if t > self.max_tokens + 1:
            raise ValueError(
                f"{t} tokens per series exceeds max_tokens {self.max_tokens}; "
                "raise --max-tokens or the sampling step"
            )

        z = self.proj_in(tokens)
        # rank 0 for the target, then 1 for the newest context token upwards
        rank = torch.arange(t - 1, -1, -1, device=tokens.device)
        z = z + self.pos(rank)[None, None, :, :]

        z = z.reshape(b, c * t, -1)
        for block in self.blocks:
            z = block(z)
        z = self.norm_out(z.view(b, c, t, -1)[:, :, -1, :])

        out = self.proj_out(z)
        return out.view(b, c, len(self.quantiles), self.pred_len)


def quantile_loss(pred: torch.Tensor, true: torch.Tensor,
                  quantiles: list[float]) -> torch.Tensor:
    """Pinball loss. pred (B, C, Q, H), true (B, C, H)."""
    q = torch.as_tensor(quantiles, dtype=pred.dtype, device=pred.device)
    diff = true.unsqueeze(2) - pred
    return torch.maximum(q[None, None, :, None] * diff,
                         (q[None, None, :, None] - 1.0) * diff).mean()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
