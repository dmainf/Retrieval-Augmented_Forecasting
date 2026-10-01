"""Chronos-2 forward for the parallel container, computing only the tokens that exist.

An example row is NaN everywhere except its last seq_len context points and its
pred_len future, so out of the (H / 16) + 1 + (pred_len / 16) tokens of a row
only seq_len / 16 + 1 + pred_len / 16 carry anything (11 of 132 at H = 2032).
The rest are masked out of both time and group attention, so they never reach a
valid token. This module drops them: the query row runs at full length, every
example row runs on its last S tokens only, and group attention mixes rows at
those S positions. The result equals Chronos2Model.forward on the padded batch
(checked by `check`), at a cost that grows by 11 tokens per example instead of
132, which is what makes hundreds of examples per query affordable.

It is differentiable, so the same path trains latent example rows and takes
per-example gradients for curriculum latent gradient selection.
"""
import numpy as np
import torch
from einops import rearrange

MODEL_QUANTILES = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                   0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
REPORT_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
REPORT_INDEX = [MODEL_QUANTILES.index(q) for q in REPORT_QUANTILES]


def load_chronos2(name: str = "amazon/chronos-2", threads: int = 0):
    from chronos import Chronos2Pipeline
    if threads > 0:
        torch.set_num_threads(threads)
    model = Chronos2Pipeline.from_pretrained(name, device_map="cpu").model
    model.eval()
    model.requires_grad_(False)
    return model


class SparseRows:
    """Query rows of length H, each grouped with K example rows of seq_len + pred_len."""

    def __init__(self, model, history: int, seq_len: int, pred_len: int):
        cfg = model.chronos_config
        p = cfg.input_patch_size
        if history % p or seq_len % p or pred_len % cfg.output_patch_size:
            raise ValueError("history, seq_len and pred_len must be multiples of the patch size")
        self.m = model
        self.h, self.seq_len, self.pred_len = history, seq_len, pred_len
        self.n_ctx = history // p
        self.n_out = pred_len // cfg.output_patch_size
        self.n_ex_ctx = seq_len // p
        self.t = self.n_ctx + 1 + self.n_out
        self.s = self.n_ex_ctx + 1 + self.n_out

    def _embed(self, context, future):
        """context (N, H) with NaN for missing, future (N, pred_len) or None -> tokens, mask, loc_scale."""
        m = self.m
        patched, mask, loc_scale = m._prepare_patched_context(context=context)
        emb = m.input_patch_embedding(patched)
        reg = m.shared(torch.full((len(context), 1), m.config.reg_token_id))
        fut, _ = m._prepare_patched_future(future_covariates=future, future_covariates_mask=None,
                                           loc_scale=loc_scale, num_output_patches=self.n_out,
                                           batch_size=len(context))
        fut = m.input_patch_embedding(fut)
        tokens = torch.cat([emb, reg, fut], dim=1)
        mask = torch.cat([mask.to(tokens.dtype), torch.ones(len(context), 1 + self.n_out)], dim=1)
        return tokens, mask, loc_scale

    def _example_tokens(self, ex_ctx, ex_fut):
        """(N, seq_len), (N, pred_len) -> the last S tokens of each padded example row.

        The row is never padded with NaN here: the time encoding of the last
        seq_len points is the same whatever the padding, and a NaN pad would turn
        the instance-norm gradient into NaN (0 * NaN), breaking the latent rows.
        """
        n = ex_ctx.shape[0]
        m = self.m
        patched, mask, loc_scale = m._prepare_patched_context(context=ex_ctx)
        emb = m.input_patch_embedding(patched)
        reg = m.shared(torch.full((n, 1), m.config.reg_token_id))
        fut, _ = m._prepare_patched_future(future_covariates=ex_fut, future_covariates_mask=None,
                                           loc_scale=loc_scale, num_output_patches=self.n_out,
                                           batch_size=n)
        fut = m.input_patch_embedding(fut)
        tokens = torch.cat([emb, reg, fut], dim=1)
        mask = torch.cat([mask.to(tokens.dtype),
                          torch.ones(n, 1 + self.n_out)], dim=1)
        return tokens, mask

    @staticmethod
    def _invert(mask):
        return (1.0 - mask) * torch.finfo(torch.float32).min

    def forward(self, context, ex_ctx=None, ex_fut=None):
        """context (B, H); ex_ctx (B or 1, K, seq_len); ex_fut (B or 1, K, pred_len).

        A leading 1 on the examples shares one set across every query -- the
        task-level case. Returns (normalized quantile preds (B, 21, pred_len),
        loc_scale) so a caller can take a loss in normalized space or unscale.
        """
        m = self.m
        b = context.shape[0]
        q, qmask, loc_scale = self._embed(context, None)
        k = 0 if ex_ctx is None else ex_ctx.shape[1]
        if k:
            shared = ex_ctx.shape[0] == 1
            e, emask = self._example_tokens(ex_ctx.reshape(-1, self.seq_len),
                                            ex_fut.reshape(-1, self.pred_len))
            e = e.reshape(-1, k, self.s, e.shape[-1])
            emask = emask.reshape(-1, k, self.s)
            if shared:
                e = e.expand(b, -1, -1, -1)
                emask = emask.expand(b, -1, -1)
            e = e.reshape(b * k, self.s, -1)
            emask = emask.reshape(b * k, self.s)

        pos_q = torch.arange(self.t).unsqueeze(0)
        pos_e = torch.arange(self.t - self.s, self.t).unsqueeze(0)
        tmask_q = self._invert(qmask)[:, None, None, :]
        if k:
            tmask_e = self._invert(emask)[:, None, None, :]
            # group rows at the shared S positions: query first, then its K examples
            gvalid = torch.cat([qmask[:, -self.s:].unsqueeze(1), emask.reshape(b, k, self.s)], dim=1)
            gmask = self._invert(rearrange(gvalid, "b r s -> (b s) r"))[:, None, None, :]

        for block in m.encoder.block:
            tsa, gsa, ffn = block.layer
            q = tsa(q, attention_mask=tmask_q, position_ids=pos_q)[0]
            if k:
                e = tsa(e, attention_mask=tmask_e, position_ids=pos_e)[0]
            # group attention: a lone query row before the shared positions
            head = q[:, :self.t - self.s]
            n_head = gsa.layer_norm(head).reshape(-1, 1, head.shape[-1])
            out = gsa.self_attention(n_head, mask=torch.zeros(1, 1, 1, 1))[0]
            head = head + out.reshape(head.shape)
            tail = q[:, self.t - self.s:]
            if k:
                rows = torch.cat([tail.unsqueeze(1), e.reshape(b, k, self.s, -1)], dim=1)
                rows = rearrange(rows, "b r s d -> (b s) r d")
                out = gsa.self_attention(gsa.layer_norm(rows), mask=gmask)[0]
                rows = rows + out
                rows = rearrange(rows, "(b s) r d -> b r s d", b=b, s=self.s)
                tail, e = rows[:, 0], rows[:, 1:].reshape(b * k, self.s, -1)
            else:
                n_tail = gsa.layer_norm(tail).reshape(-1, 1, tail.shape[-1])
                tail = tail + gsa.self_attention(n_tail, mask=torch.zeros(1, 1, 1, 1))[0].reshape(tail.shape)
            q = ffn(torch.cat([head, tail], dim=1))
            if k:
                e = ffn(e)

        h = m.encoder.final_layer_norm(q[:, -self.n_out:])
        preds = m.output_patch_embedding(h)
        preds = rearrange(preds, "b n (q p) -> b q (n p)", n=self.n_out, q=m.num_quantiles,
                          p=m.chronos_config.output_patch_size)
        return preds, loc_scale

    def unscale(self, preds, loc_scale):
        b, nq, h = preds.shape
        flat = self.m.instance_norm.inverse(preds.reshape(b, nq * h), loc_scale)
        return flat.reshape(b, nq, h)

    def loss(self, preds, loc_scale, future):
        """Pinball loss in the query's normalized space, as Chronos-2 trains: mean over
        horizon, sum over quantiles, mean over the batch. Returns one value per query."""
        y, _ = self.m.instance_norm(future, loc_scale)
        levels = torch.tensor(MODEL_QUANTILES).view(1, -1, 1)
        diff = y.unsqueeze(1) - preds
        pin = torch.maximum(levels * diff, (levels - 1) * diff)
        return pin.mean(dim=-1).sum(dim=-1)

    @torch.no_grad()
    def predict(self, context, ex_ctx=None, ex_fut=None, chunk: int = 32) -> np.ndarray:
        """Raw-scale forecasts at REPORT_QUANTILES, (B, 9, pred_len)."""
        out = []
        for lo in range(0, len(context), chunk):
            c = torch.as_tensor(context[lo:lo + chunk], dtype=torch.float32)
            if ex_ctx is not None and ex_ctx.shape[0] != 1:
                ec = torch.as_tensor(ex_ctx[lo:lo + chunk], dtype=torch.float32)
                ef = torch.as_tensor(ex_fut[lo:lo + chunk], dtype=torch.float32)
            elif ex_ctx is not None:
                ec = torch.as_tensor(ex_ctx, dtype=torch.float32)
                ef = torch.as_tensor(ex_fut, dtype=torch.float32)
            else:
                ec = ef = None
            preds, ls = self.forward(c, ec, ef)
            out.append(self.unscale(preds, ls)[:, REPORT_INDEX].numpy())
        return np.concatenate(out)


def check(seed: int = 0, b: int = 3, k: int = 5, history: int = 2032):
    """Compare against Chronos2Model.forward on the explicitly padded batch."""
    torch.manual_seed(seed)
    model = load_chronos2()
    sr = SparseRows(model, history, 96, 64)
    ctx = torch.randn(b, history).cumsum(-1)
    ctx[0, :500] = float("nan")
    ex_ctx = torch.randn(b, k, 96).cumsum(-1) + 3
    ex_fut = torch.randn(b, k, 64).cumsum(-1) + 3
    with torch.no_grad():
        mine, ls = sr.forward(ctx, ex_ctx, ex_fut)
        mine = sr.unscale(mine, ls)
        rows, futs, gids = [], [], []
        for i in range(b):
            rows.append(ctx[i])
            futs.append(torch.full((64,), float("nan")))
            gids.append(i)
            for j in range(k):
                r = torch.full((history,), float("nan"))
                r[-96:] = ex_ctx[i, j]
                rows.append(r)
                futs.append(ex_fut[i, j])
                gids.append(i)
        ref = model(context=torch.stack(rows), group_ids=torch.tensor(gids),
                    future_covariates=torch.stack(futs), num_output_patches=4).quantile_preds
        ref = ref[torch.arange(b) * (k + 1)]
        mine0, ls0 = sr.forward(ctx, None, None)
        ref0 = model(context=ctx, num_output_patches=4).quantile_preds
    print("with examples  max |diff|", float((mine - ref).abs().max()),
          " scale", float(ref.abs().mean()))
    print("no examples    max |diff|", float((sr.unscale(mine0, ls0) - ref0).abs().max()))


if __name__ == "__main__":
    check()
