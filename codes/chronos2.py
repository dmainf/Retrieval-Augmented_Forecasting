"""Frozen Chronos-2 that reads reference examples, in either of two containers.

parallel  every example becomes its own covariate row beside the query: its
          seq_len context sits on the last seq_len points of the history, its
          pred_len future is a known future covariate on the forecast horizon,
          and group attention mixes the rows position by position. The query
          history is untouched and every row is instance-normed on its own.
concat    the usual RAF layout: examples spliced into the time axis in front of
          the history, nearest last, a separator patch after each one.

An example row is NaN except for its own window, so of the (H/16) + 1 + 4
tokens of a padded row only seq_len/16 + 1 + pred_len/16 carry anything (11 of
132 at H = 2032); the rest are masked out of both attentions and never reach a
valid token. The parallel forward therefore runs each example row on those 11
tokens only. It equals Chronos2Model.forward on the padded batch (see `check`)
at a cost of 11 tokens per example instead of 132, which is what makes hundreds
of examples per query affordable.
"""
import numpy as np
import torch
from einops import rearrange

MODEL_QUANTILES = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                   0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
REPORT_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
REPORT_INDEX = [MODEL_QUANTILES.index(q) for q in REPORT_QUANTILES]
CONTEXT_LIMIT = 8192
TOKEN_BUDGET = 24576


def load_chronos2(name: str = "amazon/chronos-2", threads: int = 0):
    from chronos import Chronos2Pipeline
    if threads > 0:
        torch.set_num_threads(threads)
    model = Chronos2Pipeline.from_pretrained(name, device_map="cpu").model
    model.eval()
    model.requires_grad_(False)
    return model


def concat_context(history: np.ndarray, examples: np.ndarray, sep_len: int = 16) -> np.ndarray:
    """history (B, H), examples (B or 1, K, W), column 0 nearest ->
    (B, K * (W + sep_len) + H) laid out as [x_K | sep] ... [x_1 | sep] [history].

    The separator is the query history's mean (qmean), so its level does not move with K.
    """
    b = history.shape[0]
    ex = np.broadcast_to(examples, (b,) + examples.shape[1:])[:, ::-1]
    k, w = ex.shape[1], ex.shape[2]
    sep = np.repeat(np.nanmean(history, axis=1, keepdims=True), sep_len, axis=1)
    parts = []
    for j in range(k):
        parts += [ex[:, j], sep]
    out = np.concatenate(parts + [history], axis=1).astype(np.float32)
    if out.shape[1] > CONTEXT_LIMIT:
        raise ValueError(f"concatenated input {out.shape[1]} exceeds the {CONTEXT_LIMIT}-point "
                         f"context; at most {(CONTEXT_LIMIT - history.shape[1]) // (w + sep_len)} "
                         "examples fit")
    return out


class Chronos2Rows:
    """Query rows of length H, each grouped with K example rows of seq_len + pred_len."""

    def __init__(self, model, seq_len: int, pred_len: int):
        cfg = model.chronos_config
        self.patch = cfg.input_patch_size
        if seq_len % self.patch or pred_len % cfg.output_patch_size:
            raise ValueError("seq_len and pred_len must be multiples of the patch size")
        self.m = model
        self.seq_len, self.pred_len = seq_len, pred_len
        self.n_out = pred_len // cfg.output_patch_size
        self.s = seq_len // self.patch + 1 + self.n_out

    def _embed(self, context, future=None):
        """(N, L) context with NaN for missing, (N, pred_len) future or None ->
        tokens [context patches | REG | future patches], their mask, loc/scale."""
        m = self.m
        patched, mask, loc_scale = m._prepare_patched_context(context=context)
        n = len(context)
        tokens = torch.cat([
            m.input_patch_embedding(patched),
            m.shared(torch.full((n, 1), m.config.reg_token_id)),
            m.input_patch_embedding(m._prepare_patched_future(
                future_covariates=future, future_covariates_mask=None, loc_scale=loc_scale,
                num_output_patches=self.n_out, batch_size=n)[0]),
        ], dim=1)
        mask = torch.cat([mask.to(tokens.dtype), torch.ones(n, 1 + self.n_out)], dim=1)
        return tokens, mask, loc_scale

    @staticmethod
    def _invert(mask):
        return (1.0 - mask) * torch.finfo(torch.float32).min

    def forward(self, context, ex_ctx=None, ex_fut=None):
        """context (B, L); ex_ctx (B or 1, K, seq_len); ex_fut (B or 1, K, pred_len).

        A leading 1 on the examples shares one set across every query. Returns
        normalized quantile predictions (B, 21, pred_len) and the query loc/scale.
        Example contexts are embedded at their own length: the time encoding of
        the last seq_len points does not depend on the padding in front.
        """
        m = self.m
        b = context.shape[0]
        q, qmask, loc_scale = self._embed(context)
        t = q.shape[1]
        k = 0 if ex_ctx is None else ex_ctx.shape[1]
        if k:
            e, emask, _ = self._embed(ex_ctx.reshape(-1, self.seq_len),
                                      ex_fut.reshape(-1, self.pred_len))
            e = e.reshape(-1, k, self.s, e.shape[-1]).expand(b, -1, -1, -1).reshape(b * k, self.s, -1)
            emask = emask.reshape(-1, k, self.s).expand(b, -1, -1).reshape(b * k, self.s)
            tmask_e = self._invert(emask)[:, None, None, :]
            gvalid = torch.cat([qmask[:, -self.s:].unsqueeze(1), emask.reshape(b, k, self.s)], dim=1)
            gmask = self._invert(rearrange(gvalid, "b r s -> (b s) r"))[:, None, None, :]
        pos_q = torch.arange(t).unsqueeze(0)
        pos_e = torch.arange(t - self.s, t).unsqueeze(0)
        tmask_q = self._invert(qmask)[:, None, None, :]
        alone = torch.zeros(1, 1, 1, 1)

        for block in m.encoder.block:
            tsa, gsa, ffn = block.layer
            q = tsa(q, attention_mask=tmask_q, position_ids=pos_q)[0]
            # group attention: the query row alone wherever no example has a token
            split = t - self.s if k else t
            head = q[:, :split]
            out = gsa.self_attention(gsa.layer_norm(head).reshape(-1, 1, head.shape[-1]), mask=alone)[0]
            head = head + out.reshape(head.shape)
            if k:
                e = tsa(e, attention_mask=tmask_e, position_ids=pos_e)[0]
                rows = torch.cat([q[:, split:].unsqueeze(1), e.reshape(b, k, self.s, -1)], dim=1)
                rows = rearrange(rows, "b r s d -> (b s) r d")
                rows = rows + gsa.self_attention(gsa.layer_norm(rows), mask=gmask)[0]
                rows = rearrange(rows, "(b s) r d -> b r s d", b=b, s=self.s)
                q = ffn(torch.cat([head, rows[:, 0]], dim=1))
                e = ffn(rows[:, 1:].reshape(b * k, self.s, -1))
            else:
                q = ffn(head)

        h = m.encoder.final_layer_norm(q[:, -self.n_out:])
        preds = rearrange(m.output_patch_embedding(h), "b n (q p) -> b q (n p)", n=self.n_out,
                          q=m.num_quantiles, p=m.chronos_config.output_patch_size)
        return preds, loc_scale

    def unscale(self, preds, loc_scale):
        b, nq, h = preds.shape
        return self.m.instance_norm.inverse(preds.reshape(b, nq * h), loc_scale).reshape(b, nq, h)

    def _chunk(self, length: int, k: int) -> int:
        tokens = length // self.patch + 1 + self.n_out + k * self.s
        return max(1, TOKEN_BUDGET // tokens)

    @torch.no_grad()
    def predict(self, context, ex_ctx=None, ex_fut=None) -> np.ndarray:
        """Raw-scale forecasts at REPORT_QUANTILES, (B, 9, pred_len), in batches
        that fit the token budget."""
        k = 0 if ex_ctx is None else ex_ctx.shape[1]
        chunk = self._chunk(context.shape[1], k)
        shared = ex_ctx is not None and ex_ctx.shape[0] == 1
        out = []
        for lo in range(0, len(context), chunk):
            c = torch.as_tensor(context[lo:lo + chunk], dtype=torch.float32)
            ec = ef = None
            if k:
                sl = slice(None) if shared else slice(lo, lo + chunk)
                ec = torch.as_tensor(ex_ctx[sl], dtype=torch.float32)
                ef = torch.as_tensor(ex_fut[sl], dtype=torch.float32)
            preds, ls = self.forward(c, ec, ef)
            out.append(self.unscale(preds, ls)[:, REPORT_INDEX].numpy())
        return np.concatenate(out)

def check(seed: int = 0, b: int = 3, k: int = 5, history: int = 2032):
    """Compare the parallel forward against Chronos2Model.forward on the padded batch."""
    torch.manual_seed(seed)
    model = load_chronos2()
    rows = Chronos2Rows(model, 96, 64)
    ctx = torch.randn(b, history).cumsum(-1)
    ctx[0, :500] = float("nan")
    ex_ctx = torch.randn(b, k, 96).cumsum(-1) + 3
    ex_fut = torch.randn(b, k, 64).cumsum(-1) + 3
    with torch.no_grad():
        mine = rows.unscale(*rows.forward(ctx, ex_ctx, ex_fut))
        padded, futs, gids = [], [], []
        for i in range(b):
            padded.append(ctx[i])
            futs.append(torch.full((64,), float("nan")))
            gids.append(i)
            for j in range(k):
                r = torch.full((history,), float("nan"))
                r[-96:] = ex_ctx[i, j]
                padded.append(r)
                futs.append(ex_fut[i, j])
                gids.append(i)
        ref = model(context=torch.stack(padded), group_ids=torch.tensor(gids),
                    future_covariates=torch.stack(futs), num_output_patches=4).quantile_preds
        ref = ref[torch.arange(b) * (k + 1)]
        alone = rows.unscale(*rows.forward(ctx))
        ref0 = model(context=ctx, num_output_patches=4).quantile_preds
    print(f"with examples  max |diff| {float((mine - ref).abs().max()):.2e}"
          f"  (mean |value| {float(ref.abs().mean()):.1f})")
    print(f"no examples    max |diff| {float((alone - ref0).abs().max()):.2e}")


if __name__ == "__main__":
    check()
