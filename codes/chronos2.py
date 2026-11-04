"""Frozen Chronos-2 that reads reference examples, in either of two containers.

parallel  every example becomes its own covariate row beside the query: its
          seq_len context sits on the last seq_len points of the history, its
          pred_len future is a known future covariate on the forecast horizon,
          and group attention mixes the rows column by column. The query
          history is untouched and every row is instance-normed on its own.
concat    the usual RAF layout: examples spliced into the time axis in front of
          the history, nearest last, one missing (NaN) patch after each one.

An example row is NaN except for its own window, so of the (H/16) + 1 + 4
tokens of a padded row only seq_len/16 + 1 + pred_len/16 carry anything; the
rest are masked out of both attentions and never reach a valid token. The
parallel forward therefore runs each example row on those tokens only. It
equals Chronos2Model.forward on the padded batch (see `check`) at a fraction of
the cost, which is what makes hundreds of examples per query affordable.

Columns of the example window are named C<j> (the j-th context patch counted
back from the forecast origin, C1 the last), REG and P<j> (the j-th forecast
patch, P1 the first). `hide` lists the columns where the query row may not read
the example rows in group attention; the examples still read each other and the
query everywhere.
"""
import re

import numpy as np
import torch
from einops import rearrange

GROUP_BYTES = 2 ** 30


def load_chronos2(name: str, device: str = "cpu", threads: int = 0):
    """device "mps" (Apple GPU) falls back to "cpu" where it is unavailable."""
    from chronos import Chronos2Pipeline
    if threads > 0:
        torch.set_num_threads(threads)
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    model = Chronos2Pipeline.from_pretrained(name, device_map=device).model
    model.eval()
    model.requires_grad_(False)
    return model


def concat_context(history: np.ndarray, examples: np.ndarray, gap: int, limit: int) -> np.ndarray:
    """history (B, H), examples (B, K, W), column 0 nearest ->
    (B, K * (W + gap) + H) laid out as [x_K | NaN] ... [x_1 | NaN] [history]."""
    b = history.shape[0]
    ex = examples[:, ::-1]
    k, w = ex.shape[1], ex.shape[2]
    blank = np.full((b, gap), np.nan, dtype=np.float32)
    parts = []
    for j in range(k):
        parts += [ex[:, j], blank]
    out = np.concatenate(parts + [history], axis=1).astype(np.float32)
    if out.shape[1] > limit:
        raise ValueError(f"concatenated input {out.shape[1]} exceeds the {limit}-point "
                         f"context; at most {(limit - history.shape[1]) // (w + gap)} examples fit")
    return out


def parse_heads(spec: str):
    """"L12H1,L11H10" -> [(11, 0), (10, 9)]: 1-based layer and head as in the notes."""
    out = []
    for h in spec.split(","):
        m = re.fullmatch(r"L(\d+)H(\d+)", h.strip())
        if not m:
            raise ValueError(f"bad head {h!r}; write L<layer>H<head>, e.g. L12H1")
        out.append((int(m.group(1)) - 1, int(m.group(2)) - 1))
    return out


class Chronos2Rows:
    """Query rows of length H, each grouped with K example rows of seq_len + pred_len."""

    def __init__(self, model, seq_len: int, pred_len: int, quantiles, token_budget: int):
        cfg = model.chronos_config
        missing = [q for q in quantiles if q not in cfg.quantiles]
        if missing:
            raise ValueError(f"quantiles {missing} are not among the model's {cfg.quantiles}")
        self.patch = cfg.input_patch_size
        if seq_len % self.patch or pred_len % cfg.output_patch_size:
            raise ValueError("seq_len and pred_len must be multiples of the patch size")
        self.m = model
        self.device = next(model.parameters()).device
        self.seq_len, self.pred_len = seq_len, pred_len
        self.n_ctx = seq_len // self.patch
        self.n_out = pred_len // cfg.output_patch_size
        self.s = self.n_ctx + 1 + self.n_out
        self.quantile_index = [cfg.quantiles.index(q) for q in quantiles]
        self.token_budget = token_budget
        self.context_length = cfg.context_length
        self.n_heads = model.encoder.block[0].layer[1].self_attention.n_heads

    def columns(self, spec: str) -> tuple:
        """"P1,C1,REG" -> token indices within the example window."""
        out = set()
        for name in filter(None, (c.strip() for c in spec.split(","))):
            m = re.fullmatch(r"([CP])(\d+)", name)
            if name == "REG":
                out.add(self.n_ctx)
            elif m and m.group(1) == "P" and 1 <= int(m.group(2)) <= self.n_out:
                out.add(self.n_ctx + int(m.group(2)))
            elif m and m.group(1) == "C" and 1 <= int(m.group(2)) <= self.n_ctx:
                out.add(self.n_ctx - int(m.group(2)))
            else:
                raise ValueError(f"bad column {name!r}; use P1..P{self.n_out}, C1..C{self.n_ctx} or REG")
        return tuple(sorted(out))

    def _embed(self, context, future=None):
        """(N, L) context with NaN for missing, (N, pred_len) future or None ->
        tokens [context patches | REG | future patches], their mask, loc/scale."""
        m = self.m
        patched, mask, loc_scale = m._prepare_patched_context(context=context)
        n = len(context)
        tokens = torch.cat([
            m.input_patch_embedding(patched),
            m.shared(torch.full((n, 1), m.config.reg_token_id, device=self.device)),
            m.input_patch_embedding(m._prepare_patched_future(
                future_covariates=future, future_covariates_mask=None, loc_scale=loc_scale,
                num_output_patches=self.n_out, batch_size=n)[0]),
        ], dim=1)
        mask = torch.cat([mask.to(tokens.dtype), torch.ones(n, 1 + self.n_out, device=self.device)], dim=1)
        return tokens, mask, loc_scale

    @staticmethod
    def _invert(mask):
        return (1.0 - mask) * torch.finfo(torch.float32).min

    def forward(self, context, ex_ctx=None, ex_fut=None, row_valid=None, attn=None, hide=()):
        """context (B, L); ex_ctx (B, K, seq_len); ex_fut (B, K, pred_len).

        Returns normalized quantile predictions (B, 21, pred_len) and the query loc/scale.
        row_valid (B, K) bool drops example rows from a query's group.
        attn: a dict whose keys are layer indices; each receives the group attention
        weights (B, n_out, n_heads, 1 + K) of the query row at the forecast columns,
        over itself (index 0) and the example rows.
        hide: token indices of the example window (see `columns`) where the query row
        does not read the example rows.
        """
        m = self.m
        b = context.shape[0]
        q, qmask, loc_scale = self._embed(context)
        t = q.shape[1]
        k = 0 if ex_ctx is None else ex_ctx.shape[1]
        if k:
            e, emask, _ = self._embed(ex_ctx.reshape(-1, self.seq_len), ex_fut.reshape(-1, self.pred_len))
            tmask_e = self._invert(emask)[:, None, None, :]
            evalid = emask.reshape(b, k, self.s)
            if row_valid is not None:
                evalid = evalid * row_valid.to(evalid.dtype).unsqueeze(-1)
            gvalid = torch.cat([qmask[:, -self.s:].unsqueeze(1), evalid], dim=1)
            gmask = self._invert(rearrange(gvalid, "b r s -> (b s) r"))[:, None, None, :]
            if hide:
                r = k + 1
                gmask = gmask.expand(-1, -1, r, -1).clone().view(b, self.s, 1, r, r)
                gmask[:, list(hide), :, 0, 1:] = torch.finfo(torch.float32).min
                gmask = gmask.view(b * self.s, 1, r, r)
        pos_q = torch.arange(t, device=self.device).unsqueeze(0)
        pos_e = torch.arange(t - self.s, t, device=self.device).unsqueeze(0)
        tmask_q = self._invert(qmask)[:, None, None, :]
        alone = torch.zeros(1, 1, 1, 1, device=self.device)

        for li, block in enumerate(m.encoder.block):
            tsa, gsa, ffn = block.layer
            q = tsa(q, attention_mask=tmask_q, position_ids=pos_q)[0]
            split = t - self.s if k else t
            head = q[:, :split]
            if split:
                out = gsa.self_attention(gsa.layer_norm(head).reshape(-1, 1, head.shape[-1]), mask=alone)[0]
                head = head + out.reshape(head.shape)
            if k:
                e = tsa(e, attention_mask=tmask_e, position_ids=pos_e)[0]
                rows = torch.cat([q[:, split:].unsqueeze(1), e.reshape(b, k, self.s, -1)], dim=1)
                rows = rearrange(rows, "b r s d -> (b s) r d")
                normed = gsa.layer_norm(rows)
                if attn is not None and li in attn:
                    attn[li] = self._query_attention(gsa.self_attention, normed, gmask, b)
                rows = rows + self._group(gsa.self_attention, normed, gmask)
                rows = rearrange(rows, "(b s) r d -> b r s d", b=b, s=self.s)
                q = ffn(torch.cat([head, rows[:, 0]], dim=1))
                e = ffn(rows[:, 1:].reshape(b * k, self.s, -1))
            else:
                q = ffn(head)

        h = m.encoder.final_layer_norm(q[:, -self.n_out:])
        preds = rearrange(m.output_patch_embedding(h), "b n (q p) -> b q (n p)", n=self.n_out,
                          q=m.num_quantiles, p=m.chronos_config.output_patch_size)
        return preds, loc_scale

    def _group(self, mha, normed, gmask):
        """Group attention over the rows, a few columns at a time when the weights of all
        columns at once would exceed GROUP_BYTES (hundreds of rows)."""
        n, r = normed.shape[:2]
        step = max(1, int(GROUP_BYTES // (mha.n_heads * r * r * 8)))
        if step >= n:
            return mha(normed, mask=gmask)[0]
        return torch.cat([mha(normed[i:i + step], mask=gmask[i:i + step])[0] for i in range(0, n, step)])

    def _query_attention(self, mha, normed, gmask, b):
        """Group attention weights of the query row (row 0) at the forecast columns
        -> (b, n_out, n_heads, rows)."""
        h = mha.n_heads
        q = rearrange(mha.q(normed[:, :1]), "n 1 (h d) -> n h d", h=h)
        k = rearrange(mha.k(normed), "n r (h d) -> n h r d", h=h)
        w = torch.softmax(torch.einsum("nhd,nhrd->nhr", q, k).float() + gmask[:, :, 0], dim=-1)
        return rearrange(w, "(b s) h r -> b s h r", b=b)[:, -self.n_out:]

    def _alone_states(self, context, future, layers):
        """Rows run on their own (group attention over themselves only), as when no
        example is given. Returns {layer: (N, n_out, d)} input of that layer's group
        attention (after its layer norm) at the forecast columns."""
        x, mask, _ = self._embed(context, future)
        tmask = self._invert(mask)[:, None, None, :]
        pos = torch.arange(x.shape[1], device=self.device).unsqueeze(0)
        alone = torch.zeros(1, 1, 1, 1, device=self.device)
        out = {}
        for li, block in enumerate(self.m.encoder.block[:max(layers) + 1]):
            tsa, gsa, ffn = block.layer
            x = tsa(x, attention_mask=tmask, position_ids=pos)[0]
            normed = gsa.layer_norm(x)
            if li in layers:
                out[li] = normed[:, -self.n_out:]
            x = x + gsa.self_attention(normed.reshape(-1, 1, x.shape[-1]), mask=alone)[0].reshape(x.shape)
            x = ffn(x)
        return out

    @torch.no_grad()
    def head_vectors(self, context, future, heads):
        """query and key vectors of the given (layer, head)s at the forecast columns,
        each row run on its own: two arrays (N, n_heads_used, n_out, head_dim)."""
        layers = sorted({li for li, _ in heads})
        chunk = self._chunk(context.shape[1], 0)
        qs, ks = [], []
        for lo in range(0, len(context), chunk):
            c = torch.as_tensor(context[lo:lo + chunk], dtype=torch.float32, device=self.device)
            f = None if future is None else torch.as_tensor(future[lo:lo + chunk], dtype=torch.float32,
                                                            device=self.device)
            st = self._alone_states(c, f, layers)
            q, k = [], []
            for li, hi in heads:
                mha = self.m.encoder.block[li].layer[1].self_attention
                d = mha.kv_proj_dim
                q.append(mha.q(st[li])[..., hi * d:(hi + 1) * d])
                k.append(mha.k(st[li])[..., hi * d:(hi + 1) * d])
            qs.append(torch.stack(q, 1).float().cpu().numpy())
            ks.append(torch.stack(k, 1).float().cpu().numpy())
        return np.concatenate(qs), np.concatenate(ks)

    @torch.no_grad()
    def head_attention(self, context, ex_ctx, ex_fut, row_valid, heads, hide=()) -> np.ndarray:
        """Group attention of each query row with its own candidate rows in one forward
        (rows read each other), at the given (layer, head)s: (B, n_heads_used, n_out, 1 + K)."""
        layers = sorted({li for li, _ in heads})
        chunk = self._chunk(context.shape[1], ex_ctx.shape[1])
        out = []
        for lo in range(0, len(context), chunk):
            sl = slice(lo, lo + chunk)
            attn = {li: None for li in layers}
            self.forward(torch.as_tensor(context[sl], dtype=torch.float32, device=self.device),
                         torch.as_tensor(ex_ctx[sl], dtype=torch.float32, device=self.device),
                         torch.as_tensor(ex_fut[sl], dtype=torch.float32, device=self.device),
                         torch.as_tensor(row_valid[sl], device=self.device), attn, hide)
            out.append(torch.stack([attn[li][:, :, hi] for li, hi in heads], 1).cpu().numpy())
        return np.concatenate(out)

    def unscale(self, preds, loc_scale):
        b, nq, h = preds.shape
        return self.m.instance_norm.inverse(preds.reshape(b, nq * h), loc_scale).reshape(b, nq, h)

    def _chunk(self, length: int, k: int) -> int:
        tokens = length // self.patch + 1 + self.n_out + k * self.s
        return max(1, self.token_budget // tokens)

    @torch.no_grad()
    def predict(self, context, ex_ctx=None, ex_fut=None, hide=()) -> np.ndarray:
        """Raw-scale forecasts at the chosen quantiles, (B, n_quantiles, pred_len),
        in batches that fit the token budget."""
        k = 0 if ex_ctx is None else ex_ctx.shape[1]
        chunk = self._chunk(context.shape[1], k)
        out = []
        for lo in range(0, len(context), chunk):
            c = torch.as_tensor(context[lo:lo + chunk], dtype=torch.float32, device=self.device)
            ec = ef = None
            if k:
                ec = torch.as_tensor(ex_ctx[lo:lo + chunk], dtype=torch.float32, device=self.device)
                ef = torch.as_tensor(ex_fut[lo:lo + chunk], dtype=torch.float32, device=self.device)
            preds, ls = self.forward(c, ec, ef, hide=hide)
            out.append(self.unscale(preds, ls)[:, self.quantile_index].cpu().numpy())
        return np.concatenate(out)


def check(seed: int = 0, b: int = 3, k: int = 5):
    """Compare the parallel forward against Chronos2Model.forward on the padded batch,
    the precomputed head vectors against the forward's own attention with no examples,
    and hiding every column against no examples at all."""
    from run import DEVICE, HISTORY, MODEL_NAME, PRED_LEN, SEQ_LEN, TOKEN_BUDGET
    s, p, history = SEQ_LEN, PRED_LEN, HISTORY
    torch.manual_seed(seed)
    model = load_chronos2(MODEL_NAME, DEVICE)
    rows = Chronos2Rows(model, s, p, [0.5], TOKEN_BUDGET)
    dev = rows.device
    ctx = torch.randn(b, history).cumsum(-1).to(dev)
    ctx[0, :history // 4] = float("nan")
    ex_ctx = (torch.randn(b, k, s).cumsum(-1) + 3).to(dev)
    ex_fut = (torch.randn(b, k, p).cumsum(-1) + 3).to(dev)
    with torch.no_grad():
        mine = rows.unscale(*rows.forward(ctx, ex_ctx, ex_fut))
        padded, futs, gids = [], [], []
        for i in range(b):
            padded.append(ctx[i])
            futs.append(torch.full((p,), float("nan"), device=dev))
            gids.append(i)
            for j in range(k):
                r = torch.full((history,), float("nan"), device=dev)
                r[-s:] = ex_ctx[i, j]
                padded.append(r)
                futs.append(ex_fut[i, j])
                gids.append(i)
        ref = model(context=torch.stack(padded), group_ids=torch.tensor(gids, device=dev),
                    future_covariates=torch.stack(futs), num_output_patches=rows.n_out).quantile_preds
        ref = ref[torch.arange(b, device=dev) * (k + 1)]
        alone = rows.unscale(*rows.forward(ctx))
        ref0 = model(context=ctx, num_output_patches=rows.n_out).quantile_preds
        every = tuple(range(rows.s))
        hidden = rows.unscale(*rows.forward(ctx, ex_ctx, ex_fut, hide=every))
    heads = [(11, 0), (10, 9)]
    qv, kv = rows.head_vectors(ctx.cpu().numpy(), None, heads)
    att = rows.head_attention(ctx.cpu().numpy(), ex_ctx.cpu().numpy(), ex_fut.cpu().numpy(),
                              np.zeros((b, k), dtype=bool), heads)
    print(f"device {dev}  seq_len {s}  pred_len {p}  history {history}")
    print(f"with examples  max |diff| {float((mine - ref).abs().max()):.2e}"
          f"  (mean |value| {float(ref.abs().mean()):.1f})")
    print(f"no examples    max |diff| {float((alone - ref0).abs().max()):.2e}")
    print(f"hide every column vs no examples  max |diff| {float((hidden - alone).abs().max()):.2e}")
    print(f"head vectors {qv.shape}; self weight with no valid example row "
          f"min {float(att[..., 0].min()):.4f} (should be 1)")


if __name__ == "__main__":
    check()
