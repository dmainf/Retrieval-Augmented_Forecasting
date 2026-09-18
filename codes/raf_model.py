"""The RAF model: a frozen Chronos-Bolt backbone fed a retrieval-augmented context.

This is plain RAF: no offset alignment at the joins and no per-window
renormalization. Retrieved windows are concatenated at their raw values, and the
backbone's own instance norm is applied once to the whole concatenated input.
"""
import numpy as np
import torch
from chronos.chronos_bolt import ChronosBoltModelForForecasting

CONCAT_ORDERS = ("nearest-last", "nearest-first")
SEPARATORS = ("none", "nan", "mean", "qmean", "rmean", "local", "zero", "linear")
INJECTIONS = ("concat", "parallel")


def _separator_block(separator: str, sep_len: int, left: torch.Tensor,
                     right: torch.Tensor, whole: torch.Tensor, query: torch.Tensor,
                     retrieved: torch.Tensor) -> torch.Tensor:
    """One separator block of shape (B, sep_len).

    left / right are (B, 1): the values on either side of this particular gap,
    so a separator can bridge them. whole / query / retrieved are the pieces a
    constant separator can take its level from.
    """
    data = whole
    b = data.shape[0]
    if separator == "nan":
        return torch.full((b, sep_len), float("nan"), dtype=data.dtype, device=data.device)
    if separator == "zero":
        return torch.zeros((b, sep_len), dtype=data.dtype, device=data.device)
    if separator == "mean":       # level of the whole input (depends on k)
        return whole.nanmean(dim=-1, keepdim=True).expand(b, sep_len)
    if separator == "qmean":      # level of the query only (independent of k)
        return query.nanmean(dim=-1, keepdim=True).expand(b, sep_len)
    if separator == "rmean":      # level of the retrieved windows only
        return retrieved.nanmean(dim=-1, keepdim=True).expand(b, sep_len)
    if separator == "local":      # level of this gap's two neighbours
        return ((left + right) / 2).expand(b, sep_len)
    if separator == "linear":
        t = torch.arange(1, sep_len + 1, dtype=data.dtype, device=data.device) / (sep_len + 1)
        return left + (right - left) * t
    raise ValueError(f"unknown separator {separator}; choose from {SEPARATORS}")


def concat_retrieved(context: torch.Tensor, retrieved: torch.Tensor,
                     order: str = "nearest-last", separator: str = "none",
                     sep_len: int = 0) -> torch.Tensor:
    """context (B, S), retrieved (B, k, W) -> (B, k * (W + sep_len) + S).

    retrieved[:, 0] is the closest match. With "nearest-last" it ends up adjacent
    to the query, which is the usual in-context-learning layout:

        nearest-last   [x_k | y_k] ... [x_1 | y_1] [context]
        nearest-first  [x_1 | y_1] ... [x_k | y_k] [context]

    A separator of sep_len points is inserted after every retrieved window, so
    one also sits directly in front of the query -- the layout text ICL uses:

        [x_k | y_k] SEP ... [x_1 | y_1] SEP [context]

    The variants trade off two things: how visible the gap is as a marker, and
    how much of the join discontinuity it removes.

      nan     unobserved -- excluded from the instance norm, and a full patch of
              it is dropped from attention while still consuming a position
      mean    the mean of the whole input; a flat, observed patch. Note this
              level shifts with k, since more retrieved windows outweigh the query
      qmean   the mean of the query context only, so the level does not move with k
      rmean   the mean of the retrieved windows only
      local   the midpoint of this gap's two neighbouring values
      zero    a raw zero; usually far from the data level, so it marks the gap
              loudly but also widens the jump
      linear  interpolates between the two neighbouring values, removing the
              jump entirely -- the opposite extreme from zero
    """
    if order == "nearest-last":
        retrieved = retrieved.flip(1)
    elif order != "nearest-first":
        raise ValueError(f"unknown concat order {order}; choose from {CONCAT_ORDERS}")

    b, k, w = retrieved.shape
    flat = retrieved.reshape(b, k * w)
    if separator == "none" or sep_len == 0:
        return torch.cat([flat, context], dim=-1)

    whole = torch.cat([flat, context], dim=-1)
    parts = []
    for i in range(k):
        seg = retrieved[:, i, :]
        right = retrieved[:, i + 1, :1] if i + 1 < k else context[:, :1]
        parts.append(seg)
        parts.append(_separator_block(separator, sep_len, seg[:, -1:], right,
                                      whole, context, flat))
    parts.append(context)
    return torch.cat(parts, dim=-1)


def parallel_inputs(context, retrieved, seq_len: int, pred_len: int) -> list[dict]:
    """context (B, H), retrieved (B, k, seq_len + pred_len) -> B Chronos-2 tasks.

    The other container. Instead of splicing the retrieved windows into the time
    axis in front of the query, each one becomes its own row alongside the query
    and the rows are mixed by Chronos-2's group attention. Three things follow,
    and all three are exactly what concatenation gets wrong:

      the history is not touched      the query keeps all H points; adding an
                                      example costs no context budget at all
      each row is scaled on its own    Chronos-2 instance-norms per row, so a
                                      window carrying the level it had months
                                      ago cannot drag the query's loc/scale
      the time index stays honest      every patch carries a "how far back is
                                      this" feature; a retrieved window laid on
                                      the time axis is labelled as recent
                                      history, which is a lie. Here it is not.

    Group attention is position-wise across rows, so the alignment is forced:
    each example's context occupies the LAST seq_len points of the history --
    the same patch positions as the query's own recent past -- and its future
    is handed over as a known future covariate, landing on the patch positions
    the query is predicting. Everything before that is NaN, which Chronos-2
    reads as missing and drops from attention.

    Row order carries no positional signal here, so --concat-order has no effect.
    """
    ctx = np.asarray(context, dtype=np.float32)
    ret = np.asarray(retrieved, dtype=np.float32)
    b, h = ctx.shape
    win = seq_len + pred_len
    if ret.shape[-1] != win:
        raise ValueError(
            f"parallel injection needs whole examples of {win} points "
            f"(seq_len {seq_len} + pred_len {pred_len}), found {ret.shape[-1]}"
        )
    if seq_len > h:
        raise ValueError(
            f"seq_len {seq_len} exceeds the query history {h}; the example "
            "context has nowhere to sit"
        )
    k = ret.shape[1]
    past = np.full((b, k, h), np.nan, dtype=np.float32)
    past[:, :, h - seq_len:] = ret[:, :, :seq_len]
    future = ret[:, :, seq_len:]
    names = [f"ret{j}" for j in range(k)]
    return [
        {
            "target": ctx[i],
            "past_covariates": {n: past[i, j] for j, n in enumerate(names)},
            "future_covariates": {n: future[i, j] for j, n in enumerate(names)},
        }
        for i in range(b)
    ]


QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
CHRONOS2_CONTEXT = 8192
# Chronos-2 falls off a memory cliff on MPS once batch x context passes roughly
# this many points: batch 16 at 8192 runs in 1.5 s, batch 32 never returns.
CHRONOS2_POINT_BUDGET = 131072


class RAFModel:
    """Frozen backbone that forecasts from a (possibly augmented) context.

    Two backbones are supported and expose the same interface. Chronos-Bolt is
    a 2048-point encoder-decoder; Chronos-2 is a 120M encoder-only model whose
    context reaches 8192, which matters here because retrieved windows and real
    history compete for exactly that budget. Neither was pretrained on inputs
    made of concatenated examples -- Chronos-2 only widens the container, it
    does not change what the container is.
    """

    def __init__(self, chronos_model: str, pred_len: int, device: torch.device):
        self.is_chronos2 = "chronos-2" in chronos_model
        self.pred_len = pred_len
        self.device = device
        self.quantiles = list(QUANTILES)

        if self.is_chronos2:
            # MPS falls off a memory cliff on this model and is no faster than CPU
            # once the batch is chunked, so run it on CPU unless asked otherwise
            if device.type == "mps":
                device = torch.device("cpu")
                self.device = device
            from chronos import Chronos2Pipeline
            self.pipeline = Chronos2Pipeline.from_pretrained(
                chronos_model, device_map=str(device))
            self.context_length = CHRONOS2_CONTEXT
            self.patch_size = self.pipeline.model.patch.patch_size
            return

        self.model = ChronosBoltModelForForecasting.from_pretrained(chronos_model)
        self.model.to(device).eval()
        self.model.requires_grad_(False)

        config = self.model.chronos_config
        if pred_len > config.prediction_length:
            raise ValueError(
                f"pred_len {pred_len} exceeds the backbone prediction_length "
                f"{config.prediction_length}; multi-step rollout is not implemented."
            )
        self.context_length = config.context_length
        self.patch_size = config.input_patch_size
        self.quantiles = [float(q) for q in config.quantiles]

    def max_top_k(self, seq_len: int, win_len: int, sep_len: int = 0) -> int:
        """How many windows fit in front of the context without truncation."""
        return max(0, (self.context_length - seq_len) // (win_len + sep_len))

    @torch.no_grad()
    def predict(self, model_input) -> torch.Tensor:
        """model_input (B, L) -> quantile forecasts (B, n_quantiles, pred_len).

        Every quantile the backbone produces is returned at raw scale; picking a
        point forecast is left to whoever reads the output table.
        """
        x = torch.as_tensor(model_input, dtype=torch.float32)
        if x.shape[-1] > self.context_length:
            raise ValueError(
                f"input length {x.shape[-1]} exceeds the backbone context_length "
                f"{self.context_length}; it would be silently truncated."
            )
        if self.is_chronos2:
            rows = [row.numpy() for row in x]
            chunk = max(1, CHRONOS2_POINT_BUDGET // max(1, x.shape[-1]))
            out = []
            for lo in range(0, len(rows), chunk):
                preds, _ = self.pipeline.predict_quantiles(
                    rows[lo:lo + chunk], prediction_length=self.pred_len,
                    quantile_levels=self.quantiles, batch_size=chunk,
                    context_length=self.context_length)
                # each element is (n_variates=1, pred_len, n_quantiles)
                out += [torch.as_tensor(np.asarray(q)).squeeze(0) for q in preds]
            return torch.stack(out).permute(0, 2, 1)

        out = self.model(context=x.to(self.device))
        return out.quantile_preds[:, :, :self.pred_len]

    @torch.no_grad()
    def predict_parallel(self, context, retrieved, seq_len: int) -> torch.Tensor:
        """Forecast with the examples placed beside the history, not in front of it.

        context (B, H), retrieved (B, k, seq_len + pred_len) -> (B, n_quantiles,
        pred_len). Chronos-2 only: Chronos-Bolt is univariate and has no slot
        for a parallel series.

        Each task costs (1 + k) rows of H points rather than one row of
        k * (win_len + sep_len) + H, so the point budget is spent per row and
        the chunking has to count rows, not queries.
        """
        if not self.is_chronos2:
            raise ValueError(
                "parallel injection needs the chronos-2 backbone; chronos-bolt "
                "is univariate and takes no covariates"
            )
        ctx = torch.as_tensor(context, dtype=torch.float32).cpu().numpy()
        ret = torch.as_tensor(retrieved, dtype=torch.float32).cpu().numpy()
        if ctx.shape[-1] > self.context_length:
            raise ValueError(
                f"history length {ctx.shape[-1]} exceeds the backbone "
                f"context_length {self.context_length}"
            )
        tasks = parallel_inputs(ctx, ret, seq_len, self.pred_len)
        rows = 1 + ret.shape[1]
        chunk = max(1, CHRONOS2_POINT_BUDGET // max(1, ctx.shape[-1] * rows))
        out = []
        for lo in range(0, len(tasks), chunk):
            preds, _ = self.pipeline.predict_quantiles(
                tasks[lo:lo + chunk], prediction_length=self.pred_len,
                quantile_levels=self.quantiles, batch_size=chunk * rows,
                context_length=self.context_length)
            # each element is (n_variates=1, pred_len, n_quantiles)
            out += [torch.as_tensor(np.asarray(q)).squeeze(0) for q in preds]
        return torch.stack(out).permute(0, 2, 1)
