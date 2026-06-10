"""``raf``: context-augmented Retrieval Augmented Forecasting (Tire et al. 2026).

Retrieve the top-k matching windows, offset-align them, and prepend them to the
query context to form an augmented input for the Chronos-Bolt backbone. With
k=1 this is the paper's headline setting; k>1 concatenates several retrieved
motifs before the query (paper Appendix D.3 / Table 12).

    naive    : backbone fully frozen (pure zero-shot, no training)
    advanced : backbone fine-tuned on the retrieval-augmented inputs
"""
import torch

from methods.base import Method
from methods._backbone import load_plain, model_pred_len, median_point


def build_raf_context(context: torch.Tensor, retrieved: torch.Tensor) -> torch.Tensor:
    """Prepend offset-aligned retrieved windows to the query context.

    Each retrieved window ``[retrieved_context | retrieved_future]`` is shifted
    so its last value matches the first value of the piece that follows it
    (continuity at every join). The most similar window (index 0) ends up
    adjacent to the context; less similar ones are chained further to the left.
    Everything is in the standardized space, so no per-piece re-normalization is
    needed here.

    context  (B, seq_len)
    retrieved (B, k, win_len)  -- index 0 is the most similar
    returns  (B, k * win_len + seq_len)
    """
    result = context
    for i in range(retrieved.shape[1]):  # i=0 (most similar) attaches nearest to context
        seg = retrieved[:, i, :]
        offset = result[:, :1] - seg[:, -1:]
        result = torch.cat([seg + offset, result], dim=-1)
    return result


class RAFMethod(Method):
    name = "raf"
    needs_retrieval = True

    def __init__(self, args):
        super().__init__(args)
        self.needs_training = (args.raf_mode == "advanced")

    def build_model(self, device):
        model = load_plain(self.args.chronos_model).to(device)
        if self.needs_training:
            for p in model.parameters():
                p.requires_grad = True
            model.train()
        else:
            for p in model.parameters():
                p.requires_grad = False
            model.eval()
        self._mpl = model_pred_len(model)
        assert self.args.pred_len <= self._mpl, (
            f"pred_len {self.args.pred_len} > backbone prediction_length {self._mpl}."
        )

        # cap k so the augmented input [k*win_len + seq_len] fits the backbone
        # context window (otherwise Chronos silently truncates the far motifs)
        win_len = self.args.seq_len + self.args.pred_len
        ctx_cap = model.chronos_config.context_length
        k_fit = max(1, (ctx_cap - self.args.seq_len) // win_len)
        self._k = min(self.args.top_k, k_fit)
        if self._k < self.args.top_k:
            print(f"[raf] top_k capped {self.args.top_k} -> {self._k} "
                  f"to fit backbone context_length={ctx_cap}.")
        return model

    def _augment(self, context, device):
        ctx = context.to(device).float()
        idx, _ = self.retriever.search(ctx, top_k=self._k, exclude_self=self._training_query)
        retrieved = self.retriever.gather(idx).to(device)  # (B, k, win_len)
        return build_raf_context(ctx, retrieved)

    _training_query = False

    def compute_loss(self, model, context, target, device):
        self._training_query = True
        augmented = self._augment(context, device)
        out = model(context=augmented, target=target.to(device).float())
        return out.loss

    @torch.no_grad()
    def predict(self, model, context, device):
        self._training_query = False
        augmented = self._augment(context, device)
        out = model(context=augmented)
        point = median_point(out.quantile_preds, model)
        return point[:, : self.args.pred_len]
