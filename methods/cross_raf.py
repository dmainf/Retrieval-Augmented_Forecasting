"""``cross_raf``: Cross-RAG (Lee et al. 2026).

Top-k retrieved windows are fused with the query via a cross-attention branch
(query <- retrieved) and a retrieval self-attention summary branch. The Chronos
backbone is frozen; only the fusion modules are trained.

Note: the vendored Cross-RAG model hardcodes a retrieved-future length of 64, so
``pred_len`` must equal the backbone prediction_length (64 for chronos-bolt-*),
and the encoder MLP expects context length == ``seq_len``.
"""
import os

import torch

from methods.base import Method
from methods._backbone import load_with_retrieval, model_pred_len, median_point

_FUSION_KEYS = (
    "encode_mlp_x", "encode_mlp_y",
    "cross_mha", "self_mha", "ffn_cross", "ffn_self",
    "gate_layer", "mha", "ffn",
)


class CrossRAFMethod(Method):
    name = "cross_raf"
    needs_training = True
    needs_retrieval = True

    def build_model(self, device):
        # the vendored model reads these at construction time
        os.environ["INPUT_LEN"] = str(self.args.seq_len)
        os.environ["LAMBDA"] = str(self.args.mix_lambda)

        model = load_with_retrieval(self.args.chronos_model, self.args.augment_mode)

        self._mpl = model_pred_len(model)
        assert self.args.pred_len == self._mpl, (
            f"cross_raf requires pred_len == backbone prediction_length ({self._mpl}); "
            f"got {self.args.pred_len}."
        )

        # Explicitly initialise every fusion module. Current transformers'
        # from_pretrained leaves modules absent from the checkpoint on empty
        # (NaN) memory, so we must init all of them, not just the MLPs.
        init_layers = [getattr(model, n) for n in _FUSION_KEYS if hasattr(model, n)]
        if init_layers:
            model.init_extra_weights(init_layers)

        # freeze backbone, train only fusion modules
        for p in model.parameters():
            p.requires_grad = False
        for name, p in model.named_parameters():
            if any(k in name for k in _FUSION_KEYS):
                p.requires_grad = True

        return model.to(device)

    def _fuse_inputs(self, context, device, exclude_self):
        ctx = context.to(device).float()
        idx, dist = self.retriever.search(ctx, top_k=self.args.top_k, exclude_self=exclude_self)
        retrieved = self.retriever.gather(idx).to(device).float()  # (B, k, win_len)
        return ctx, retrieved, dist.to(device).float()

    def compute_loss(self, model, context, target, device):
        ctx, retrieved, dist = self._fuse_inputs(context, device, exclude_self=True)
        out = model(
            context=ctx,
            target=target.to(device).float(),
            retrieved_seq=retrieved,
            distances=dist,
        )
        return out.loss

    @torch.no_grad()
    def predict(self, model, context, device):
        ctx, retrieved, dist = self._fuse_inputs(context, device, exclude_self=False)
        out = model(context=ctx, retrieved_seq=retrieved, distances=dist)
        point = median_point(out.quantile_preds, model)
        return point[:, : self.args.pred_len]
