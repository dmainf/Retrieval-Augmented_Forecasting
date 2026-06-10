"""``none``: plain frozen Chronos-Bolt, zero-shot, no retrieval."""
import torch

from methods.base import Method
from methods._backbone import load_plain, model_pred_len, median_point


class ChronosBaseMethod(Method):
    name = "none"
    needs_training = False
    needs_retrieval = False

    def build_model(self, device):
        model = load_plain(self.args.chronos_model).to(device)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        self._mpl = model_pred_len(model)
        assert self.args.pred_len <= self._mpl, (
            f"pred_len {self.args.pred_len} > backbone prediction_length {self._mpl}; "
            "multi-step rollout is not implemented."
        )
        return model

    @torch.no_grad()
    def predict(self, model, context, device):
        context = context.to(device).float()
        out = model(context=context)
        point = median_point(out.quantile_preds, model)  # (B, mpl)
        return point[:, : self.args.pred_len]
