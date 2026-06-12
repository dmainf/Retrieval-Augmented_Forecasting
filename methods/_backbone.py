"""Helpers for loading the Chronos-Bolt backbone."""
import torch
from transformers import logging as hf_logging

from models.chronos import (
    ChronosBoltModelForForecasting,
    ChronosBoltModelForForecastingWithRetrieval,
)

# silence the verbose per-tensor "newly initialized" dump; we print a summary
hf_logging.set_verbosity_error()


def _report_new_weights(missing_keys):
    if not missing_keys:
        return
    groups = {}
    for k in missing_keys:
        top = k.split(".")[0]
        groups[top] = groups.get(top, 0) + 1
    summary = ", ".join(f"{name}({n})" for name, n in sorted(groups.items()))
    print(f"[backbone] newly initialized (not in checkpoint): {summary}")


def load_plain(chronos_model: str):
    return ChronosBoltModelForForecasting.from_pretrained(chronos_model)


def load_with_retrieval(chronos_model: str, augment: str):
    model, info = ChronosBoltModelForForecastingWithRetrieval.from_pretrained(
        chronos_model, augment=augment, output_loading_info=True
    )
    _report_new_weights(info.get("missing_keys", []))
    return model


def model_pred_len(model) -> int:
    return model.chronos_config.prediction_length


def median_point(quantile_preds: torch.Tensor, model) -> torch.Tensor:
    """(B, num_q, pred_len) -> (B, pred_len) using the median quantile."""
    quantiles = torch.as_tensor(model.chronos_config.quantiles)
    idx = int(torch.argmin((quantiles - 0.5).abs()).item())
    return quantile_preds[:, idx, :]
