"""Helpers for loading the Chronos-Bolt backbone."""
import torch

from models.chronos import (
    ChronosBoltModelForForecasting,
    ChronosBoltModelForForecastingWithRetrieval,
)


def load_plain(chronos_model: str):
    return ChronosBoltModelForForecasting.from_pretrained(chronos_model)


def load_with_retrieval(chronos_model: str, augment: str):
    return ChronosBoltModelForForecastingWithRetrieval.from_pretrained(
        chronos_model, augment=augment
    )


def model_pred_len(model) -> int:
    return model.chronos_config.prediction_length


def median_point(quantile_preds: torch.Tensor, model) -> torch.Tensor:
    """(B, num_q, pred_len) -> (B, pred_len) using the median quantile."""
    quantiles = torch.as_tensor(model.chronos_config.quantiles)
    idx = int(torch.argmin((quantiles - 0.5).abs()).item())
    return quantile_preds[:, idx, :]
