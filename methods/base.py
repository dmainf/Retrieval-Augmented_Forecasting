"""Common interface shared by all forecasting methods.

train.py / evaluate.py are method-agnostic: they obtain a ``Method`` from the
registry and call this interface. Each method owns its model construction, its
(optional) retrieval, its training loss and its prediction. The data loader only
ever yields plain ``(context, target)`` windows.
"""
from typing import Iterable, Optional

import torch
import torch.nn as nn

from retrieval.retriever import Retriever


class Method:
    name: str = "base"
    needs_training: bool = False
    needs_retrieval: bool = False

    def __init__(self, args):
        self.args = args
        self.retriever: Optional[Retriever] = None

    # --- setup ---------------------------------------------------------------
    def build_model(self, device: torch.device) -> nn.Module:
        raise NotImplementedError

    def set_retriever(self, retriever: Retriever):
        self.retriever = retriever

    def trainable_parameters(self, model: nn.Module) -> Iterable[nn.Parameter]:
        return (p for p in model.parameters() if p.requires_grad)

    # --- core ----------------------------------------------------------------
    def compute_loss(self, model, context, target, device) -> torch.Tensor:
        """Return a scalar loss for one batch (only for trainable methods)."""
        raise NotImplementedError

    def predict(self, model, context, device) -> torch.Tensor:
        """Return point forecasts of shape (B, pred_len) in standardized space."""
        raise NotImplementedError
