import os

import numpy as np
import torch


def pick_device(gpu: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def median_from_quantiles(quantile_preds: torch.Tensor, quantiles) -> torch.Tensor:
    """Extract the point (median) forecast from Chronos-Bolt quantile outputs.

    quantile_preds: (B, num_quantiles, pred_len) -> (B, pred_len)
    """
    q = torch.as_tensor(quantiles)
    idx = int(torch.argmin((q - 0.5).abs()).item())
    return quantile_preds[:, idx, :]


class EarlyStopping:
    def __init__(self, patience: int = 3, delta: float = 0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best = None
        self.early_stop = False

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best is None or score > self.best + self.delta:
            self.best = score
            self.counter = 0
            self.save(model, path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save(self, model, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)


def save_results(file_path, model_id, mse, mae):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a") as f:
        f.write(f"{model_id}\n")
        f.write(f"mse:{mse:.4f}, mae:{mae:.4f}\n")
