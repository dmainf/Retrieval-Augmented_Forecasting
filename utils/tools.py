import os
import random

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(gpu: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_results(file_path, model_id, mse, mae):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a") as f:
        f.write(f"{model_id}\n")
        f.write(f"mse:{mse:.4f}, mae:{mae:.4f}\n")


def save_predictions(file_path, preds, trues):
    # tidy long format (one row per forecast step) for later plotting
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    n, length = preds.shape
    window = np.repeat(np.arange(n), length)
    t = np.tile(np.arange(length), n)
    data = np.column_stack([window, t, preds.reshape(-1), trues.reshape(-1)])
    np.savetxt(file_path, data, delimiter=",", header="window,t,pred,true",
               comments="", fmt=["%d", "%d", "%.6f", "%.6f"])
