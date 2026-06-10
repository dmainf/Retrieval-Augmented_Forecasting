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
