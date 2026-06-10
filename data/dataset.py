"""Lightweight CSV time-series loader for long-horizon forecasting.

Self-contained (pandas + numpy only, no gluonts/autogluon). Loads a standard
benchmark CSV (first column ``date``, remaining columns are variables), applies
Informer-style train/val/test splits, standardizes per channel using train
statistics, and produces sliding windows of length ``seq_len + pred_len``.

Each channel is treated as an independent univariate series, which matches how
Chronos performs zero-shot forecasting and how the retrieval database is built.
"""
import os
from typing import List, Tuple

import numpy as np
import pandas as pd


def get_borders(name: str, n: int, seq_len: int) -> Tuple[List[int], List[int]]:
    """Return (border1s, border2s) for [train, val, test] splits, Informer-style."""
    name = name.lower()
    if name in ("etth1", "etth2"):
        b1 = [0, 12 * 30 * 24 - seq_len, 12 * 30 * 24 + 4 * 30 * 24 - seq_len]
        b2 = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
    elif name in ("ettm1", "ettm2"):
        b1 = [0, 12 * 30 * 24 * 4 - seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - seq_len]
        b2 = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
    else:
        num_train = int(n * 0.7)
        num_test = int(n * 0.2)
        num_vali = n - num_train - num_test
        b1 = [0, num_train - seq_len, n - num_test - seq_len]
        b2 = [num_train, num_train + num_vali, n]
    return b1, b2


SPLIT_INDEX = {"train": 0, "val": 1, "test": 2}


class TSData:
    """Holds standardized per-channel arrays and produces sliding windows."""

    def __init__(
        self,
        root_path: str,
        data_path: str,
        dataset_name: str,
        seq_len: int,
        pred_len: int,
        features: str = "M",
        target: str = "OT",
        scale: bool = True,
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.win_len = seq_len + pred_len

        df = pd.read_csv(os.path.join(root_path, data_path))
        cols = list(df.columns)
        if "date" in cols:
            cols.remove("date")
        if features == "S":
            cols = [target]
        data = df[cols].values.astype(np.float32)  # (L, C)

        b1, b2 = get_borders(dataset_name, len(data), seq_len)
        self.borders = (b1, b2)
        self.channels = cols

        if scale:
            train_slice = data[b1[0]:b2[0]]
            self.mean = train_slice.mean(0, keepdims=True)
            self.std = train_slice.std(0, keepdims=True) + 1e-8
            data = (data - self.mean) / self.std
        else:
            self.mean = np.zeros((1, data.shape[1]), dtype=np.float32)
            self.std = np.ones((1, data.shape[1]), dtype=np.float32)
        self.data = data  # (L, C) standardized

    def split_array(self, split: str) -> np.ndarray:
        """Return the (L_split, C) standardized slice for the given split."""
        i = SPLIT_INDEX[split]
        b1, b2 = self.borders
        return self.data[b1[i]:b2[i]]

    def make_windows(self, split: str, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Produce sliding windows pooled over all channels.

        Returns
        -------
        windows : (N, win_len) float32  -- each row is [context | future]
        meta    : (N, 2) int32          -- (channel_index, window_start)
        """
        arr = self.split_array(split)  # (L, C)
        L, C = arr.shape
        windows, meta = [], []
        for c in range(C):
            series = arr[:, c]
            for s in range(0, L - self.win_len + 1, stride):
                windows.append(series[s:s + self.win_len])
                meta.append((c, s))
        if not windows:
            raise ValueError(
                f"No windows for split={split}: series length {L} < win_len {self.win_len}."
            )
        return np.asarray(windows, dtype=np.float32), np.asarray(meta, dtype=np.int32)
