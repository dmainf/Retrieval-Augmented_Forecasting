"""Parquet loading, train/val/test splits, and sliding-window extraction.

Every column is treated as an independent univariate series: that is the unit
Chronos forecasts and the unit the retrieval database is built from, so windows
are always cut per channel.

Values are kept at their raw scale. No standardization is applied here -- the
backbone's own instance norm handles scaling.
"""
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

SPLITS = {"train": 0, "val": 1, "test": 2}


def get_borders(name: str, n: int, seq_len: int) -> Tuple[List[int], List[int]]:
    """Return (starts, ends) of the [train, val, test] splits, Informer-style."""
    name = name.lower()
    if name in ("etth1", "etth2"):
        unit = 30 * 24
    elif name in ("ettm1", "ettm2"):
        unit = 30 * 24 * 4
    else:
        n_train, n_test = int(n * 0.7), int(n * 0.2)
        return [0, n_train - seq_len, n - n_test - seq_len], [n_train, n - n_test, n]
    return ([0, 12 * unit - seq_len, 16 * unit - seq_len],
            [12 * unit, 16 * unit, 20 * unit])


class TimeSeriesData:
    """One dataset, sliced into per-channel sliding windows."""

    def __init__(self, root_path: str, dataset: str, seq_len: int, pred_len: int,
                 channels: str = ""):
        df = pd.read_parquet(os.path.join(root_path, f"{dataset}.parquet"))
        cols = [c for c in df.columns if c != "date"]
        if channels:
            wanted = [c.strip() for c in channels.split(",")]
            missing = [c for c in wanted if c not in cols]
            if missing:
                raise ValueError(f"unknown channels {missing}; available: {cols}")
            cols = wanted

        self.values = df[cols].to_numpy(dtype=np.float32)  # (L, C), raw scale
        self.channels = cols
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.win_len = seq_len + pred_len
        self.borders = get_borders(dataset, len(self.values), seq_len)

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    def history(self, split: str, channel: int, starts: np.ndarray,
                length: int) -> np.ndarray:
        """(n, length) of the values immediately preceding each window's future.

        Row i ends where window i's context ends, so it is exactly the past a
        forecaster is allowed to see at that point. Reaching back before the
        split boundary is fine -- at test time every earlier value is observed;
        the split only governs which windows are *evaluated*, and the retrieval
        database is restricted separately. Rows that run off the start of the
        series are left-padded with NaN, which the backbone reads as missing.
        """
        i = SPLITS[split]
        base = self.borders[0][i]
        out = np.full((len(starts), length), np.nan, dtype=np.float32)
        for row, start in enumerate(starts):
            end = base + int(start) + self.seq_len
            lo = max(0, end - length)
            chunk = self.values[lo:end, channel]
            out[row, length - len(chunk):] = chunk
        return out

    def all_windows(self, channel: int, stride: int = 1) -> np.ndarray:
        """(N, win_len) windows over the whole series, window i starting at i*stride.

        Used for a retrieval database that expands up to each query instead of
        being frozen at the train split. Which of these a given query may see is
        the caller's job -- see `past_limit`.
        """
        win = sliding_window_view(self.values[:, channel], self.win_len)[::stride]
        return np.array(win, dtype=np.float32)

    def past_limit(self, split: str, starts: np.ndarray, stride: int) -> np.ndarray:
        """Highest all_windows index each query is allowed to retrieve.

        A window may be used only if it ends at or before the query context
        begins, so nothing it contains overlaps what the query is predicting.
        """
        base = self.borders[0][SPLITS[split]]
        cutoff = base + starts.astype(np.int64)
        return (cutoff - self.win_len) // stride

    def preceding_windows(self, split: str, channel: int, starts: np.ndarray,
                          k: int) -> np.ndarray:
        """(n, k, win_len) blocks lying immediately before each query context.

        Index 0 is the nearest block, matching the retriever's convention that
        column 0 is the closest match. Laid out nearest-last with no separator,
        these reassemble the contiguous history exactly -- which is what makes
        them the control for "same content, different container".
        """
        i = SPLITS[split]
        base = self.borders[0][i]
        out = np.full((len(starts), k, self.win_len), np.nan, dtype=np.float32)
        for row, start in enumerate(starts):
            ctx_start = base + int(start)
            for j in range(k):
                end = ctx_start - j * self.win_len
                lo = max(0, end - self.win_len)
                chunk = self.values[lo:end, channel]
                if len(chunk):
                    out[row, j, self.win_len - len(chunk):] = chunk
        return out

    def windows(self, split: str, channel: int, stride: int = 1) -> np.ndarray:
        """Return (N, seq_len + pred_len) windows; each row is [context | future]."""
        i = SPLITS[split]
        starts, ends = self.borders
        series = self.values[starts[i]:ends[i], channel]
        if len(series) < self.win_len:
            raise ValueError(
                f"split={split} has length {len(series)} < window length {self.win_len}"
            )
        win = sliding_window_view(series, self.win_len)[::stride]
        return np.array(win, dtype=np.float32)  # copy: the view is read-only
