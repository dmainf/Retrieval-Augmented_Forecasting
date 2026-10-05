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


def get_borders(name: str, n: int, seq_len: int, ett_months: Tuple[int, int, int],
                train_frac: float, test_frac: float) -> Tuple[List[int], List[int]]:
    """Return (starts, ends) of the [train, val, test] splits, Informer-style: ETT in
    months of 30 days, every other dataset by fraction of its length."""
    name = name.lower()
    if name in ("etth1", "etth2"):
        unit = 30 * 24
    elif name in ("ettm1", "ettm2"):
        unit = 30 * 24 * 4
    else:
        n_train, n_test = int(n * train_frac), int(n * test_frac)
        return [0, n_train - seq_len, n - n_test - seq_len], [n_train, n - n_test, n]
    tr, va, te = (m * unit for m in ett_months)
    return [0, tr - seq_len, tr + va - seq_len], [tr, tr + va, tr + va + te]


class TimeSeriesData:
    """One dataset, sliced into per-channel sliding windows."""

    def __init__(self, root_path: str, dataset: str, seq_len: int, pred_len: int,
                 channels: str = "", *, ett_months: Tuple[int, int, int],
                 train_frac: float, test_frac: float):
        df = pd.read_parquet(os.path.join(root_path, f"{dataset}.parquet"))
        if "date" not in df.columns:
            raise ValueError(f"{dataset} has no date column")
        self.dates = pd.to_datetime(df["date"]).to_numpy()
        cols = [c for c in df.columns if c != "date"]
        if channels:
            wanted = [c.strip() for c in channels.split(",")]
            missing = [c for c in wanted if c not in cols]
            if missing:
                raise ValueError(f"unknown channels {missing}; available: {cols}")
            cols = wanted

        self.values = df[cols].to_numpy(dtype=np.float32)  # (L, C), raw scale
        nan = np.isnan(self.values).sum(0)
        if nan.any():
            bad = {c: int(n) for c, n in zip(cols, nan) if n}
            raise ValueError(f"{dataset} has NaN values (count per channel): {bad}")
        self.channels = cols
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.win_len = seq_len + pred_len
        self.borders = get_borders(dataset, len(self.values), seq_len,
                                   ett_months, train_frac, test_frac)

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    def history(self, split: str, channel: int, starts: np.ndarray,
                length: int) -> np.ndarray:
        """(n, length) of the values immediately preceding each window's future.

        Row i ends where window i's context ends, so it is exactly the past a
        forecaster is allowed to see at that point. Reaching back before the
        split boundary is fine -- at test time every earlier value is observed;
        the split only governs which windows are *evaluated*; examples come from
        the train windows only. Rows that run off the start of the
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

    def future_dates(self, split: str, starts: np.ndarray) -> np.ndarray:
        """(n, pred_len) timestamps of each window's future points."""
        base = self.borders[0][SPLITS[split]]
        first = base + starts.astype(np.int64) + self.seq_len
        return self.dates[first[:, None] + np.arange(self.pred_len)[None, :]]

    def past_limit(self, split: str, starts: np.ndarray, stride: int) -> np.ndarray:
        """Highest index into windows("train", channel, stride) each query may use.

        A window may be used only if it ends at or before the query context
        begins. Ending before the forecast origin would already rule out leaking
        the answer, but a window whose future falls inside the query context would
        only echo the query's own history back as an example's outcome; the
        stricter cutoff keeps every example a past episode disjoint from the query.
        """
        base = self.borders[0][SPLITS[split]]
        cutoff = base + starts.astype(np.int64)
        return (cutoff - self.win_len - self.borders[0][SPLITS["train"]]) // stride

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
