"""MSE, MAE and quantile loss over a prediction table written by run.py.

Main scores: errors divided by each channel's train std, as in the long-term
forecasting literature (PatchTST etc. standardize every channel with train
statistics), then averaged over channels. The train std is recomputed from the
dataset named in the run's _args.json.

e-scores: errors divided by the std of the channel's own evaluated ground truth
instead. That weights the channels by how hard the evaluated period is rather
than by their train scale, and can change an improvement ratio noticeably.

The table is long (one row per channel, window and horizon step, columns true
and q0.1 ... q0.9). Older wide tables (pred_q<q>_<t> / true_<t>) are converted.

  python3 codes/evaluate.py results/ETTh1_val_none.parquet [more.parquet ...]
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

from dataset import TimeSeriesData

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS = ["MSE", "MAE", "QL"]


def to_long(table: pd.DataFrame) -> pd.DataFrame:
    """Convert a wide table (pred_q<q>_<t>, true_<t>) to the long layout."""
    if "true" in table.columns:
        return table
    steps = sorted(int(c.rsplit("_", 1)[1]) for c in table.columns if c.startswith("true_"))
    levels = sorted({c.split("_")[1][1:] for c in table.columns if c.startswith("pred_q")}, key=float)
    n, p = len(table), len(steps)
    out = pd.DataFrame({"channel": np.repeat(table["channel"].to_numpy(), p),
                        "h": np.tile(steps, n),
                        "true": table[[f"true_{t}" for t in steps]].to_numpy().reshape(-1)})
    for q in levels:
        out[f"q{q}"] = table[[f"pred_q{q}_{t}" for t in steps]].to_numpy().reshape(-1)
    return out


def quantile_columns(table: pd.DataFrame) -> list:
    return sorted([c for c in table.columns if re.fullmatch(r"q\d*\.?\d+", c)], key=lambda c: float(c[1:]))


def train_std(args_path: str) -> dict:
    """Per-channel std (ddof 0) of the train split of the run's dataset."""
    with open(args_path) as f:
        a = json.load(f)
    if "ett_months" not in a:
        from run import ETT_MONTHS, TEST_FRAC, TRAIN_FRAC
        a.update(ett_months=ETT_MONTHS, train_frac=TRAIN_FRAC, test_frac=TEST_FRAC)
    root = a["root_path"]
    if not os.path.isabs(root) and not os.path.isdir(root):
        root = os.path.join(REPO, root)
    data = TimeSeriesData(root, a["dataset"], a["seq_len"], a["pred_len"],
                          ett_months=tuple(a["ett_months"]), train_frac=a["train_frac"],
                          test_frac=a["test_frac"])
    lo, hi = data.borders[0][0], data.borders[1][0]
    return dict(zip(data.channels, data.values[lo:hi].std(0).astype(np.float64)))


def channel_scores(table: pd.DataFrame, scale: dict = None, median: str = "q0.5") -> pd.DataFrame:
    """MSE / MAE of the median and mean pinball loss per channel, each divided by
    scale[channel] (squared for MSE), or by the std of the channel's own truth
    when scale is None."""
    if median not in table.columns:
        raise ValueError(f"no {median} column in the table")
    cols = quantile_columns(table)
    rows = {}
    for name, g in table.groupby("channel", sort=False):
        y = g["true"].to_numpy(np.float64)
        err = g[median].to_numpy(np.float64) - y
        ql = 0.0
        for c in cols:
            q, diff = float(c[1:]), y - g[c].to_numpy(np.float64)
            ql += np.maximum(q * diff, (q - 1.0) * diff).mean()
        sd = y.std() if scale is None else scale[name]
        rows[name] = {"MSE": (err ** 2).mean() / sd ** 2, "MAE": np.abs(err).mean() / sd,
                      "QL": ql / len(cols) / sd}
    return pd.DataFrame.from_dict(rows, orient="index")


def evaluate(path: str) -> dict:
    """Main scores (train std) and e-scores (evaluated-truth std), averaged over channels."""
    table = to_long(pd.read_parquet(path))
    main = channel_scores(table, train_std(path[:-len(".parquet")] + "_args.json")).mean()
    own = channel_scores(table).mean()
    return {**main.to_dict(), **{f"e{k}": v for k, v in own.items()}}


def main():
    paths = [p for p in sys.argv[1:] if not p.endswith("_selected.parquet")]
    if not paths:
        raise SystemExit(__doc__)
    names = METRICS + [f"e{m}" for m in METRICS]
    print("".join(f"{n:>8s}" for n in names) + "  path")
    for path in paths:
        s = evaluate(path)
        print("".join(f"{s[n]:8.4f}" for n in names) + f"  {path}")


if __name__ == "__main__":
    main()
