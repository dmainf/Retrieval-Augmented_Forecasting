"""Channel-normalized MSE over a prediction table written by predict.py.

Each channel is divided by the standard deviation of its own evaluated ground
truth, then the per-channel MSEs are averaged. That makes channels with wildly
different scales contribute equally, which is what the ETT columns need.

  python3 codes/evaluate.py results/s16_k0.parquet [more.parquet ...]
"""
import sys

import numpy as np
import pandas as pd


def channel_normalized_mse(table: pd.DataFrame, quantile: str = "0.5") -> float:
    pred_cols = sorted([c for c in table.columns if c.startswith(f"pred_q{quantile}_")],
                       key=lambda c: int(c.rsplit("_", 1)[1]))
    true_cols = sorted([c for c in table.columns if c.startswith("true_")],
                       key=lambda c: int(c.rsplit("_", 1)[1]))
    if not pred_cols:
        raise ValueError(f"no pred_q{quantile}_* columns in the table")

    pred = table[pred_cols].to_numpy(np.float64)
    true = table[true_cols].to_numpy(np.float64)
    per_channel = []
    for name in table["channel"].unique():
        m = (table["channel"] == name).to_numpy()
        per_channel.append(((pred[m] - true[m]) ** 2).mean() / true[m].var())
    return float(np.mean(per_channel))


def channel_normalized_wql(table: pd.DataFrame) -> float:
    """Mean pinball loss over all quantiles, divided by each channel's own std.

    MSE uses only the median and therefore measures none of the spread. The
    ablations show retrieval injects a distribution rather than a mapping, so a
    median-only metric misses what it actually supplies.
    """
    levels = sorted({float(c.split("_")[1][1:]) for c in table.columns
                     if c.startswith("pred_q")})
    true_cols = sorted([c for c in table.columns if c.startswith("true_")],
                       key=lambda c: int(c.rsplit("_", 1)[1]))
    true = table[true_cols].to_numpy(np.float64)

    per_channel = []
    for name in table["channel"].unique():
        m = (table["channel"] == name).to_numpy()
        y = true[m]
        loss = 0.0
        for q in levels:
            cols = sorted([c for c in table.columns if c.startswith(f"pred_q{q:g}_")],
                          key=lambda c: int(c.rsplit("_", 1)[1]))
            diff = y - table.loc[m, cols].to_numpy(np.float64)
            loss += np.maximum(q * diff, (q - 1.0) * diff).mean()
        per_channel.append(loss / len(levels) / y.std())
    return float(np.mean(per_channel))


def main():
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit(__doc__)
    print(f"{'MSE':>8s}{'QL':>8s}  path")
    for path in paths:
        if path.endswith("_selected.parquet"):
            continue
        t = pd.read_parquet(path)
        print(f"{channel_normalized_mse(t):8.4f}{channel_normalized_wql(t):8.4f}  {path}")


if __name__ == "__main__":
    main()
