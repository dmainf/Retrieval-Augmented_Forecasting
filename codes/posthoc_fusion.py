"""Post-hoc fusion of the retrieved futures with a plain Chronos-2 forecast.

The control for the parallel container: the model never sees the examples.
Chronos-2 forecasts from the history alone, the K retrieved futures are reduced
to their empirical quantiles, and the two are blended quantile by quantile,

    fused_q = (1 - alpha) * chronos_q + alpha * quantile_q(retrieved futures)

with one alpha shared by every query, chosen on val by MSE and then applied to
test. If this matches the parallel container fed the same examples, the
container adds nothing beyond averaging them in.

  python3 codes/posthoc_fusion.py --val-base results/fuse_val_k0.parquet \
      --val-retrieval results/fuse_val_par_future_k8_retrieval.parquet \
      --test-base results/full_k0.parquet \
      --test-retrieval results/sweep_par_future_k8_retrieval.parquet
"""
import argparse

import numpy as np
import pandas as pd

from dataset import TimeSeriesData
from evaluate import channel_normalized_mse, channel_normalized_wql


def retrieved_futures(data, log: pd.DataFrame, table: pd.DataFrame, stride: int) -> np.ndarray:
    """(n_rows, K, pred_len) retrieved futures aligned with the rows of table."""
    k = int(log["rank"].max()) + 1
    out = np.full((len(table), k, data.pred_len), np.nan, dtype=np.float32)
    row_of = {(c, w): i for i, (c, w) in enumerate(zip(table["channel"], table["window"]))}
    for name, part in log.groupby("channel"):
        db = data.all_windows(data.channels.index(name), stride)[:, data.seq_len:]
        rows = np.array([row_of[(name, w)] for w in part["window"]])
        out[rows, part["rank"].to_numpy()] = db[part["db_index"].to_numpy()]
    if np.isnan(out).any():
        raise ValueError("some rows have no retrieved futures")
    return out


def fuse(table: pd.DataFrame, futures: np.ndarray, alpha: float) -> pd.DataFrame:
    out = table.copy()
    levels = sorted({float(c.split("_")[1][1:]) for c in table.columns if c.startswith("pred_q")})
    h = futures.shape[-1]
    for q in levels:
        cols = [f"pred_q{q:g}_{t}" for t in range(h)]
        ret_q = np.quantile(futures, q, axis=1)
        out[cols] = (1 - alpha) * table[cols].to_numpy() + alpha * ret_q
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root-path", default="./Datasets/_parquet")
    p.add_argument("--dataset", default="ETTh1")
    p.add_argument("--retrieval-stride", default=8, type=int)
    p.add_argument("--val-base", required=True)
    p.add_argument("--val-retrieval", required=True)
    p.add_argument("--test-base", required=True)
    p.add_argument("--test-retrieval", required=True)
    p.add_argument("--grid", default=0.05, type=float)
    args = p.parse_args()

    data = TimeSeriesData(args.root_path, args.dataset, 96, 64)
    splits = {}
    for split, base, log in [("val", args.val_base, args.val_retrieval),
                             ("test", args.test_base, args.test_retrieval)]:
        t = pd.read_parquet(base)
        splits[split] = (t, retrieved_futures(data, pd.read_parquet(log), t, args.retrieval_stride))

    alphas = np.round(np.arange(0, 1 + 1e-9, args.grid), 4)
    val_t, val_f = splits["val"]
    val_mse = [channel_normalized_mse(fuse(val_t, val_f, a)) for a in alphas]
    best = float(alphas[int(np.argmin(val_mse))])
    test_t, test_f = splits["test"]
    for label, a in [("no fusion", 0.0), (f"alpha={best:g} (val)", best), ("retrieval only", 1.0)]:
        v = fuse(val_t, val_f, a)
        t = fuse(test_t, test_f, a)
        print(f"{label:22s} val MSE {channel_normalized_mse(v):.4f} QL {channel_normalized_wql(v):.4f}"
              f" | test MSE {channel_normalized_mse(t):.4f} QL {channel_normalized_wql(t):.4f}")
    print("val MSE by alpha:", " ".join(f"{a:g}:{m:.4f}" for a, m in zip(alphas, val_mse)))


if __name__ == "__main__":
    main()
