"""How close did the retrieved futures actually land to the answer?

Every ablation so far has measured retrieval through the backbone, which mixes
the quality of the retrieval with whatever the backbone does with it. This
measures the retrieval alone, in the only space that matters for the mechanism
the destruction tests found: the futures.

For each query it takes the windows the run actually retrieved, and reports

    d_y = || y_retrieved - y_true ||  /  median over the database of || y_db - y_true ||

so 1.0 means "no better than picking at random" and 0 means "found the answer".
The normalization is per query, which is what makes channels and regimes
comparable; it is the same quantity `diagnose_retrieval.py` plots against K.

Read it together with the end-to-end MSE of the same runs. The pair
(d_y reached, gain obtained) is one point on the curve that decides whether a
partly-good retriever buys a partly-good result -- which is the question the
oracle on its own cannot answer, because its selection rule needs the answer.

  python3 codes/future_distance.py par_abl_raw sweep_par_future_k8 ...
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from dataset import SPLITS, TimeSeriesData


def _database(data: TimeSeriesData, channel: int, args: dict) -> np.ndarray:
    """The same window array the run searched over."""
    if args["retrieval_split"] == "past":
        return data.all_windows(channel, args["retrieval_stride"])
    return data.windows(args["retrieval_split"], channel, args["retrieval_stride"])


def run_distances(run: str, results_dir: str = "./results") -> pd.DataFrame:
    """(channel, window, rank, d_y) for one run, normalized per query."""
    with open(os.path.join(results_dir, f"{run}_args.json")) as f:
        args = json.load(f)
    log = pd.read_parquet(os.path.join(results_dir, f"{run}_retrieval.parquet"))
    table = pd.read_parquet(os.path.join(results_dir, f"{run}.parquet"))

    data = TimeSeriesData(args["root_path"], args["dataset"], args["seq_len"],
                          args["pred_len"], args["channels"])
    true_cols = sorted([c for c in table.columns if c.startswith("true_")],
                       key=lambda c: int(c.rsplit("_", 1)[1]))
    fut = slice(args["seq_len"], args["seq_len"] + args["pred_len"])

    out = []
    for channel, name in enumerate(data.channels):
        db_future = _database(data, channel, args)[:, fut].astype(np.float64)
        rows = table[table["channel"] == name]
        y_true = rows[true_cols].to_numpy(np.float64)                  # (n, H)

        # distance from each query's future to every database future, so the
        # per-query median can normalize away how volatile that stretch was
        all_d = np.linalg.norm(y_true[:, None, :] - db_future[None, :, :], axis=-1)
        scale = np.median(all_d, axis=1)                               # (n,)

        hit = log[log["channel"] == name]
        q = hit["window"].to_numpy()
        d = np.linalg.norm(db_future[hit["db_index"].to_numpy()] - y_true[q], axis=-1)
        out.append(pd.DataFrame({"run": run, "channel": name, "window": q,
                                 "rank": hit["rank"].to_numpy(),
                                 "d_y": d / np.maximum(scale[q], 1e-12)}))
    return pd.concat(out, ignore_index=True)


def main():
    runs = sys.argv[1:]
    if not runs:
        raise SystemExit(__doc__)
    print(f"{'d_y mean':>9s}{'best':>8s}{'worst':>8s}  run")
    for run in runs:
        d = run_distances(run)
        per_query = d.groupby(["channel", "window"])["d_y"]
        print(f"{d['d_y'].mean():9.3f}{per_query.min().mean():8.3f}"
              f"{per_query.max().mean():8.3f}  {run}")


if __name__ == "__main__":
    main()
