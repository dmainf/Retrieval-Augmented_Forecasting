"""Forecast every query window with a frozen Chronos-2 and save predictions with ground truth.

  --fusion   parallel (examples as covariate rows) or concat (examples spliced in front)
  --select   none, a task-level set (random / kmeans) or an instance-level
             choice (l2 / oracle / recent / placebo); see selection.py

The output table holds the 9 quantile forecasts and the true future per query,
which evaluate.py turns into MSE and QL.

  python3 codes/run.py --select none
  python3 codes/run.py --select random --top-k 128 --seed 0 --eval-split val
  python3 codes/run.py --select l2 --top-k 8 --fusion concat
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from chronos2 import REPORT_QUANTILES, Chronos2Rows, concat_context, load_chronos2
from dataset import TimeSeriesData
from selection import INSTANCE_LEVEL, TASK_LEVEL, ablate, instance_level, task_level


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root-path", default="./Datasets/_parquet")
    p.add_argument("--dataset", default="ETTh1")
    p.add_argument("--channels", default="", help="comma-separated; empty = every column")
    p.add_argument("--eval-split", default="test", choices=["val", "test"])
    p.add_argument("--eval-stride", default=16, type=int)
    p.add_argument("--seq-len", default=96, type=int, help="context length of an example")
    p.add_argument("--pred-len", default=64, type=int)
    p.add_argument("--history", default=2032, type=int, help="query history fed to the model")
    p.add_argument("--fusion", default="parallel", choices=["parallel", "concat"])
    p.add_argument("--select", default="none", choices=("none",) + TASK_LEVEL + INSTANCE_LEVEL)
    p.add_argument("--top-k", default=8, type=int)
    p.add_argument("--ablation", default="none", choices=["none", "shuffle-future", "truth-future"])
    p.add_argument("--db-stride", default=8, type=int,
                   help="stride for cutting the windows examples are drawn from")
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--threads", default=0, type=int)
    p.add_argument("--output-dir", default="./results")
    p.add_argument("--run-name", default="")
    return p.parse_args()


def run_name(a) -> str:
    if a.run_name:
        return a.run_name
    stem = f"{a.dataset}_{a.eval_split}_{a.select}"
    if a.select != "none":
        stem += f"_k{a.top_k}_{a.fusion}"
        if a.select in ("random", "placebo"):
            stem += f"_s{a.seed}"
        if a.ablation != "none":
            stem += f"_{a.ablation}"
    return stem


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    data = TimeSeriesData(args.root_path, args.dataset, args.seq_len, args.pred_len, args.channels)
    rows = Chronos2Rows(load_chronos2(threads=args.threads), args.seq_len, args.pred_len)
    s = args.seq_len

    preds, trues, meta, logs, sets = [], [], [], [], {}
    for ch, name in enumerate(data.channels):
        query = data.windows(args.eval_split, ch, args.eval_stride)
        starts = np.arange(len(query), dtype=np.int64) * args.eval_stride
        history = data.history(args.eval_split, ch, starts, args.history)
        futures = query[:, s:]

        ex = None
        if args.select in TASK_LEVEL:
            pool = data.windows("train", ch, args.db_stride)
            ex, idx = task_level(args.select, pool, args.top_k, args.seed, s, rng)
            sets[name] = idx.tolist()
        elif args.select in INSTANCE_LEVEL:
            ex, idx = instance_level(args.select, data, args.eval_split, ch, starts, query,
                                     args.top_k, args.db_stride, rng)
            logs.append(pd.DataFrame({"channel": name,
                                      "window": np.repeat(np.arange(len(query)), args.top_k),
                                      "rank": np.tile(np.arange(args.top_k), len(query)),
                                      "db_index": idx.reshape(-1)}))
        if ex is not None:
            ex = ablate(ex, args.ablation, s, futures, rng)

        if ex is None:
            out = rows.predict(history)
        elif args.fusion == "parallel":
            out = rows.predict(history, ex[..., :s], ex[..., s:])
        else:
            out = rows.predict(concat_context(history, ex))
        preds.append(out)
        trues.append(futures)
        meta.append(pd.DataFrame({"channel": name, "window": np.arange(len(query)),
                                  "start": starts}))
        print(f"  [{ch + 1}/{data.n_channels}] {name}: {len(query)} queries", flush=True)

    preds = np.concatenate(preds)
    frames = [pd.concat(meta, ignore_index=True)]
    for i, q in enumerate(REPORT_QUANTILES):
        frames.append(pd.DataFrame(preds[:, i], columns=[f"pred_q{q:g}_{t}" for t in range(args.pred_len)]))
    frames.append(pd.DataFrame(np.concatenate(trues), columns=[f"true_{t}" for t in range(args.pred_len)]))

    os.makedirs(args.output_dir, exist_ok=True)
    stem = os.path.join(args.output_dir, run_name(args))
    pd.concat(frames, axis=1).to_parquet(f"{stem}.parquet", compression="zstd", index=False)
    with open(f"{stem}_args.json", "w") as f:
        json.dump({**vars(args), "task_sets": sets, "channels_used": data.channels}, f, indent=1)
    if logs:
        pd.concat(logs, ignore_index=True).to_parquet(f"{stem}_selected.parquet", index=False)
    print(f"saved {stem}.parquet")


if __name__ == "__main__":
    main()
