"""Task-level demonstrations: one fixed example set per channel, shared by every query.

The set is drawn from the channel's train windows (cut at --pool-stride) and
handed to Chronos-2 as covariate rows beside each query's history, the same
parallel container as predict.py --inject parallel. Nothing is retrieved per
query. Output tables match predict.py, so evaluate.py reads them unchanged.

  python3 codes/taskset.py --method random --top-k 32 --seed 0
  python3 codes/taskset.py --method file --set-file results/clg_sets.npz --top-k 32
  python3 codes/taskset.py --method latent --set-file results/clg_latent.npz
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from dataset import TimeSeriesData
from sparse_rows import REPORT_QUANTILES, SparseRows, load_chronos2


def kmeans_set(pool: np.ndarray, k: int, seed: int) -> np.ndarray:
    """The pool window nearest each of k centroids, on per-window z-scored shape."""
    from sklearn.cluster import KMeans
    x = (pool - pool[:, :96].mean(1, keepdims=True)) / (pool[:, :96].std(1, keepdims=True) + 1e-8)
    km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(x)
    d = ((x[:, None, :] - km.cluster_centers_[None]) ** 2).sum(-1)
    picked = []
    for c in range(k):
        for i in np.argsort(d[:, c]):
            if i not in picked:
                picked.append(int(i))
                break
    return np.array(picked)


def select(method: str, pool: np.ndarray, k: int, seed: int, name: str, sets) -> np.ndarray:
    if method == "random":
        return np.random.default_rng(seed).choice(len(pool), size=k, replace=False)
    if method == "kmeans":
        return kmeans_set(pool, k, seed)
    if method == "file":
        idx = np.asarray(sets[name])
        if len(idx) < k:
            raise ValueError(f"{name}: set file holds {len(idx)} examples, asked for {k}")
        return idx[:k]
    raise ValueError(method)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root-path", default="./Datasets/_parquet")
    p.add_argument("--dataset", default="ETTh1")
    p.add_argument("--channels", default="")
    p.add_argument("--eval-split", default="test", choices=["train", "val", "test"])
    p.add_argument("--eval-stride", default=16, type=int)
    p.add_argument("--seq-len", default=96, type=int)
    p.add_argument("--pred-len", default=64, type=int)
    p.add_argument("--history", default=2032, type=int)
    p.add_argument("--pool-stride", default=8, type=int,
                   help="stride for cutting the train windows the set is drawn from")
    p.add_argument("--method", default="random",
                   choices=["none", "random", "kmeans", "file", "latent"])
    p.add_argument("--set-file", default="",
                   help="npz keyed by channel: pool indices (file) or latent rows (latent)")
    p.add_argument("--top-k", default=8, type=int)
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--ablation", default="none", choices=["none", "shuffle-future"],
                   help="shuffle-future permutes the futures among the set's examples")
    p.add_argument("--chunk", default=16, type=int)
    p.add_argument("--threads", default=0, type=int)
    p.add_argument("--output-dir", default="./results")
    p.add_argument("--run-name", default="")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    data = TimeSeriesData(args.root_path, args.dataset, args.seq_len, args.pred_len,
                          args.channels)
    model = load_chronos2(threads=args.threads)
    sr = SparseRows(model, args.history, args.seq_len, args.pred_len)
    sets = np.load(args.set_file) if args.set_file else None
    rng = np.random.default_rng(args.seed + 1)

    pred_rows, true_rows, meta_rows, chosen = [], [], [], {}
    for ch, name in enumerate(data.channels):
        query = data.windows(args.eval_split, ch, args.eval_stride)
        starts = np.arange(len(query), dtype=np.int64) * args.eval_stride
        history = data.history(args.eval_split, ch, starts, args.history)
        futures = query[:, args.seq_len:]

        ex_ctx = ex_fut = None
        if args.method == "latent":
            rows = np.asarray(sets[name], dtype=np.float32)
            ex_ctx, ex_fut = rows[None, :, :args.seq_len], rows[None, :, args.seq_len:]
        elif args.method != "none":
            pool = data.windows("train", ch, args.pool_stride)
            idx = select(args.method, pool, args.top_k, args.seed, name, sets)
            chosen[name] = idx.tolist()
            ex = pool[idx].copy()
            if args.ablation == "shuffle-future":
                ex[:, args.seq_len:] = ex[rng.permutation(len(ex)), args.seq_len:]
            ex_ctx, ex_fut = ex[None, :, :args.seq_len], ex[None, :, args.seq_len:]

        pred_rows.append(sr.predict(history, ex_ctx, ex_fut, chunk=args.chunk))
        true_rows.append(futures)
        meta_rows.append(pd.DataFrame({"channel": name,
                                       "window": np.arange(len(query), dtype=np.int64),
                                       "start": starts}))
        print(f"  [{ch + 1}/{data.n_channels}] {name}: {len(query)} queries", flush=True)

    preds = np.concatenate(pred_rows)
    trues = np.concatenate(true_rows)
    frames = [pd.concat(meta_rows, ignore_index=True)]
    for i, q in enumerate(REPORT_QUANTILES):
        frames.append(pd.DataFrame(preds[:, i, :],
                                   columns=[f"pred_q{q:g}_{t}" for t in range(args.pred_len)]))
    frames.append(pd.DataFrame(trues, columns=[f"true_{t}" for t in range(args.pred_len)]))
    stem = args.run_name or (f"ts_{args.dataset}_{args.eval_split}_{args.method}"
                             f"_k{args.top_k}_s{args.seed}")
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, f"{stem}.parquet")
    pd.concat(frames, axis=1).to_parquet(path, compression="zstd", index=False)
    with open(os.path.join(args.output_dir, f"{stem}_args.json"), "w") as f:
        json.dump({**vars(args), "chosen": chosen}, f, indent=1)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
