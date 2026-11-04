"""Choose the heads for --select attn on a dataset, from the train split only.

Queries are the last --queries forecast origins of each channel's train split
(--eval-stride apart, history inside train). Each query gets --candidates train
windows that end GAP points before its origin. For every group-attention head,
the weights the query row puts on its candidates (averaged over the forecast
columns, as --select attn does) are rank-correlated (Spearman) with how near
each candidate's future is to the query's true future (RMSE after standardizing
both on their own context). The correlation is averaged over the queries, and
the --top heads with the highest correlation inside --layers are proposed.
No val or test data is used.

  python3 codes/heads.py
  python3 codes/heads.py --dataset ETTh2 --head-weights precompute
"""
import argparse
import json
import os
import time

import numpy as np

from chronos2 import Chronos2Rows, load_chronos2
from dataset import TimeSeriesData
from run import DATASET, DEFAULTS, DEVICE, ETT_MONTHS, GAP, MODEL_NAME, TEST_FRAC, TOKEN_BUDGET, TRAIN_FRAC
from selection import candidates, head_weights, precomputed_attention, standardize


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root-path", default="./Datasets/_parquet")
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--channels", default="", help="comma-separated; empty = every column")
    p.add_argument("--queries", default=60, type=int, help="last train origins per channel")
    p.add_argument("--candidates", default=512, type=int)
    p.add_argument("--head-weights", default="exact", choices=["exact", "precompute"],
                   help="exact: query and candidates in one forward (as the default heads were "
                        "chosen); precompute: every row on its own (as --select attn reads them)")
    p.add_argument("--layers", default="10-12", help="1-based layers the heads are chosen from")
    p.add_argument("--top", default=3, type=int)
    for key in ("history", "seq_len", "pred_len", "eval_stride", "db_stride", "seed"):
        p.add_argument("--" + key.replace("_", "-"), default=DEFAULTS[key], type=int)
    p.add_argument("--threads", default=0, type=int)
    p.add_argument("--output-dir", default="./results")
    a = p.parse_args()
    if a.seq_len > a.history:
        p.error("an example's context (--seq-len) cannot be longer than the query history")
    return a


def spearman(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Rank correlation of each row of x (H, M) with y (M,)."""
    rx = x.argsort(-1).argsort(-1).astype(np.float64)
    ry = y.argsort().argsort().astype(np.float64)
    rx -= rx.mean(-1, keepdims=True)
    ry -= ry.mean()
    return (rx @ ry) / np.sqrt((rx ** 2).sum(-1) * (ry ** 2).sum())


def main():
    a = parse_args()
    data = TimeSeriesData(a.root_path, a.dataset, a.seq_len, a.pred_len, a.channels,
                          ett_months=ETT_MONTHS, train_frac=TRAIN_FRAC, test_frac=TEST_FRAC)
    rows = Chronos2Rows(load_chronos2(MODEL_NAME, DEVICE, a.threads), a.seq_len, a.pred_len,
                        [0.5], TOKEN_BUDGET)
    n_layers = len(rows.m.encoder.block)
    heads = [(li, hi) for li in range(n_layers) for hi in range(rows.n_heads)]
    s = a.seq_len
    corr = []
    for ch, name in enumerate(data.channels):
        t0 = time.time()
        windows = data.windows("train", ch, a.eval_stride)
        first = len(windows) - a.queries
        if first < 0:
            raise SystemExit(f"train has only {len(windows)} origins at stride {a.eval_stride}")
        starts = np.arange(first, len(windows), dtype=np.int64) * a.eval_stride
        query = windows[first:]
        history = data.history("train", ch, starts, a.history)
        pool = data.windows("train", ch, a.db_stride)
        limit = data.past_limit("train", starts, a.db_stride, GAP)
        cand, valid = candidates(limit, len(pool), a.candidates, a.seed, name)
        if not valid.all():
            raise SystemExit(f"{name}: an early train origin has fewer than {a.candidates} windows "
                             "before it; lower --candidates or --queries")
        q_fut = standardize(query[:, s:].astype(np.float64), history[:, -s:].astype(np.float64))
        if a.head_weights == "precompute":
            qv, k_self = rows.head_vectors(history, None, heads)
        for i in range(len(query)):
            ex = pool[cand[i]]
            if a.head_weights == "exact":
                att = rows.head_attention(history[i:i + 1], ex[None, :, :s], ex[None, :, s:],
                                          valid[i:i + 1], heads)[0]
            else:
                _, k = rows.head_vectors(ex[:, :s], ex[:, s:], heads)
                att = precomputed_attention(qv[i], k_self[i], k, np.arange(len(ex)), valid[i])
            w = head_weights(att, valid[i])
            e_fut = standardize(ex[:, s:].astype(np.float64), ex[:, :s].astype(np.float64))
            near = -np.sqrt(((e_fut - q_fut[i]) ** 2).mean(-1))
            corr.append(spearman(w, near))
        print(f"  [{ch + 1}/{data.n_channels}] {name}: {len(query)} queries, {time.time() - t0:.0f} s",
              flush=True)

    mean = np.mean(corr, 0).reshape(n_layers, rows.n_heads)
    lo, hi = map(int, a.layers.split("-"))
    inside = [(mean[li, h], li, h) for li in range(lo - 1, hi) for h in range(rows.n_heads)]
    chosen = sorted(inside, reverse=True)[:a.top]
    spec = ",".join(f"L{li + 1}H{h + 1}" for _, li, h in chosen)
    order = np.argsort(-mean, axis=None)[:20]
    print(f"\ntop 20 heads by rank correlation (mean over {len(corr)} train queries)")
    for j in order:
        li, h = divmod(int(j), rows.n_heads)
        print(f"  L{li + 1}H{h + 1:<3d} {mean[li, h]:+.3f}")
    print(f"\nlayers {a.layers}, top {a.top}: --heads {spec}")
    os.makedirs(a.output_dir, exist_ok=True)
    path = os.path.join(a.output_dir, f"heads_{a.dataset}_{a.head_weights}_h{a.history}.json")
    with open(path, "w") as f:
        json.dump({**vars(a), "gap": GAP, "model": MODEL_NAME, "heads": spec,
                   "correlation": mean.round(4).tolist()}, f, indent=1)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
