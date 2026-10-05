"""Forecast every query window with a frozen Chronos-2 and save predictions with ground truth.

  --fusion   parallel (examples as covariate rows) or concat (examples spliced in front)
  --select   none, a task-level set (random / clg / pclg) or an instance-level
             choice (l2 / oracle / truth); see selection.py

The output table is long: one row per channel, query window and horizon step h,
with the timestamp of that step, the true value and one column per quantile
(q0.1 ... q0.9). evaluate.py turns it into MSE, MAE and QL.

  python3 codes/run.py --select none
  python3 codes/run.py --select random --top-k 128 --seed 0
  python3 codes/run.py --select random --top-k 128 --seed 0 --eval-split test
  python3 codes/run.py --select l2 --top-k 8 --fusion concat
"""
import argparse
import json
import os
import re
import zlib

import numpy as np
import pandas as pd

from chronos2 import Chronos2Rows, concat_context, load_chronos2
from dataset import TimeSeriesData
from selection import INSTANCE_LEVEL, TASK_LEVEL, ablate, instance_level, task_level

DATASET = "ETTh1"
EVAL_SPLIT = "val"
# coprime with 24 and 168 so forecast origins cover every hour of the day and week;
# 16 hit only 00/08/16h and biased val MSE by +2.6% against all origins
EVAL_STRIDE = 17
SEQ_LEN = 96
PRED_LEN = 64
HISTORY = 2032
TOP_K = 8
# coprime with 24 and 168 like EVAL_STRIDE, so examples exist at every hour-of-week phase;
# on val (l2, K=8) 5 / 7 / 8 / 11 were within noise, and 7 divides 168
DB_STRIDE = 11
SEED = 0
SEP_LEN = 16
ETT_MONTHS = (12, 4, 4)
TRAIN_FRAC = 0.7
TEST_FRAC = 0.2
MODEL_NAME = "amazon/chronos-2"
# Apple GPU; falls back to CPU where MPS is unavailable
DEVICE = "mps"
REPORT_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TOKEN_BUDGET = 24576
# CLG / Parallel-CLG (clg.py). Epochs and batch follow Zhang et al.; the latent is K value-space
# windows instead of embedding-space prefix tokens, so the official AdamW lr 1e-3 does not carry
# over and CLG_LR (Adam, on the standardized latent) is still to be checked on the train loss
CLG_LATENT = 8
CLG_EPOCHS = 10
CLG_LR = 0.05
CLG_BATCH = 64
CLG_MICRO_BATCH = 8
# the paper states 32 swap iterations; the official code iterates to convergence (up to 1000)
CLG_SWAPS = 1000
# Parallel-CLG ranks windows by gradient alignment: on val it beat mean matching on MAE / QL /
# interval width at every K, which mean matching (and clg with cosine) did not; clg stays as published
DEFAULT_MATCH = {"pclg": "cosine", "clg": "mean"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root-path", default="./Datasets/_parquet")
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--channels", default="", help="comma-separated; empty = every column")
    p.add_argument("--eval-split", default=EVAL_SPLIT, choices=["val", "test"])
    p.add_argument("--eval-stride", default=EVAL_STRIDE, type=int)
    p.add_argument("--seq-len", default=SEQ_LEN, type=int, help="context length of an example")
    p.add_argument("--pred-len", default=PRED_LEN, type=int)
    p.add_argument("--history", default=HISTORY, type=int, help="query history fed to the model")
    p.add_argument("--fusion", default="parallel", choices=["parallel", "concat"])
    p.add_argument("--select", default="none", choices=("none",) + TASK_LEVEL + INSTANCE_LEVEL)
    p.add_argument("--top-k", default=TOP_K, type=int)
    p.add_argument("--ablation", default="none", choices=["none", "shuffle-future"])
    p.add_argument("--db-stride", default=DB_STRIDE, type=int,
                   help="stride for cutting the windows examples are drawn from")
    p.add_argument("--seed", default=SEED, type=int)
    p.add_argument("--threads", default=0, type=int)
    p.add_argument("--output-dir", default="./results")
    p.add_argument("--run-name", default="")
    p.add_argument("--inst-k", default=0, type=int,
                   help="task-level select: also add this many l2 windows per query (hybrid)")
    p.add_argument("--recent", default=0, type=int,
                   help="clg / pclg: target the windows ending in the last RECENT points of train")
    p.add_argument("--match", default=None, choices=["mean", "cosine"],
                   help="clg / pclg: match the mean gradient, or rank windows by alignment with it "
                        "(default: cosine for pclg, mean for clg)")
    p.add_argument("--reverse-match", action="store_true",
                   help="clg / pclg: pick the set farthest from the mean gradient (diagnostic)")
    p.add_argument("--overwrite", action="store_true", help="replace an existing result")
    a = p.parse_args()
    if a.select == "none" and a.ablation != "none":
        p.error("--ablation needs examples; pick a --select other than none")
    if (a.reverse_match or a.match) and a.select not in ("clg", "pclg"):
        p.error("--reverse-match / --match apply to clg / pclg only")
    if a.reverse_match and a.match == "cosine":
        p.error("--reverse-match reverses the mean matching only")
    a.match = "mean" if a.reverse_match else (a.match or DEFAULT_MATCH.get(a.select, "mean"))
    if a.recent and a.select not in ("clg", "pclg"):
        p.error("--recent applies to clg / pclg only")
    if a.recent and a.eval_split != "val":
        p.error("--recent needs gradients of the windows just before the split; only val has them")
    if a.inst_k and a.select not in TASK_LEVEL:
        p.error("--inst-k adds l2 windows to a task-level set (random / clg / pclg)")
    return a


def run_name(a) -> str:
    if a.run_name:
        return a.run_name
    stem = f"{a.dataset}_{a.eval_split}_{a.select}"
    if a.select != "none":
        stem += f"_k{a.top_k}_{a.fusion}"
        if a.select in TASK_LEVEL or a.ablation == "shuffle-future":
            stem += f"_s{a.seed}"
        if a.reverse_match:
            stem += "_reverse"
        if a.select in DEFAULT_MATCH and not a.reverse_match and a.match != DEFAULT_MATCH[a.select]:
            stem += f"_{a.match}"
        if a.recent:
            stem += f"_recent{a.recent}"
        if a.inst_k:
            stem += f"_inst{a.inst_k}"
        if a.ablation != "none":
            stem += f"_{a.ablation}"
    tags = [("sl", a.seq_len, SEQ_LEN), ("pl", a.pred_len, PRED_LEN), ("h", a.history, HISTORY),
            ("es", a.eval_stride, EVAL_STRIDE), ("db", a.db_stride, DB_STRIDE)]
    stem += "".join(f"_{t}{v}" for t, v, default in tags if v != default)
    if a.channels:
        stem += "_ch" + "+".join(re.sub(r"[^0-9A-Za-z]", "", c) for c in a.channels.split(","))
    return stem


def channel_rng(seed: int, channel: str):
    """One generator per channel, so a channel draws the same examples whichever
    other channels run alongside it."""
    return np.random.default_rng([seed, zlib.crc32(channel.encode())])


def load_grads(a, channel: str, n_pool: int) -> np.ndarray:
    """Curriculum latent gradients written by clg.py, flattened to (n_pool, d)."""
    from clg import grads_path
    path = grads_path(a.output_dir, a.dataset, a.select, a.seed, channel)
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing; run python3 codes/clg.py --method {a.select} first")
    f = np.load(path)
    meta = json.loads(str(f["meta"]))
    for key in ("seq_len", "pred_len", "history", "db_stride"):
        if meta[key] != getattr(a, key):
            raise SystemExit(f"{path} was made with {key}={meta[key]}, not {getattr(a, key)}")
    grads = f["grads"]
    if len(grads) != n_pool:
        raise SystemExit(f"{path} has {len(grads)} windows, the pool has {n_pool}")
    return grads.reshape(n_pool, -1)


def main():
    args = parse_args()
    stem = os.path.join(args.output_dir, run_name(args))
    if os.path.exists(f"{stem}.parquet") and not args.overwrite:
        raise SystemExit(f"{stem}.parquet exists; pass --overwrite or --run-name")
    data = TimeSeriesData(args.root_path, args.dataset, args.seq_len, args.pred_len, args.channels,
                          ett_months=ETT_MONTHS, train_frac=TRAIN_FRAC, test_frac=TEST_FRAC)
    rows = Chronos2Rows(load_chronos2(MODEL_NAME, DEVICE, args.threads), args.seq_len,
                        args.pred_len, REPORT_QUANTILES, TOKEN_BUDGET)
    if args.history > rows.context_length:
        raise SystemExit(f"--history {args.history} exceeds the model's "
                         f"{rows.context_length}-point context")
    s = args.seq_len

    tables, logs, sets = [], [], {}
    for ch, name in enumerate(data.channels):
        rng = channel_rng(args.seed, name)
        query = data.windows(args.eval_split, ch, args.eval_stride)
        starts = np.arange(len(query), dtype=np.int64) * args.eval_stride
        history = data.history(args.eval_split, ch, starts, args.history)
        futures = query[:, s:]

        ex = None
        pool = data.windows("train", ch, args.db_stride)
        if args.select in TASK_LEVEL:
            grads = load_grads(args, name, len(pool)) if args.select != "random" else None
            target_idx = None
            if args.recent:
                ends = np.arange(len(pool)) * args.db_stride + data.win_len + data.borders[0][0]
                target_idx = np.where(ends > data.borders[1][0] - args.recent)[0]
            ex, idx = task_level(args.select, pool, args.top_k, rng, grads, CLG_SWAPS,
                                 args.reverse_match, args.match,
                                 -(-data.win_len // args.db_stride), target_idx)
            sets[name] = idx.tolist()
            if args.inst_k:
                limit = data.past_limit(args.eval_split, starts, args.db_stride)
                inst, iidx = instance_level("l2", pool, query, limit, args.inst_k, s, exclude=idx)
                ex = np.concatenate([np.broadcast_to(ex, (len(query),) + ex.shape[1:]), inst], 1)
                logs.append(pd.DataFrame({"channel": name,
                                          "window": np.repeat(np.arange(len(query)), args.inst_k),
                                          "rank": np.tile(np.arange(args.inst_k), len(query)),
                                          "db_index": iidx.reshape(-1)}))
        elif args.select in INSTANCE_LEVEL:
            limit = data.past_limit(args.eval_split, starts, args.db_stride)
            ex, idx = instance_level(args.select, pool, query, limit, args.top_k, s)
            if idx is not None:
                logs.append(pd.DataFrame({"channel": name,
                                          "window": np.repeat(np.arange(len(query)), args.top_k),
                                          "rank": np.tile(np.arange(args.top_k), len(query)),
                                          "db_index": idx.reshape(-1)}))
        if ex is not None:
            ex = ablate(ex, args.ablation, s, rng)

        if ex is None:
            out = rows.predict(history)
        elif args.fusion == "parallel":
            out = rows.predict(history, ex[..., :s], ex[..., s:])
        else:
            out = rows.predict(concat_context(history, ex, SEP_LEN, rows.context_length))
        n, p = len(query), args.pred_len
        dates = data.future_dates(args.eval_split, starts)
        table = pd.DataFrame({"channel": name,
                              "window": np.repeat(np.arange(n), p),
                              "start": np.repeat(starts, p),
                              "forecast_start": np.repeat(dates[:, 0], p),
                              "h": np.tile(np.arange(p), n),
                              "date": dates.reshape(-1),
                              "true": futures.reshape(-1)})
        for i, q in enumerate(REPORT_QUANTILES):
            table[f"q{q:g}"] = out[:, i].reshape(-1)
        tables.append(table)
        print(f"  [{ch + 1}/{data.n_channels}] {name}: {len(query)} queries", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    pd.concat(tables, ignore_index=True).to_parquet(f"{stem}.parquet", compression="zstd", index=False)
    with open(f"{stem}_args.json", "w") as f:
        json.dump({**vars(args), "sep_len": SEP_LEN, "ett_months": ETT_MONTHS,
                   "train_frac": TRAIN_FRAC, "test_frac": TEST_FRAC, "model": MODEL_NAME,
                   "quantiles": REPORT_QUANTILES, "task_sets": sets,
                   "channels_used": data.channels}, f, indent=1)
    if logs:
        pd.concat(logs, ignore_index=True).to_parquet(f"{stem}_selected.parquet", index=False)
    elif os.path.exists(f"{stem}_selected.parquet"):
        os.remove(f"{stem}_selected.parquet")
    print(f"saved {stem}.parquet")


if __name__ == "__main__":
    main()
