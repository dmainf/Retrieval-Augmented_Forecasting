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
from dataset import SPLITS, TimeSeriesData
from selection import INSTANCE_LEVEL, TASK_LEVEL, ablate, instance_level, task_level

DATASET = "ETTh1"
EVAL_SPLIT = "val"
# coprime with 24 and 168 so forecast origins cover every hour of the day and week;
# 16 hit only 00/08/16h and biased val MSE by +2.6% against all origins
EVAL_STRIDE = 17
# K x seq_len grid on val at history 2032 (2026-10-11): l2 best at K=256 / 384 (-5.2%), random
# K=128 / 384 -4.6%, oracle best at 384; to be re-chosen at history 8192
SEQ_LEN = 384
PRED_LEN = 64
# Chronos-2's full context; 2032 left room for 35 concatenated examples, which parallel rows do not need
HISTORY = 8192
TOP_K = 128
# every train window, so all methods draw from one pool and phase matching has enough windows
# per hour (11 left 31 per hour, fewer than K=128); 5 / 7 / 8 / 11 were within noise for l2 at K=8
DB_STRIDE = 1
# an example must end this many points before the forecast origin, fixed so the usable windows
# stay the same when --seq-len changes
GAP = 96
# run names tag the settings that differ from these (the defaults before 2026-10-19), so the
# names of earlier results keep their meaning
NAME_DEFAULTS = {"sl": 96, "pl": 64, "h": 2032, "es": 17, "db": 11}
ATTN_LAYERS = "0-11"
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
# CLG was set up before the 2026-10-19 defaults; concat needs room for the latent windows
CLG_SEQ_LEN = 96
CLG_HISTORY = 2032
CLG_DB_STRIDE = 11
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
    p.add_argument("--select", default="none",
                   choices=("none",) + TASK_LEVEL + INSTANCE_LEVEL + ("phase", "antiphase"))
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
    p.add_argument("--proxy-lambda", default=1.0, type=float,
                   help="proxy: match futures on y + lambda * (no-reference forecast - y)")
    p.add_argument("--attn-layers", default=ATTN_LAYERS,
                   help="attn: encoder layers (first-last) whose group attention is averaged")
    p.add_argument("--group-edit", default="",
                   help="edit the group attention over the rows: uniform:<first>-<last> or "
                        "bias<beta>:<first>-<last> (encoder blocks)")
    p.add_argument("--phase-period", default=24, type=int,
                   help="phase: examples are drawn at random among the stride-1 train windows "
                        "whose future starts at the query's phase modulo this period")
    p.add_argument("--pool-dataset", default="",
                   help="random / strat: draw the examples from the train windows of every channel "
                        "of this dataset instead of the query's own channel")
    p.add_argument("--overwrite", action="store_true", help="replace an existing result")
    a = p.parse_args()
    if a.select == "none" and a.ablation != "none":
        p.error("--ablation needs examples; pick a --select other than none")
    if (a.reverse_match or a.match) and a.select not in ("clg", "pclg"):
        p.error("--reverse-match / --match apply to clg / pclg only")
    if a.reverse_match and a.match == "cosine":
        p.error("--reverse-match reverses the mean matching only")
    if a.pool_dataset and a.select not in ("random", "strat"):
        p.error("--pool-dataset applies to random / strat only")
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
        if a.select == "proxy":
            stem += f"_lam{a.proxy_lambda:g}"
        if a.select == "attn" and a.attn_layers != ATTN_LAYERS:
            stem += f"_L{a.attn_layers}"
        if a.select in DEFAULT_MATCH and not a.reverse_match and a.match != DEFAULT_MATCH[a.select]:
            stem += f"_{a.match}"
        if a.recent:
            stem += f"_recent{a.recent}"
        if a.inst_k:
            stem += f"_inst{a.inst_k}"
        if a.ablation != "none":
            stem += f"_{a.ablation}"
        if a.pool_dataset:
            stem += f"_pool{a.pool_dataset}"
        if a.select in ("phase", "antiphase"):
            stem += f"_s{a.seed}_P{a.phase_period}"
        if a.group_edit:
            stem += "_ge" + re.sub(r"[^0-9A-Za-z.-]", "", a.group_edit)
    tags = [("sl", a.seq_len), ("pl", a.pred_len), ("h", a.history), ("es", a.eval_stride), ("db", a.db_stride)]
    stem += "".join(f"_{t}{v}" for t, v in tags if v != NAME_DEFAULTS[t] and not (t == "sl" and a.select == "none"))
    if a.channels:
        stem += "_ch" + "+".join(re.sub(r"[^0-9A-Za-z]", "", c) for c in a.channels.split(","))
    return stem


def attn_cache(a, channel: str) -> str:
    """Group attention of every query on every usable pool window, which depends on
    neither K nor the layers averaged, so attn runs share it."""
    b = argparse.Namespace(**{**vars(a), "top_k": 0, "attn_layers": ATTN_LAYERS,
                              "run_name": "", "channels": ""})
    return os.path.join(a.output_dir, "attn", f"{run_name(b)}_{re.sub(r'[^0-9A-Za-z]', '', channel)}.npy")


def base_forecast(a, channel: str, n: int) -> np.ndarray:
    """(n, pred_len) no-reference median forecast of one channel, from the --select none run."""
    b = argparse.Namespace(**{**vars(a), "select": "none", "run_name": "", "channels": ""})
    path = os.path.join(a.output_dir, f"{run_name(b)}.parquet")
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing; run --select none first")
    t = pd.read_parquet(path, columns=["channel", "window", "h", "q0.5"])
    t = t[t["channel"] == channel].sort_values(["window", "h"])
    return t["q0.5"].to_numpy(np.float64).reshape(n, a.pred_len)


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
    os.makedirs(args.output_dir, exist_ok=True)
    data = TimeSeriesData(args.root_path, args.dataset, args.seq_len, args.pred_len, args.channels,
                          ett_months=ETT_MONTHS, train_frac=TRAIN_FRAC, test_frac=TEST_FRAC)
    rows = Chronos2Rows(load_chronos2(MODEL_NAME, DEVICE, args.threads), args.seq_len,
                        args.pred_len, REPORT_QUANTILES, TOKEN_BUDGET)
    if args.group_edit:
        mode, span = args.group_edit.split(":")
        lo, hi = map(int, span.split("-"))
        rows.group_edit = {"layers": set(range(lo, hi + 1)),
                           "mode": "uniform" if mode == "uniform" else "bias",
                           "beta": 0.0 if mode == "uniform" else float(mode[4:])}
    if args.history > rows.context_length:
        raise SystemExit(f"--history {args.history} exceeds the model's "
                         f"{rows.context_length}-point context")
    s = args.seq_len
    foreign = None
    if args.pool_dataset:
        pd_data = TimeSeriesData(args.root_path, args.pool_dataset, args.seq_len, args.pred_len,
                                 ett_months=ETT_MONTHS, train_frac=TRAIN_FRAC, test_frac=TEST_FRAC)
        foreign = np.concatenate([pd_data.windows("train", c, args.db_stride)
                                  for c in range(pd_data.n_channels)])

    tables, logs, sets = [], [], {}
    for ch, name in enumerate(data.channels):
        rng = channel_rng(args.seed, name)
        query = data.windows(args.eval_split, ch, args.eval_stride)
        starts = np.arange(len(query), dtype=np.int64) * args.eval_stride
        history = data.history(args.eval_split, ch, starts, args.history)
        futures = query[:, s:]

        ex = None
        pool = data.windows("train", ch, args.db_stride) if foreign is None else foreign
        if args.select in TASK_LEVEL:
            grads = load_grads(args, name, len(pool)) if args.select in ("clg", "pclg") else None
            target_idx = None
            if args.recent:
                ends = np.arange(len(pool)) * args.db_stride + data.win_len + data.borders[0][0]
                target_idx = np.where(ends > data.borders[1][0] - args.recent)[0]
            ex, idx = task_level(args.select, pool, args.top_k, rng, grads, CLG_SWAPS,
                                 args.reverse_match, args.match,
                                 -(-data.win_len // args.db_stride), target_idx, s)
            sets[name] = idx.tolist()
            if args.inst_k:
                limit = data.past_limit(args.eval_split, starts, args.db_stride, GAP)
                inst, iidx = instance_level("l2", pool, query, limit, args.inst_k, s, exclude=idx)
                ex = np.concatenate([np.broadcast_to(ex, (len(query),) + ex.shape[1:]), inst], 1)
                logs.append(pd.DataFrame({"channel": name,
                                          "window": np.repeat(np.arange(len(query)), args.inst_k),
                                          "rank": np.tile(np.arange(args.inst_k), len(query)),
                                          "db_index": iidx.reshape(-1)}))
        elif args.select in ("phase", "antiphase"):
            pool1 = data.windows("train", ch, 1)
            limit = data.past_limit(args.eval_split, starts, 1, GAP)
            origin = data.borders[0][SPLITS[args.eval_split]] + starts + s - data.borders[0][0]
            shift = 0 if args.select == "phase" else args.phase_period // 2
            j = np.arange(len(pool1))
            idx = np.empty((len(query), args.top_k), dtype=np.int64)
            for i in range(len(query)):
                ok = j[((j + s - origin[i] - shift) % args.phase_period == 0) & (j <= limit[i])]
                idx[i] = rng.choice(ok, size=args.top_k, replace=False)
            ex = pool1[idx]
            logs.append(pd.DataFrame({"channel": name,
                                      "window": np.repeat(np.arange(len(query)), args.top_k),
                                      "rank": np.tile(np.arange(args.top_k), len(query)),
                                      "db_index": idx.reshape(-1)}))
        elif args.select == "attn":
            limit = data.past_limit(args.eval_split, starts, args.db_stride, GAP)
            valid = np.arange(len(pool))[None, :] <= limit[:, None]
            cache = attn_cache(args, name)
            if os.path.exists(cache):
                att = np.load(cache).astype(np.float32)
            else:
                att = rows.row_attention(history, pool[None, :, :s], pool[None, :, s:], valid)
                os.makedirs(os.path.dirname(cache), exist_ok=True)
                np.save(cache, att.astype(np.float16))
            lo, hi = map(int, args.attn_layers.split("-"))
            score = np.where(valid, att[:, lo:hi + 1].mean(1), -np.inf)
            idx = np.argsort(-score, axis=1)[:, :args.top_k]
            ex = pool[idx].copy()
            logs.append(pd.DataFrame({"channel": name,
                                      "window": np.repeat(np.arange(len(query)), args.top_k),
                                      "rank": np.tile(np.arange(args.top_k), len(query)),
                                      "db_index": idx.reshape(-1)}))
        elif args.select in INSTANCE_LEVEL:
            limit = data.past_limit(args.eval_split, starts, args.db_stride, GAP)
            target = None
            if args.select == "proxy":
                f = base_forecast(args, name, len(query))
                target = futures + args.proxy_lambda * (f - futures)
            ex, idx = instance_level(args.select, pool, query, limit, args.top_k, s,
                                     history=history, target=target)
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

    pd.concat(tables, ignore_index=True).to_parquet(f"{stem}.parquet", compression="zstd", index=False)
    with open(f"{stem}_args.json", "w") as f:
        json.dump({**vars(args), "sep_len": SEP_LEN, "gap": GAP, "ett_months": ETT_MONTHS,
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
