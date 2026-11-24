"""Forecast every query window with a frozen Chronos-2 and save predictions with ground truth.

  --fusion   parallel (examples as covariate rows) or concat (examples spliced in front)
  --select   none, random, a match (ctx=ctx, fc=fc, fc=fut, fut=fc, fut=fut), attn or state;
             see selection.py
  --hide     columns where the query row does not read the example rows (parallel only)

Every setting but --select and --heads defaults to the proposed method, so --select attn
with the dataset's selection heads (heads.py) is the proposal and the other selections are
compared under the same conditions.

The output table is long: one row per channel, query window and horizon step h,
with the timestamp of that step, the true value and one column per quantile
(q0.1 ... q0.9). evaluate.py turns it into MSE, MAE and QL.

  python3 codes/run.py                      # no reference
  python3 codes/run.py --select attn --heads L12H1,L11H10,L10H12     # the proposed method on ETTh1
  python3 codes/run.py --select attn --heads L12H1,L11H10,L10H12 --hide ""
  python3 codes/run.py --select random
  python3 codes/run.py --select ctx=ctx --top-k 8 --fusion concat

--select-only picks the examples but does not forecast: it writes only <name>_selected.parquet
and <name>_args.json, which evaluate.py --picks compares with the examples nearest to the true
future without running the model.
"""
import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd

from chronos2 import Chronos2Rows, concat_context, load_chronos2, parse_heads
from dataset import TimeSeriesData
from selection import (JOINT_SCORES, MASKED_NEEDS, RAW_SCORES, MATCHES, METRICS, STATES, VALUE_SCORES, candidates, head_score,
                       joint_projection, masked_score, match, out_projection, precomputed_attention, state_distance, top_k)

# settings an argument can change: the proposed method but for --select; a run is named by
# the ones that differ from these
DEFAULTS = {
    "select": "none",
    "fusion": "parallel",
    "top_k": 256,
    "candidates": 2048,
    "hide": "P1",
    "metric": "l2",
    "match_norm": "context",
    # no default: --select attn needs --heads (per dataset, from heads.py on val)
    "heads": "",
    "state": "",
    "layers": "",
    "head_weights": "precompute",
    "select_hide": False,
    "value_score": "none",
    # 512 / 64 as in TS-RAG, Cross-RAG, TimeRAF and Align-RAG; 8192 (the model's limit) is the
    # long-history setting, 2032 the 10/28 report. K and candidates were chosen at 2032 and kept;
    # an example's context is as long as the query history, as in RAFT and TS-RAG
    "history": 512,
    "seq_len": 512,
    "pred_len": 64,
    # coprime with 24 and 168 so forecast origins cover every hour of the day and week
    "eval_stride": 17,
    "db_stride": 1,
    "seed": 0,
}
DATASET = "ETTh1"
# an example must end this many points before the forecast origin, fixed so the usable windows
# stay the same when --seq-len changes
GAP = 96
ETT_MONTHS = (12, 4, 4)
TRAIN_FRAC = 0.7
TEST_FRAC = 0.2
MODEL_NAME = "amazon/chronos-2"
# Apple GPU; falls back to CPU where MPS is unavailable
DEVICE = "mps"
# vectors of the train windows, saved per dataset, channel and head or layer (see pool_features)
CACHE_DIR = "./results/cache"
REPORT_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MEDIAN = REPORT_QUANTILES.index(0.5)
TOKEN_BUDGET = 24576
HISTORY, SEQ_LEN, PRED_LEN = DEFAULTS["history"], DEFAULTS["seq_len"], DEFAULTS["pred_len"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root-path", default="./Datasets/_parquet")
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--channels", default="", help="comma-separated; empty = every column")
    p.add_argument("--eval-split", default="val", choices=["val", "test"])
    p.add_argument("--select", choices=("none", "random") + MATCHES + ("attn", "state"))
    p.add_argument("--fusion", choices=["parallel", "concat"])
    p.add_argument("--top-k", type=int)
    p.add_argument("--candidates", type=int, help="drawn per query at random; 0 = every usable window")
    p.add_argument("--hide", help="e.g. P1 or P1,C1 (P: forecast patch, C: context patch from the "
                                  "origin back, REG); empty = read everywhere")
    p.add_argument("--metric", choices=METRICS)
    p.add_argument("--match-norm", choices=["context", "raw"])
    p.add_argument("--heads", help="needed by --select attn, e.g. L12H1,L11H10,L10H12 (1-based layer and "
                                   "head, as heads.py proposes); L12H1*0.5,L12H7*0.3 weights the average")
    p.add_argument("--head-weights", choices=["precompute", "exact"])
    p.add_argument("--select-hide", action="store_const", const=True,
                   help="exact: apply --hide in the forward that picks the examples too")
    p.add_argument("--value-score", choices=VALUE_SCORES,
                   help="attn, precompute: score with the candidates run with their future hidden (selection.py)")
    p.add_argument("--state", choices=STATES, help="needed by --select state (selection.py)")
    p.add_argument("--layers", help="--state out or in: 1-based layers, e.g. L10,L11,L12")
    p.add_argument("--select-only", action="store_true", help="pick the examples without forecasting")
    p.add_argument("--history", type=int, help="query history fed to the model")
    p.add_argument("--seq-len", type=int, help="context length of an example")
    p.add_argument("--pred-len", type=int)
    p.add_argument("--eval-stride", type=int)
    p.add_argument("--db-stride", type=int, help="stride for cutting the train windows")
    p.add_argument("--seed", type=int)
    p.add_argument("--threads", default=0, type=int)
    p.add_argument("--output-dir", default="./results")
    p.add_argument("--run-name", default="")
    p.add_argument("--overwrite", action="store_true", help="replace an existing result")
    a = p.parse_args()
    hide_given = a.hide is not None
    for key, value in DEFAULTS.items():
        if getattr(a, key) is None:
            setattr(a, key, value)
    if a.seq_len > a.history:
        p.error("an example's context (--seq-len) cannot be longer than the query history")
    if a.fusion == "concat":
        if hide_given and a.hide:
            p.error("--hide applies to parallel rows only")
        a.hide = ""
    if a.value_score != "none" and (a.select != "attn" or a.head_weights != "precompute"):
        p.error("--value-score needs --select attn and --head-weights precompute")
    if a.value_score != "none" and ("*" in a.heads or a.db_stride != 1 or a.candidates == 0):
        p.error(f"--value-score {a.value_score} needs unweighted --heads, --db-stride 1 and --candidates > 0")
    if a.select == "attn" and not a.heads:
        p.error("--select attn needs --heads; take them from heads.py (results/heads_<dataset>_val_*.json)")
    if a.select == "state":
        if not a.state:
            p.error("--select state needs --state")
        if a.state in ("out", "in") and not a.layers:
            p.error(f"--state {a.state} needs --layers (heads.py --score {a.state})")
    elif a.state or a.layers:
        p.error("--state and --layers go with --select state")
    if a.select_only and a.select == "none":
        p.error("--select-only needs a --select")
    if a.select_hide and a.head_weights != "exact":
        p.error("--select-hide needs --head-weights exact (precomputed rows never read each other)")
    return a


def run_name(a) -> str:
    if a.run_name:
        return a.run_name
    d = DEFAULTS
    parts = [a.dataset, a.eval_split, a.select.replace("=", "-")]
    if a.select != "none":
        parts += [a.fusion] if a.fusion != d["fusion"] else []
        parts += [f"k{a.top_k}"] if a.top_k != d["top_k"] else []
        parts += [f"m{a.candidates}"] if a.candidates != d["candidates"] else []
        parts += [f"s{a.seed}"] if a.seed != d["seed"] else []
        if a.fusion == "parallel" and a.hide != d["hide"]:
            parts.append("hide" + re.sub(r"[^0-9A-Za-z]", "", a.hide) if a.hide else "nohide")
        if a.select in MATCHES:
            parts += [a.metric] if a.metric != d["metric"] else []
            parts += [a.match_norm] if a.match_norm != d["match_norm"] else []
        if a.select == "attn":
            parts += [a.heads.replace(",", "+").replace("*", "x")]
            parts += [a.head_weights] if a.head_weights != d["head_weights"] else []
            parts += ["selhide"] if a.select_hide else []
            parts += [f"v{a.value_score}"] if a.value_score != d["value_score"] else []
        if a.select == "state":
            parts += [a.state]
            parts += [a.layers.replace(",", "+")] if a.state in ("out", "in") else []
        parts += [f"sl{a.seq_len}"] if a.seq_len != d["seq_len"] else []
        parts += [f"db{a.db_stride}"] if a.db_stride != d["db_stride"] else []
    for tag, key in (("pl", "pred_len"), ("h", "history"), ("es", "eval_stride")):
        parts += [f"{tag}{getattr(a, key)}"] if getattr(a, key) != d[key] else []
    parts += ["selonly"] if a.select_only else []
    if a.channels:
        parts.append("ch" + "+".join(re.sub(r"[^0-9A-Za-z]", "", c) for c in a.channels.split(",")))
    return "_".join(parts)


def head_mix(spec: str):
    """"L12H1*0.5,L12H7" -> heads and their weights in the average (1 when not given)."""
    names, w = zip(*[(h.split("*")[0], float(h.split("*")[1]) if "*" in h else 1.0) for h in spec.split(",")])
    return parse_heads(",".join(names)), np.array(w)


def parse_layers(spec: str) -> list:
    """"L10,L11" -> 0-based layer indices."""
    return [int(x.strip().lstrip("Ll")) - 1 for x in spec.split(",")]


def pool_features(a, rows, name, pool, future: bool, heads=(), layers=(), want=("k",)):
    """Arrays want of rows.alone_features over the train windows pool, with their true future or with
    it hidden. Saved per head (q, k, v) or layer (in, out) under CACHE_DIR and read back on later runs
    with the same dataset, channel, lengths and --db-stride."""
    tag = (f"{a.dataset}_{re.sub(r'[^0-9A-Za-z]', '', name)}_s{a.seq_len}_p{a.pred_len}_db{a.db_stride}_"
           + ("fut" if future else "mask"))
    def paths(x):
        if x in ("q", "k", "v"):
            return [os.path.join(CACHE_DIR, f"{tag}_L{h[0] + 1}H{h[1] + 1}_{x}.npy") for h in heads]
        if x in ("in", "out"):
            return [os.path.join(CACHE_DIR, f"{tag}_L{li + 1}_{x}.npy") for li in layers]
        return [os.path.join(CACHE_DIR, f"{tag}_{x}.npy")]
    if not all(os.path.exists(f) for x in want for f in paths(x)):
        s = a.seq_len
        got = rows.alone_features(pool[:, :s], pool[:, s:] if future else None, heads, layers, want)
        os.makedirs(CACHE_DIR, exist_ok=True)
        for x, arr in zip(want, got):
            for j, f in enumerate(paths(x)):
                if not os.path.exists(f):
                    np.save(f, arr[:, j])
        return got
    return tuple(np.stack([np.load(f) for f in paths(x)], 1) for x in want)


def select_head(a, rows, heads, mix, history, pool, cand, valid, hide, name):
    """(n, M) scores of the candidates by the group attention of the chosen heads (with
    --value-score, of the candidates run with their future hidden too; see selection.py)."""
    score = np.empty(cand.shape)
    s = a.seq_len
    if a.head_weights == "precompute":
        if a.value_score != "none":
            if a.value_score in RAW_SCORES:
                proj = lambda v: v
            elif a.value_score in JOINT_SCORES:
                proj = joint_projection(rows.head_out_joint_gram(heads), len(heads))
            else:
                proj = out_projection(rows.head_out_gram(heads))
            q, k_self, v_self = rows.head_vectors(history, None, heads, values=True)
            need = MASKED_NEEDS[a.value_score]
            vecs = {}
            fut = [x for x in ("q", "k", "v") if x in need]
            mask = [x for x in ("q", "k", "v") if x + "m" in need]
            if fut:
                vecs.update(zip(fut, pool_features(a, rows, name, pool, True, heads, want=tuple(fut))))
            if mask:
                vecs.update(zip([x + "m" for x in mask], pool_features(a, rows, name, pool, False, heads,
                                                                      want=tuple(mask))))
            vecs = {x: proj(y) if x.startswith("v") else y for x, y in vecs.items()}
            v_self = proj(v_self)
            jac, cols = [None] * len(cand), None
            if a.value_score in RAW_SCORES:
                jac = rows.head_jacobian(history, heads)
                cols = np.ones(rows.n_out)
                cols[[c - rows.n_ctx - 1 for c in hide if c > rows.n_ctx]] = 0
            for i in range(len(cand)):
                if not valid[i].all():
                    raise ValueError(f"--value-score {a.value_score} needs every candidate valid")
                score[i] = masked_score(a.value_score, q[i], k_self[i], v_self[i], cand[i], vecs, jac[i], cols)
            return score
        q, k_self = rows.head_vectors(history, None, heads)
        k_pool, = pool_features(a, rows, name, pool, True, heads, want=("k",))
        for i in range(len(cand)):
            att = precomputed_attention(q[i], k_self[i], k_pool, cand[i], valid[i])
            score[i] = head_score(att, valid[i], mix)
        return score
    step = max(1, rows.token_budget // ((history.shape[1] // rows.patch + 1 + rows.n_out)
                                        + cand.shape[1] * rows.s))
    for lo in range(0, len(cand), step):
        sl = slice(lo, lo + step)
        ex = pool[cand[sl]]
        att = rows.head_attention(history[sl], ex[..., :s], ex[..., s:], valid[sl], heads,
                                  hide if a.select_hide else ())
        for j, i in enumerate(range(lo, min(lo + step, len(cand)))):
            score[i] = head_score(att[j], valid[i], mix)
    return score


def select_state(a, rows, history, pool, cand, valid, name):
    """(n, M) scores of the candidates by how near their state with the future hidden is to the
    query's (--state, see selection.py): minus the distance."""
    layers = parse_layers(a.layers) if a.state in ("out", "in") else ()
    want = (a.state,)
    x0, = rows.alone_features(history, None, (), layers, want)
    x, = pool_features(a, rows, name, pool, False, (), layers, want)
    score = np.full(cand.shape, -np.inf)
    for i in range(len(cand)):
        c = cand[i][valid[i]]
        score[i, valid[i]] = -state_distance(x0[i], x[c])
    return score


def main():
    args = parse_args()
    stem = os.path.join(args.output_dir, run_name(args))
    done = f"{stem}_selected.parquet" if args.select_only else f"{stem}.parquet"
    if os.path.exists(done) and not args.overwrite:
        raise SystemExit(f"{done} exists; pass --overwrite or --run-name")
    os.makedirs(args.output_dir, exist_ok=True)
    data = TimeSeriesData(args.root_path, args.dataset, args.seq_len, args.pred_len, args.channels,
                          ett_months=ETT_MONTHS, train_frac=TRAIN_FRAC, test_frac=TEST_FRAC)
    rows = Chronos2Rows(load_chronos2(MODEL_NAME, DEVICE, args.threads), args.seq_len,
                        args.pred_len, REPORT_QUANTILES, TOKEN_BUDGET)
    if args.history > rows.context_length:
        raise SystemExit(f"--history {args.history} exceeds the model's {rows.context_length}-point context")
    hide = rows.columns(args.hide)
    heads, mix = head_mix(args.heads) if args.heads else (None, None)
    s = args.seq_len

    tables, logs = [], []
    for ch, name in enumerate(data.channels):
        t0 = time.time()
        query = data.windows(args.eval_split, ch, args.eval_stride)
        win = np.arange(len(query), dtype=np.int64)
        starts = win * args.eval_stride
        history = data.history(args.eval_split, ch, starts, args.history)
        futures = query[:, s:]
        if args.select == "none":
            out = rows.predict(history)
        else:
            pool = data.windows("train", ch, args.db_stride)
            limit = data.past_limit(args.eval_split, starts, args.db_stride, GAP)
            cand, valid = candidates(limit, len(pool), args.candidates, args.seed, name, win)
            if args.select == "random":
                idx, score = cand[:, :args.top_k], np.zeros((len(cand), args.top_k))
                if not valid[:, :args.top_k].all():
                    raise ValueError(f"a query has fewer than k={args.top_k} candidates")
            elif args.select in MATCHES:
                qparts = {"ctx": history[:, -s:], "fut": futures}
                wparts = {"ctx": pool[:, :s], "fut": pool[:, s:]}
                if args.select.startswith("fc"):
                    qparts["fc"] = rows.predict(history)[:, MEDIAN]
                if args.select.endswith("fc"):
                    wparts["fc"] = rows.predict(pool[:, :s])[:, MEDIAN]
                idx, score = match(args.select, qparts, wparts, cand, valid, args.top_k,
                                   args.metric, args.match_norm)
            elif args.select == "attn":
                idx, score = top_k(select_head(args, rows, heads, mix, history, pool, cand, valid, hide, name),
                                   cand, valid, args.top_k)
            else:
                idx, score = top_k(select_state(args, rows, history, pool, cand, valid, name),
                                   cand, valid, args.top_k)
            ex = pool[idx]
            if args.select_only:
                out = None
            elif args.fusion == "parallel":
                out = rows.predict(history, ex[..., :s], ex[..., s:], hide)
            else:
                out = rows.predict(concat_context(history, ex, rows.patch, rows.context_length))
            logs.append(pd.DataFrame({"channel": name,
                                      "window": np.repeat(win, args.top_k),
                                      "rank": np.tile(np.arange(args.top_k), len(query)),
                                      "db_index": idx.reshape(-1),
                                      "score": score.reshape(-1)}))
        n, p = len(query), args.pred_len
        if out is None:
            print(f"  [{ch + 1}/{data.n_channels}] {name}: {n} queries picked, {time.time() - t0:.0f} s", flush=True)
            continue
        dates = data.future_dates(args.eval_split, starts)
        table = pd.DataFrame({"channel": name,
                              "window": np.repeat(win, p),
                              "start": np.repeat(starts, p),
                              "forecast_start": np.repeat(dates[:, 0], p),
                              "h": np.tile(np.arange(p), n),
                              "date": dates.reshape(-1),
                              "true": futures.reshape(-1)})
        for i, q in enumerate(REPORT_QUANTILES):
            table[f"q{q:g}"] = out[:, i].reshape(-1)
        tables.append(table)
        print(f"  [{ch + 1}/{data.n_channels}] {name}: {n} queries, {time.time() - t0:.0f} s", flush=True)

    if tables:
        pd.concat(tables, ignore_index=True).to_parquet(f"{stem}.parquet", compression="zstd", index=False)
    with open(f"{stem}_args.json", "w") as f:
        json.dump({**vars(args), "gap": GAP, "ett_months": ETT_MONTHS, "train_frac": TRAIN_FRAC,
                   "test_frac": TEST_FRAC, "model": MODEL_NAME, "quantiles": REPORT_QUANTILES,
                   "channels_used": data.channels}, f, indent=1)
    if logs:
        pd.concat(logs, ignore_index=True).to_parquet(f"{stem}_selected.parquet", index=False)
    elif os.path.exists(f"{stem}_selected.parquet"):
        os.remove(f"{stem}_selected.parquet")
    print(f"saved {done}")


if __name__ == "__main__":
    main()
