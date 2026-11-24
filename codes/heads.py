"""Choose the heads (or layers) for --select attn and --select state on a dataset, from its train or
val split.

Queries are --queries forecast origins per channel, --eval-stride apart: the last
ones of the train split (--split train, history inside train) or evenly spread
over the val split (--split val, 0 = all of them). Each query gets --candidates train
windows that end GAP points before its origin. For every unit (a group-attention head, or a
layer), a score of the candidates is rank-correlated (Spearman) with how near each candidate's
future is to the query's true future (RMSE after standardizing both on their own context). The
correlation is averaged over the queries, and the --top units with the highest correlation inside
--layers are proposed. No test data is used.

  attn      the attention weight of --select attn (per head)
--head-weights precompute also scores, every row run on its own (selection.py):
  <kind>    every --value-score of --select attn (selection.py head_terms), per head: q, k, v alone
            (q, qpre, qchange, ...), qk (qkpre, qkchange) and qkv (qkv, qkvpre, qkvchange)
  out, in   minus the distance of a layer's group-attention output or input (--state out, in; per layer)
  final     minus the distance of the encoder's final state (--state final; one unit)

  python3 codes/heads.py --head-weights exact
  python3 codes/heads.py --head-weights precompute
"""
import argparse
import json
import os
import time

import numpy as np

from chronos2 import Chronos2Rows, load_chronos2
from dataset import TimeSeriesData
from run import DATASET, DEFAULTS, DEVICE, ETT_MONTHS, GAP, MODEL_NAME, TEST_FRAC, TOKEN_BUDGET, TRAIN_FRAC
from selection import MASKED_ONLY, VALUE_SCORES, candidate_terms, candidates, head_terms, head_weights, out_projection, standardize


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root-path", default="./Datasets/_parquet")
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--channels", default="", help="comma-separated; empty = every column")
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--queries", default=60, type=int, help="origins per channel")
    p.add_argument("--candidates", default=512, type=int)
    p.add_argument("--head-weights", required=True, choices=["exact", "precompute"],
                   help="exact: query and candidates in one forward (how the heads of the 11-18 "
                        "report were chosen); precompute: every row on its own (as --select attn "
                        "with --head-weights precompute reads them)")
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


HEAD_KINDS = ("attn",) + tuple(k for k in VALUE_SCORES if k not in ("none",) + MASKED_ONLY)
LAYER_KINDS = ("out", "in")
TERMS = ("qk", "qkpre", "qkchange", "q", "qpre", "qchange", "k", "kpre", "kchange", "v", "vpre", "vchange")


def pool_scores(rows, history, pool, cand, s, heads, layers):
    """Scores of every query's candidates for every kind, from rows run on their own: {kind: (n, units, M)}.
    The pool windows are run once each (twice: with the future and with it hidden), a chunk at a time."""
    n, m = cand.shape
    hq, hk, hv, xin, xout, xfin = rows.alone_features(history, None, heads, layers,
                                                      ("q", "k", "v", "in", "out", "final"))
    proj = out_projection(rows.head_out_gram(heads))
    hv = proj(hv)
    T = {x: np.empty((n, len(heads), hq.shape[2], m)) for x in TERMS}
    dout, din, dfin = np.empty((n, len(layers), m)), np.empty((n, len(layers), m)), np.empty((n, 1, m))
    dist = lambda x0, x: np.sqrt(((x.astype(np.float64) - x0[None]) ** 2).sum(-1)).sum(-1).T
    need = np.unique(cand)
    step = 1024
    for lo in range(0, len(need), step):
        ids = need[lo:lo + step]
        qf, kf, vf = rows.alone_features(pool[ids, :s], pool[ids, s:], heads, (), ("q", "k", "v"))
        qm, km, vm, sin, sout, sfin = rows.alone_features(pool[ids, :s], None, heads, layers,
                                                          ("q", "k", "v", "in", "out", "final"))
        X = {"q": qf, "k": kf, "v": proj(vf), "qm": qm, "km": km, "vm": proj(vm)}
        for i in range(n):
            at = np.isin(cand[i], ids)
            j = np.searchsorted(ids, cand[i][at])
            t = candidate_terms(hq[i], hk[i], hv[i], {x: y[j] for x, y in X.items()})
            for x in TERMS:
                T[x][i][..., at] = t[x]
            dout[i][:, at] = dist(xout[i], sout[j])
            din[i][:, at] = dist(xin[i], sin[j])
            dfin[i][:, at] = dist(xfin[i], sfin[j])
    selfl = (hq * hk).sum(-1)[..., None].astype(np.float64)
    out = {k: np.empty((n, len(heads), m)) for k in HEAD_KINDS}
    for i in range(n):
        t = {"self": selfl[i], **{x: T[x][i] for x in TERMS}}
        for k in HEAD_KINDS:
            out[k][i] = head_terms("qk" if k == "attn" else k, t)
    return {**out, "out": -dout, "in": -din, "final": -dfin}


def main():
    a = parse_args()
    data = TimeSeriesData(a.root_path, a.dataset, a.seq_len, a.pred_len, a.channels,
                          ett_months=ETT_MONTHS, train_frac=TRAIN_FRAC, test_frac=TEST_FRAC)
    rows = Chronos2Rows(load_chronos2(MODEL_NAME, DEVICE, a.threads), a.seq_len, a.pred_len,
                        [0.5], TOKEN_BUDGET)
    n_layers = len(rows.m.encoder.block)
    heads = [(li, hi) for li in range(n_layers) for hi in range(rows.n_heads)]
    layers = list(range(n_layers))
    kinds = ("attn",) if a.head_weights == "exact" else HEAD_KINDS + LAYER_KINDS + ("final",)
    s = a.seq_len
    corr = {k: [] for k in kinds}
    for ch, name in enumerate(data.channels):
        t0 = time.time()
        windows = data.windows(a.split, ch, a.eval_stride)
        if len(windows) < a.queries:
            raise SystemExit(f"{a.split} has only {len(windows)} origins at stride {a.eval_stride}")
        if a.split == "train":
            idx = np.arange(len(windows) - a.queries, len(windows))
        else:
            idx = np.unique(np.linspace(0, len(windows) - 1, a.queries or len(windows)).round().astype(np.int64))
        starts = idx * a.eval_stride
        query = windows[idx]
        history = data.history(a.split, ch, starts, a.history)
        pool = data.windows("train", ch, a.db_stride)
        limit = data.past_limit(a.split, starts, a.db_stride, GAP)
        cand, valid = candidates(limit, len(pool), a.candidates, a.seed, name)
        if not valid.all():
            raise SystemExit(f"{name}: an early origin has fewer than {a.candidates} windows "
                             "before it; lower --candidates or --queries")
        q_fut = standardize(query[:, s:].astype(np.float64), history[:, -s:].astype(np.float64))
        if a.head_weights == "precompute":
            scores = pool_scores(rows, history, pool, cand, s, heads, layers)
        else:
            scores = {"attn": []}
            for i in range(len(query)):
                ex = pool[cand[i]]
                att = rows.head_attention(history[i:i + 1], ex[None, :, :s], ex[None, :, s:],
                                          valid[i:i + 1], heads)[0]
                scores["attn"].append(head_weights(att, valid[i]))
        for i in range(len(query)):
            ex = pool[cand[i]]
            e_fut = standardize(ex[:, s:].astype(np.float64), ex[:, :s].astype(np.float64))
            near = -np.sqrt(((e_fut - q_fut[i]) ** 2).mean(-1))
            for k in kinds:
                corr[k].append(spearman(scores[k][i], near))
        print(f"  [{ch + 1}/{data.n_channels}] {name}: {len(query)} queries, {time.time() - t0:.0f} s",
              flush=True)

    lo, hi = map(int, a.layers.split("-"))
    out = {}
    for k in kinds:
        mean = np.mean(corr[k], 0)
        if k in HEAD_KINDS:
            mean = mean.reshape(n_layers, rows.n_heads)
            units = [(mean[li, h], f"L{li + 1}H{h + 1}") for li in range(n_layers) for h in range(rows.n_heads)]
            inside = [u for u, (li, h) in zip(units, heads) if lo - 1 <= li < hi]
        else:
            units = [(c, f"L{li + 1}" if k != "final" else "final") for li, c in enumerate(mean)]
            inside = [u for li, u in enumerate(units) if k == "final" or lo - 1 <= li < hi]
        spec = ",".join(u for _, u in sorted(inside, reverse=True)[:a.top])
        print(f"\n{k}: top 10 by rank correlation (mean over {len(corr[k])} {a.split} queries)")
        for c, u in sorted(units, reverse=True)[:10]:
            print(f"  {u:<7s} {c:+.3f}")
        print(f"  layers {a.layers}, top {a.top}: {spec}")
        out[k] = {"top": spec, "correlation": mean.round(4).tolist()}
    os.makedirs(a.output_dir, exist_ok=True)
    path = os.path.join(a.output_dir, f"heads_{a.dataset}_{a.split}_{a.head_weights}_h{a.history}.json")
    with open(path, "w") as f:
        json.dump({**vars(a), "gap": GAP, "model": MODEL_NAME, "heads": out["attn"]["top"],
                   "correlation": out["attn"]["correlation"], "kinds": out}, f, indent=1)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
