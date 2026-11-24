"""How the examples a run picked change the query's group-attention output, split into the
前項 (the candidate with its future hidden against the query) and the 変化項 (what the
candidate's future adds), on the val split.

Every row is run on its own (precompute). For the chosen heads, at the forecast columns and in the
space after W_O, with weights s_j the softmax of q0.k over the query itself and the picked set S:
  dpre_j  = |(v~_j - v0) W_O|,  dchg_j = |(v_j - v~_j) W_O|          (summed over heads and columns)
  delta   = sum_{j in S} s_j (v_j - v0) W_O = pre + chg               (per head and column)
  u0      = (v0* - v0) W_O, v0* the query's value with its true future put in (analysis only)
Per query:
  corr        Spearman of dpre and dchg over all the candidates
  dpre, dchg  their mean over S
  cos, cos_pre, cos_chg   cosine of delta, pre and chg with u0, averaged over heads and columns
  a_pre, a_chg  <pre, u0> / (|delta| |u0|) and <chg, u0> / (|delta| |u0|), averaged over heads and columns;
              per head and column they add up to cos, so they split it into what pre and chg bring
  n_pre, n_chg  |pre| / |u0| and |chg| / |u0|
  gain        <delta, u0> / |u0|^2, and off = |delta - gain u0| / |u0|
Values are averaged over the origins of each channel and then over the channels; differences against
--ref carry a bootstrap SE (origins resampled within each channel).

  python3 codes/contrib.py --heads L12H1,L11H10,L10H12 --runs results/ETTh1_val_attn_L12H1+L11H10+L10H12 \
      results/ETTh1_val_attn_L12H1+L11H10+L10H12_vavpre results/ETTh1_val_random --out results/ETTh1_val_contrib.json
"""
import argparse
import json
import os
import types

import numpy as np
import pandas as pd

from chronos2 import Chronos2Rows, load_chronos2
from dataset import TimeSeriesData
from heads import spearman
from run import (DATASET, DEFAULTS, DEVICE, ETT_MONTHS, GAP, MODEL_NAME, TEST_FRAC, TOKEN_BUDGET,
                 TRAIN_FRAC, REPORT_QUANTILES, parse_heads, pool_features)
from selection import candidates, out_projection

KEYS = ("corr", "dpre", "dchg", "cos", "cos_pre", "cos_chg", "a_pre", "a_chg", "n_pre", "n_chg", "gain", "off")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root-path", default="./Datasets/_parquet")
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--heads", required=True)
    p.add_argument("--runs", nargs="+", required=True, help="result stems with a _selected.parquet")
    p.add_argument("--ref", help="stem the differences are taken against (default: the first run)")
    p.add_argument("--n-boot", default=1000, type=int)
    p.add_argument("--threads", default=0, type=int)
    p.add_argument("--out", required=True)
    return p.parse_args()


def cosine(a, b):
    return (a * b).sum(-1) / np.maximum(np.sqrt((a ** 2).sum(-1) * (b ** 2).sum(-1)), 1e-30)


def per_query(q0, k0, v0, u0, K, V, Vm, cand, picked):
    """Values of one query; q0, k0, v0, u0 (heads, cols, d), K, V, Vm the pool's k, v, v~ (values mapped)."""
    v0 = v0.astype(np.float64)
    dpre = np.sqrt(((Vm[cand] - v0) ** 2).sum(-1)).sum((1, 2))
    dchg = np.sqrt(((V[cand] - Vm[cand]) ** 2).sum(-1)).sum((1, 2))
    out = {"corr": float(spearman(dpre[None], dchg)[0])}
    for name, S in picked.items():
        logit = np.einsum("hcd,mhcd->hcm", q0, K[S]).astype(np.float64)
        full = np.concatenate([(q0 * k0).sum(-1)[..., None].astype(np.float64), logit], -1)
        w = np.exp(full - full.max(-1, keepdims=True))
        s = (w / w.sum(-1, keepdims=True))[..., 1:]
        pre = np.einsum("hcm,mhcd->hcd", s, Vm[S] - v0)
        chg = np.einsum("hcm,mhcd->hcd", s, V[S] - Vm[S])
        delta = pre + chg
        uu = np.maximum((u0 ** 2).sum(-1), 1e-30)
        den = np.maximum(np.sqrt((delta ** 2).sum(-1) * uu), 1e-30)
        gain = (delta * u0).sum(-1) / uu
        off = np.sqrt(((delta - gain[..., None] * u0) ** 2).sum(-1) / uu)
        pos = np.isin(cand, S)
        out[name] = {"dpre": dpre[pos].mean(), "dchg": dchg[pos].mean(), "cos": cosine(delta, u0).mean(),
                     "cos_pre": cosine(pre, u0).mean(), "cos_chg": cosine(chg, u0).mean(),
                     "a_pre": ((pre * u0).sum(-1) / den).mean(), "a_chg": ((chg * u0).sum(-1) / den).mean(),
                     "n_pre": np.sqrt((pre ** 2).sum(-1) / uu).mean(), "n_chg": np.sqrt((chg ** 2).sum(-1) / uu).mean(),
                     "gain": gain.mean(), "off": off.mean()}
    return out


def main():
    a = parse_args()
    d = DEFAULTS
    cfg = types.SimpleNamespace(dataset=a.dataset, seq_len=d["seq_len"], pred_len=d["pred_len"],
                                db_stride=d["db_stride"])
    data = TimeSeriesData(a.root_path, a.dataset, d["seq_len"], d["pred_len"], "", ett_months=ETT_MONTHS,
                          train_frac=TRAIN_FRAC, test_frac=TEST_FRAC)
    rows = Chronos2Rows(load_chronos2(MODEL_NAME, DEVICE, a.threads), d["seq_len"], d["pred_len"],
                        REPORT_QUANTILES, TOKEN_BUDGET)
    heads = parse_heads(a.heads)
    proj = out_projection(rows.head_out_gram(heads))
    names = [os.path.basename(r) for r in a.runs]
    sel = {n: pd.read_parquet(r + "_selected.parquet") for n, r in zip(names, a.runs)}
    ref = os.path.basename(a.ref) if a.ref else names[0]
    s = d["seq_len"]
    per = {n: {k: [] for k in KEYS} for n in names}
    for ch, name in enumerate(data.channels):
        query = data.windows("val", ch, d["eval_stride"])
        win = np.arange(len(query), dtype=np.int64)
        starts = win * d["eval_stride"]
        history = data.history("val", ch, starts, d["history"])
        pool = data.windows("train", ch, d["db_stride"])
        limit = data.past_limit("val", starts, d["db_stride"], GAP)
        cand, valid = candidates(limit, len(pool), d["candidates"], d["seed"], name, win)
        if not valid.all():
            raise ValueError("every candidate must be valid")
        q0, k0, v0 = rows.head_vectors(history, None, heads, values=True)
        v0f = rows.head_vectors(history, query[:, s:], heads, values=True)[2]
        v0, v0f = proj(v0).astype(np.float64), proj(v0f).astype(np.float64)
        K, V = pool_features(cfg, rows, name, pool, True, heads, want=("k", "v"))
        Vm, = pool_features(cfg, rows, name, pool, False, heads, want=("v",))
        V, Vm = proj(V).astype(np.float64), proj(Vm).astype(np.float64)
        picks = {n: t[t.channel == name].sort_values(["window", "rank"]) for n, t in sel.items()}
        picks = {n: t.db_index.to_numpy().reshape(len(query), -1) for n, t in picks.items()}
        cols = {n: {k: [] for k in KEYS} for n in names}
        for i in range(len(query)):
            r = per_query(q0[i], k0[i], v0[i], v0f[i] - v0[i], K, V, Vm, cand[i], {n: p[i] for n, p in picks.items()})
            for n in names:
                cols[n]["corr"].append(r["corr"])
                for k in KEYS[1:]:
                    cols[n][k].append(r[n][k])
        for n in names:
            for k in KEYS:
                per[n][k].append(np.array(cols[n][k]))
        print(f"{name}: {len(query)} queries", flush=True)
    rng = np.random.default_rng(0)
    boots = [[rng.integers(0, len(x), len(x)) for x in per[ref]["corr"]] for _ in range(a.n_boot)]
    res = {}
    for n in names:
        res[n] = {}
        for k in KEYS:
            xs = per[n][k]
            res[n][k] = float(np.mean([x.mean() for x in xs]))
            if n != ref and k != "corr":
                diff = [x - y for x, y in zip(xs, per[ref][k])]
                res[n][k + "_diff"] = float(np.mean([x.mean() for x in diff]))
                res[n][k + "_se"] = float(np.std([np.mean([x[j].mean() for x, j in zip(diff, b)]) for b in boots]))
    json.dump({"dataset": a.dataset, "split": "val", "heads": a.heads, "ref": ref, "runs": res},
              open(a.out, "w"), indent=1)
    for n in names:
        print(n, " ".join(f"{k} {res[n][k]:.4f}" for k in KEYS))


if __name__ == "__main__":
    main()
