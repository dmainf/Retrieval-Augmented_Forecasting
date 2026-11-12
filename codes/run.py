"""Forecast every query window with a frozen Chronos-2 and save predictions with ground truth.

  --fusion   parallel (examples as covariate rows) or concat (examples spliced in front)
  --select   none, random, a match (ctx=ctx, fc=fc, fc=fut, fut=fc, fut=fut) or attn;
             see selection.py
  --hide     columns where the query row does not read the example rows (parallel only)

Every setting but --select defaults to the proposed method, so --select attn is the
proposal and the other selections are compared under the same conditions.

The output table is long: one row per channel, query window and horizon step h,
with the timestamp of that step, the true value and one column per quantile
(q0.1 ... q0.9). evaluate.py turns it into MSE, MAE and QL.

  python3 codes/run.py                      # no reference
  python3 codes/run.py --select attn        # the proposed method
  python3 codes/run.py --select attn --hide ""
  python3 codes/run.py --select random
  python3 codes/run.py --select ctx=ctx --top-k 8 --fusion concat
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
from selection import (MATCHES, METRICS, VALUE_SCORES, candidates, diverse_top_k, head_score, match,
                       past_head_mix, precomputed_attention, spread_match, top_k, value_score)

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
    "heads": "L12H1,L12H7,L11H10",
    "head_weights": "precompute",
    "select_hide": False,
    "value_score": "none",
    "select_cols": "P1,P2,P3,P4",
    "head_adapt": "none",
    "diversity": 0.0,
    "spread_match": 0,
    "origins": "all",
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
    p.add_argument("--select", choices=("none", "random") + MATCHES + ("attn",))
    p.add_argument("--fusion", choices=["parallel", "concat"])
    p.add_argument("--top-k", type=int)
    p.add_argument("--candidates", type=int, help="drawn per query at random; 0 = every usable window")
    p.add_argument("--hide", help="e.g. P1 or P1,C1 (P: forecast patch, C: context patch from the "
                                  "origin back, REG); empty = read everywhere")
    p.add_argument("--metric", choices=METRICS)
    p.add_argument("--match-norm", choices=["context", "raw"])
    p.add_argument("--heads", help="e.g. L12H1,L12H7,L11H10 (1-based layer and head); "
                                   "L12H1*0.5,L12H7*0.3 weights the average over the heads")
    p.add_argument("--head-weights", choices=["precompute", "exact"])
    p.add_argument("--select-hide", action="store_const", const=True,
                   help="exact: apply --hide in the forward that picks the examples too")
    p.add_argument("--value-score", choices=VALUE_SCORES,
                   help="attn, precompute: score the candidates with their values too (selection.py)")
    p.add_argument("--select-cols", help="forecast columns whose attention picks the examples, e.g. P2,P3,P4")
    p.add_argument("--head-adapt", choices=["none", "past"],
                   help="attn, precompute: past weights the heads per query by how well each picks "
                        "candidates near the query's last patch when the query is cut one patch back")
    p.add_argument("--diversity", type=float,
                   help="attn, precompute: lambda > 0 picks the K examples one by one by log score - "
                        "lambda * the largest cosine of the candidate's keys to the picked ones'")
    p.add_argument("--spread-match", type=int,
                   help="attn: M > K picks the K examples among the M best-scoring candidates whose "
                        "futures spread like the query's no-reference forecast (selection.py)")
    p.add_argument("--origins", choices=["all", "odd", "even"], help="forecast only odd or even origins")
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
    if (a.value_score != "none" or a.select_cols != DEFAULTS["select_cols"]) \
            and a.head_weights != "precompute":
        p.error("--value-score and --select-cols need --head-weights precompute")
    if a.head_adapt != "none" and (a.head_weights != "precompute" or a.value_score != "none"
                                   or a.select_cols != DEFAULTS["select_cols"]
                                   or "*" in a.heads or a.db_stride != 1):
        p.error("--head-adapt needs --head-weights precompute, --db-stride 1, unweighted --heads "
                "and no --value-score or --select-cols")
    if a.diversity and (a.head_weights != "precompute" or a.value_score != "none"):
        p.error("--diversity needs --head-weights precompute and no --value-score")
    if a.spread_match and (a.select != "attn" or a.diversity or a.spread_match <= a.top_k):
        p.error("--spread-match needs --select attn, no --diversity and M > --top-k")
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
            parts += [a.heads.replace(",", "+").replace("*", "x")] if a.heads != d["heads"] else []
            parts += [a.head_weights] if a.head_weights != d["head_weights"] else []
            parts += ["selhide"] if a.select_hide else []
            parts += [f"v{a.value_score}"] if a.value_score != d["value_score"] else []
            parts += [f"div{a.diversity:g}"] if a.diversity != d["diversity"] else []
            parts += [f"spread{a.spread_match}"] if a.spread_match != d["spread_match"] else []
            parts += [f"adapt{a.head_adapt}"] if a.head_adapt != d["head_adapt"] else []
            parts += ["sc" + re.sub(r"[^0-9]", "", a.select_cols)] if a.select_cols != d["select_cols"] else []
        parts += [f"sl{a.seq_len}"] if a.seq_len != d["seq_len"] else []
        parts += [f"db{a.db_stride}"] if a.db_stride != d["db_stride"] else []
    for tag, key in (("pl", "pred_len"), ("h", "history"), ("es", "eval_stride")):
        parts += [f"{tag}{getattr(a, key)}"] if getattr(a, key) != d[key] else []
    parts += [a.origins] if a.origins != d["origins"] else []
    if a.channels:
        parts.append("ch" + "+".join(re.sub(r"[^0-9A-Za-z]", "", c) for c in a.channels.split(",")))
    return "_".join(parts)


def head_mix(spec: str):
    """"L12H1*0.5,L12H7" -> heads and their weights in the average (1 when not given)."""
    names, w = zip(*[(h.split("*")[0], float(h.split("*")[1]) if "*" in h else 1.0) for h in spec.split(",")])
    return parse_heads(",".join(names)), np.array(w)


def select_head(a, rows, heads, mix, history, pool, cand, valid, hide, history_past=None):
    """{"score": (n, M) scores of the candidates by the group attention of the chosen heads}, with
    --head-adapt the per-query head weights "pi" and correlations "rho" (n, heads) and with
    --diversity the "order" (n, K) of the picked candidates' positions."""
    score = np.empty(cand.shape)
    pis, rhos = np.empty((len(cand), len(heads))), np.empty((len(cand), len(heads)))
    s = a.seq_len
    if a.head_weights == "precompute":
        plain = a.value_score == "none" and a.select_cols == DEFAULTS["select_cols"]
        if plain:
            q, k_self = rows.head_vectors(history, None, heads)
            _, k_pool = rows.head_vectors(pool[:, :s], pool[:, s:], heads)
        else:
            q, k_self, v_self = rows.head_vectors(history, None, heads, values=True)
            _, k_pool, v_pool = rows.head_vectors(pool[:, :s], pool[:, s:], heads, values=True)
            gram = rows.head_out_gram(heads)
            cols = [int(c.strip()[1:]) - 1 for c in a.select_cols.split(",")]
        if a.head_adapt == "past":
            q_p, k_self_p = rows.head_vectors(history_past, None, heads)
            sh = rows.patch
        for i in range(len(cand)):
            att = precomputed_attention(q[i], k_self[i], k_pool, cand[i], valid[i])
            if a.head_adapt == "past":
                vp = valid[i] & (cand[i] >= sh)
                cp = np.maximum(cand[i] - sh, 0)
                att_p = precomputed_attention(q_p[i], k_self_p[i], k_pool, cp, vp)
                dist = np.sqrt(((pool[cp, s:s + sh] - history[i, -sh:]) ** 2).sum(-1))
                pis[i], rhos[i] = past_head_mix(att_p, vp, dist)
                score[i] = head_score(att, valid[i], pis[i])
            elif plain:
                score[i] = head_score(att, valid[i], mix)
            else:
                score[i] = value_score(att, v_self[i], v_pool[cand[i]], gram, a.value_score, cols,
                                       valid[i], mix)
        out = {"score": score}
        if a.head_adapt != "none":
            out.update(pi=pis, rho=rhos)
        if a.diversity:
            kn = k_pool.reshape(len(k_pool), -1).astype(np.float64)
            kn /= np.maximum(np.linalg.norm(kn, axis=1, keepdims=True), 1e-30)
            out["order"] = np.stack([diverse_top_k(score[i], kn[cand[i]], valid[i], a.top_k, a.diversity)
                                     for i in range(len(cand))])
        return out
    step = max(1, rows.token_budget // ((history.shape[1] // rows.patch + 1 + rows.n_out)
                                        + cand.shape[1] * rows.s))
    for lo in range(0, len(cand), step):
        sl = slice(lo, lo + step)
        ex = pool[cand[sl]]
        att = rows.head_attention(history[sl], ex[..., :s], ex[..., s:], valid[sl], heads,
                                  hide if a.select_hide else ())
        for j, i in enumerate(range(lo, min(lo + step, len(cand)))):
            score[i] = head_score(att[j], valid[i], mix)
    return {"score": score}


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
    if args.history > rows.context_length:
        raise SystemExit(f"--history {args.history} exceeds the model's {rows.context_length}-point context")
    hide = rows.columns(args.hide)
    heads, mix = head_mix(args.heads)
    s = args.seq_len

    tables, logs, mixes = [], [], []
    for ch, name in enumerate(data.channels):
        t0 = time.time()
        query = data.windows(args.eval_split, ch, args.eval_stride)
        win = np.arange(len(query), dtype=np.int64)
        if args.origins != "all":
            win = win[win % 2 == (1 if args.origins == "odd" else 0)]
            query = query[win]
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
            else:
                if args.head_adapt == "past":
                    past = data.history(args.eval_split, ch, starts - rows.patch, args.history)
                    sel = select_head(args, rows, heads, mix, history, pool, cand, valid, hide, past)
                    pi, rho = sel["pi"], sel["rho"]
                    mixes.append(pd.DataFrame({"channel": name, "window": np.repeat(win, len(heads)),
                                               "head": np.tile(args.heads.split(","), len(win)),
                                               "pi": pi.reshape(-1), "rho": rho.reshape(-1)}))
                else:
                    sel = select_head(args, rows, heads, mix, history, pool, cand, valid, hide)
                if args.spread_match:
                    nr = rows.predict(history)
                    q1, q9 = REPORT_QUANTILES.index(0.1), REPORT_QUANTILES.index(0.9)
                    sigma = (nr[:, q9] - nr[:, q1]) / 2.56 / np.maximum(history.std(1, keepdims=True), 1e-5)
                    ctx = pool[:, :s].astype(np.float64)
                    fut = (pool[:, s:] - ctx.mean(1, keepdims=True)) / np.maximum(ctx.std(1, keepdims=True), 1e-5)
                    sel["order"] = np.stack([spread_match(sel["score"][i], valid[i], fut[cand[i]], sigma[i],
                                                          args.top_k, args.spread_match) for i in range(len(cand))])
                if "order" in sel:
                    idx = np.take_along_axis(cand, sel["order"], 1)
                    score = np.take_along_axis(sel["score"], sel["order"], 1)
                else:
                    idx, score = top_k(sel["score"], cand, valid, args.top_k)
            ex = pool[idx]
            if args.fusion == "parallel":
                out = rows.predict(history, ex[..., :s], ex[..., s:], hide)
            else:
                out = rows.predict(concat_context(history, ex, rows.patch, rows.context_length))
            logs.append(pd.DataFrame({"channel": name,
                                      "window": np.repeat(win, args.top_k),
                                      "rank": np.tile(np.arange(args.top_k), len(query)),
                                      "db_index": idx.reshape(-1),
                                      "score": score.reshape(-1)}))
        n, p = len(query), args.pred_len
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

    pd.concat(tables, ignore_index=True).to_parquet(f"{stem}.parquet", compression="zstd", index=False)
    with open(f"{stem}_args.json", "w") as f:
        json.dump({**vars(args), "gap": GAP, "ett_months": ETT_MONTHS, "train_frac": TRAIN_FRAC,
                   "test_frac": TEST_FRAC, "model": MODEL_NAME, "quantiles": REPORT_QUANTILES,
                   "channels_used": data.channels}, f, indent=1)
    if mixes:
        pd.concat(mixes, ignore_index=True).to_parquet(f"{stem}_headmix.parquet", index=False)
    if logs:
        pd.concat(logs, ignore_index=True).to_parquet(f"{stem}_selected.parquet", index=False)
    elif os.path.exists(f"{stem}_selected.parquet"):
        os.remove(f"{stem}_selected.parquet")
    print(f"saved {stem}.parquet")


if __name__ == "__main__":
    main()
