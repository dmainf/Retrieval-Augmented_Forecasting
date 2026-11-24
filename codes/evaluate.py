"""MSE, MAE and quantile loss over a prediction table written by run.py.

Main scores: errors divided by each channel's train std, as in the long-term
forecasting literature (PatchTST etc. standardize every channel with train
statistics), then averaged over channels. The train std is recomputed from the
dataset named in the run's _args.json.

e-scores: errors divided by the std of the channel's own evaluated ground truth
instead. That weights the channels by how hard the evaluated period is rather
than by their train scale, and can change an improvement ratio noticeably.

The table is long (one row per channel, window and horizon step, columns true
and q0.1 ... q0.9).

  python3 codes/evaluate.py results/ETTh1_val_none.parquet [more.parquet ...]

--compare puts runs side by side against a reference run (e.g. the proposed method) on the
same query windows: the change of MSE and MAE against no reference, the difference to the
reference in points with its standard error (the origins resampled per channel 1000 times),
the share of forecast steps where the movement (run - no reference) has the sign of the needed
correction (true - no reference), the mean |movement|, the change of MSE against the reference
per channel and the share of origins whose MSE is below the reference's. Every run must cover
the base run's origins.

  python3 codes/evaluate.py --compare results/ETTh1_val_none results/ETTh1_val_attn_L12H1+L11H10+L10H12 \
      results/ETTh1_val_attn_L12H1+L11H10+L10H12_vavpre [more runs ...] --out results/x.json

--picks compares the examples runs picked (<run>_selected.parquet, also from run.py --select-only)
with the oracle's, the K examples whose futures are nearest to the query's true future (run.py
--select fut=fut; it cheats by reading the true future), without running the model: per run the
share of its examples among the oracle's (overlap), the mean RMSE between two of its picked examples' futures (spread),
the RMSE between the mean of the picked futures and the query's true future (center)
and the mean RMSE between a picked example's future and the query's true
future, each standardized on its own context (dist).

  python3 codes/evaluate.py --picks results/ETTh1_val_fut-fut results/ETTh1_val_attn_L12H1+L11H10+L10H12 \
      [more runs ...] --out results/x.json
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

from dataset import TimeSeriesData

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS = ["MSE", "MAE", "QL"]


def quantile_columns(table: pd.DataFrame) -> list:
    return sorted([c for c in table.columns if re.fullmatch(r"q\d*\.?\d+", c)], key=lambda c: float(c[1:]))


def train_std(args_path: str) -> dict:
    """Per-channel std (ddof 0) of the train split of the run's dataset."""
    with open(args_path) as f:
        a = json.load(f)
    root = a["root_path"]
    if not os.path.isabs(root) and not os.path.isdir(root):
        root = os.path.join(REPO, root)
    data = TimeSeriesData(root, a["dataset"], a["seq_len"], a["pred_len"],
                          ett_months=tuple(a["ett_months"]), train_frac=a["train_frac"],
                          test_frac=a["test_frac"])
    lo, hi = data.borders[0][0], data.borders[1][0]
    return dict(zip(data.channels, data.values[lo:hi].std(0).astype(np.float64)))


def channel_scores(table: pd.DataFrame, scale: dict = None, median: str = "q0.5") -> pd.DataFrame:
    """MSE / MAE of the median and mean pinball loss per channel, each divided by
    scale[channel] (squared for MSE), or by the std of the channel's own truth
    when scale is None."""
    if median not in table.columns:
        raise ValueError(f"no {median} column in the table")
    cols = quantile_columns(table)
    rows = {}
    for name, g in table.groupby("channel", sort=False):
        y = g["true"].to_numpy(np.float64)
        err = g[median].to_numpy(np.float64) - y
        ql = 0.0
        for c in cols:
            q, diff = float(c[1:]), y - g[c].to_numpy(np.float64)
            ql += np.maximum(q * diff, (q - 1.0) * diff).mean()
        sd = y.std() if scale is None else scale[name]
        rows[name] = {"MSE": (err ** 2).mean() / sd ** 2, "MAE": np.abs(err).mean() / sd,
                      "QL": ql / len(cols) / sd}
    return pd.DataFrame.from_dict(rows, orient="index")


def evaluate(path: str) -> dict:
    """Main scores (train std) and e-scores (evaluated-truth std), averaged over channels."""
    table = pd.read_parquet(path)
    main = channel_scores(table, train_std(path[:-len(".parquet")] + "_args.json")).mean()
    own = channel_scores(table).mean()
    return {**main.to_dict(), **{f"e{k}": v for k, v in own.items()}}


KEY = ["channel", "window", "h"]


def _stem(path: str) -> str:
    return path[:-len(".parquet")] if path.endswith(".parquet") else path


def _load_run(stem: str) -> pd.DataFrame:
    return pd.read_parquet(f"{stem}.parquet").sort_values(KEY).reset_index(drop=True)


def compare(base: str, ref: str, runs: list, n_boot: int = 1000) -> dict:
    """See --compare in the module docstring. Errors are in train-std units; the reference must
    cover every query of the base run."""
    B = pd.read_parquet(f"{_stem(base)}.parquet").sort_values(KEY).reset_index(drop=True)
    scale = B.channel.map(train_std(f"{_stem(base)}_args.json")).to_numpy()
    H = int(B.h.max()) + 1
    eps = ((B["true"] - B["q0.5"]).to_numpy() / scale).reshape(-1, H)
    chq, wq = B.channel.to_numpy().reshape(-1, H)[:, 0], B.window.to_numpy().reshape(-1, H)[:, 0]
    chs = list(dict.fromkeys(chq))
    qid = pd.MultiIndex.from_arrays([chq, wq])

    def moves(stem):
        T = _load_run(_stem(stem))
        d = np.full(eps.shape, np.nan)
        tq = pd.MultiIndex.from_arrays([T.channel.to_numpy()[::H], T.window.to_numpy()[::H]])
        pos = qid.get_indexer(tq)
        if (pos < 0).any():
            raise SystemExit(f"{stem} has query windows the base run does not have")
        if not (T["true"].to_numpy().reshape(-1, H) == B["true"].to_numpy().reshape(-1, H)[pos]).all():
            raise SystemExit(f"{stem} and the base run disagree on the true values")
        d[pos] = (T["q0.5"].to_numpy().reshape(-1, H) - B["q0.5"].to_numpy().reshape(-1, H)[pos]) / scale.reshape(-1, H)[pos]
        return eps - d, d

    def stats(E, rows):
        return np.mean([((E[r] ** 2).mean(), np.abs(E[r]).mean()) for r in rows], 0)

    EP, DP = moves(ref)
    if np.isnan(DP).any():
        raise SystemExit(f"the reference {ref} must cover every query of the base run")
    rows = [np.flatnonzero(chq == c) for c in chs]
    rng = np.random.default_rng(0)
    boots = [[rng.choice(r, len(r)) for r in rows] for _ in range(n_boot)]
    b0, bp = stats(eps, rows), stats(EP, rows)
    rel = lambda E, rw: stats(E, rw) / stats(eps, rw) - 1
    res = {_stem(ref): {"mse": (bp[0] / b0[0] - 1) * 100, "mae": (bp[1] / b0[1] - 1) * 100,
                        "agree": float((np.sign(DP) == np.sign(eps)).mean() * 100), "move": float(np.abs(DP).mean())}}
    for r_ in runs:
        name, (E, D) = _stem(r_), moves(r_)
        if np.isnan(D).any():
            raise SystemExit(f"{name} does not cover every query of the base run")
        m = rel(E, rows) * 100
        diff = np.array([(rel(E, b) - rel(EP, b)) * 100 for b in boots])
        res[name] = {"mse": m[0], "mae": m[1], "dmse": m[0] - res[_stem(ref)]["mse"], "dmse_se": diff[:, 0].std(),
                     "dmae": m[1] - res[_stem(ref)]["mae"], "dmae_se": diff[:, 1].std(),
                     "agree": float((np.sign(D) == np.sign(eps)).mean() * 100), "move": float(np.abs(D).mean()),
                     "ch_mse_vs_ref": {c: (stats(E, [x])[0] / stats(EP, [x])[0] - 1) * 100 for c, x in zip(chs, rows)},
                     "origins_mse_below_ref": float(((E ** 2).mean(1) < (EP ** 2).mean(1)).mean() * 100)}
    return {"n": len(eps), "base": list(b0), "ref": _stem(ref), "runs": res}


def _load_picks(stem: str) -> pd.DataFrame:
    return pd.read_parquet(f"{stem}_selected.parquet")


def picks(oracle: str, runs: list) -> dict:
    """See --picks in the module docstring."""
    with open(f"{_stem(oracle)}_args.json") as f:
        a = json.load(f)
    root = a["root_path"] if os.path.isabs(a["root_path"]) or os.path.isdir(a["root_path"]) else os.path.join(REPO, a["root_path"])
    s, k = a["seq_len"], a["top_k"]
    data = TimeSeriesData(root, a["dataset"], s, a["pred_len"], ett_months=tuple(a["ett_months"]),
                          train_frac=a["train_frac"], test_frac=a["test_frac"])
    O = _load_picks(_stem(oracle))
    loaded = [(_stem(r), _load_picks(_stem(r))) for r in runs]
    acc = {name: [] for name, _ in loaded}
    for ch, name in enumerate(data.channels):
        pool = data.windows("train", ch, a.get("db_stride", 1)).astype(np.float64)
        pc = pool[:, :s]
        fut = (pool[:, s:] - pc.mean(1, keepdims=True)) / np.maximum(pc.std(1, keepdims=True), 1e-8)
        query = data.windows(a["eval_split"], ch, a["eval_stride"])
        starts = np.arange(len(query)) * a["eval_stride"]
        hist = data.history(a["eval_split"], ch, starts, s)
        y = (query[:, s:] - hist.mean(1, keepdims=True)) / np.maximum(hist.std(1, keepdims=True), 1e-8)
        oracle_by = O[O.channel == name].sort_values(["window", "rank"]).groupby("window")["db_index"].apply(np.array)
        for run, P in loaded:
            for w, ix in P[P.channel == name].sort_values(["window", "rank"]).groupby("window")["db_index"]:
                ix = ix.to_numpy()
                f = fut[ix]
                sq = (f ** 2).sum(1)
                d2 = np.maximum(sq[:, None] + sq[None] - 2 * f @ f.T, 0) / f.shape[1]
                spread = np.sqrt(d2)[~np.eye(len(ix), dtype=bool)].mean()
                center = np.sqrt(((f.mean(0) - y[w]) ** 2).mean())
                acc[run].append((np.isin(ix, oracle_by[w]).mean(), np.sqrt(((f - y[w]) ** 2).mean(1)).mean(), spread, center))
    out = {}
    for run, v in acc.items():
        v = np.array(v)
        out[run] = {"overlap": float(v[:, 0].mean()), "dist": float(v[:, 1].mean()), "spread": float(v[:, 2].mean()),
                    "center": float(v[:, 3].mean()),
                    "n": len(v)}
    return {"oracle": _stem(oracle), "top_k": k, "runs": out}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--picks":
        args = sys.argv[2:]
        dest = args[args.index("--out") + 1] if "--out" in args else None
        args = [x for i, x in enumerate(args) if x != "--out" and (i == 0 or args[i - 1] != "--out")]
        if len(args) < 2:
            raise SystemExit(__doc__)
        out = picks(args[0], args[1:])
        print(f"overlap with the oracle {out['oracle']} | dist to the true future | spread among the picks | center to the true future")
        for name, r in out["runs"].items():
            print(f"  {r['overlap'] * 100:5.1f}%  {r['dist']:.3f}  {r['spread']:.3f}  {r['center']:.3f}  ({r['n']} queries)  {name}")
        if dest:
            json.dump(out, open(dest, "w"), ensure_ascii=False, indent=1)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        args = sys.argv[2:]
        dest = args[args.index("--out") + 1] if "--out" in args else None
        args = [a for i, a in enumerate(args) if a != "--out" and (i == 0 or args[i - 1] != "--out")]
        if len(args) < 3:
            raise SystemExit(__doc__)
        out = compare(args[0], args[1], args[2:])
        print(f"{out['n']} queries, reference {out['ref']}")
        for name, r in out["runs"].items():
            vs = f" | vs ref MSE {r['dmse']:+5.2f}±{r['dmse_se']:.2f} MAE {r['dmae']:+5.2f}±{r['dmae_se']:.2f}" if "dmse" in r else ""
            print(f"  MSE {r['mse']:+6.2f}% MAE {r['mae']:+6.2f}%{vs} | agree {r['agree']:.1f}% move {r['move']:.3f}  {name}")
        if dest:
            json.dump(out, open(dest, "w"), ensure_ascii=False, indent=1)
        return
    paths = [p for p in sys.argv[1:] if not p.endswith("_selected.parquet")]
    if not paths:
        raise SystemExit(__doc__)
    names = METRICS + [f"e{m}" for m in METRICS]
    print("".join(f"{n:>8s}" for n in names) + "  path")
    for path in paths:
        s = evaluate(path)
        print("".join(f"{s[n]:8.4f}" for n in names) + f"  {path}")


if __name__ == "__main__":
    main()
