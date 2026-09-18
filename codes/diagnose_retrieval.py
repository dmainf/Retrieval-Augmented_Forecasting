"""Is context similarity a usable proxy for future similarity?

Every retrieval method in the RAF literature ranks database windows by how close
their *context* is to the query context, then concatenates their *futures*. That
only helps if the two orderings agree. This script measures the agreement
directly, without involving the backbone at all.

For every query window and every database window of the same channel it computes

  d_x  distance between the two contexts   (what the retriever sees)
  d_y  distance between the two futures    (what the retriever actually wants)

and reports three views:

  A  d_x vs d_y as within-query percentile ranks, i.e. the Spearman scatter.
     A diagonal band means retrieval is sound; a square blob means d_x carries
     no information about d_y.
  B  where the context-top-k land in the future-distance ranking. Uniform means
     the retriever is doing no better than picking at random.
  C  the future distance actually achieved as k grows, for the context ranking,
     the oracle ranking, and random picks. The gap between the first two is the
     headroom any better retriever has to work with.

  python3 codes/diagnose_retrieval.py --dataset ETTh1 --eval-stride 16
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

from dataset import TimeSeriesData

KS = (1, 2, 4, 8, 11)


def _ranks(d: np.ndarray) -> np.ndarray:
    """Row-wise percentile rank in [0, 1]; 0 is the closest."""
    order = np.argsort(d, axis=1)
    out = np.empty_like(d)
    idx = np.arange(d.shape[1], dtype=np.float64) / (d.shape[1] - 1)
    np.put_along_axis(out, order, np.broadcast_to(idx, d.shape), axis=1)
    return out


def _distances(query: np.ndarray, db: np.ndarray, metric: str) -> np.ndarray:
    if metric == "euclidean":
        return cdist(query, db, "euclidean")
    if metric == "correlation":
        return cdist(query, db, "correlation")
    if metric == "cosine":
        return cdist(query, db, "cosine")
    raise ValueError(metric)


def analyse(args):
    data = TimeSeriesData(args.root_path, args.dataset, args.seq_len, args.pred_len,
                          args.channels)
    stride = args.eval_stride if args.eval_stride > 0 else data.win_len

    per_channel = {}
    for c, name in enumerate(data.channels):
        query = data.windows(args.eval_split, c, stride)
        db = data.windows(args.retrieval_split, c, args.retrieval_stride)
        qx, qy = query[:, :args.seq_len], query[:, args.seq_len:]
        dx = _distances(qx, db[:, :args.seq_len], args.metric)
        dy = _distances(qy, db[:, args.seq_len:], args.metric)
        rx, ry = _ranks(dx), _ranks(dy)

        rho = float(np.mean([np.corrcoef(a, b)[0, 1] for a, b in zip(rx, ry)]))

        ctx_order = np.argsort(dx, axis=1)
        fut_order = np.argsort(dy, axis=1)
        topk_future_rank = np.take_along_axis(ry, ctx_order[:, :max(KS)], axis=1)

        median_dy = np.median(dy, axis=1, keepdims=True)
        curves = {}
        for label, order in (("context", ctx_order), ("oracle", fut_order)):
            picked = np.take_along_axis(dy, order[:, :max(KS)], axis=1) / median_dy
            curves[label] = [float(picked[:, :k].mean()) for k in KS]
        curves["random"] = [1.0] * len(KS)

        per_channel[name] = dict(rho=rho, rx=rx, ry=ry,
                                 topk_future_rank=topk_future_rank, curves=curves)
        print(f"  {name:6s} spearman(d_x, d_y) = {rho:+.3f}   "
              f"top-1 future-rank median = {np.median(topk_future_rank[:, 0]):.3f}")

    return data, per_channel


def plot(data, per_channel, out_dir, tag):
    os.makedirs(out_dir, exist_ok=True)
    names = list(per_channel)
    n = len(names)

    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.6), sharex=True, sharey=True)
    for ax, name in zip(np.atleast_1d(axes), names):
        d = per_channel[name]
        rx, ry = d["rx"].ravel(), d["ry"].ravel()
        if rx.size > 200_000:
            sel = np.random.default_rng(0).choice(rx.size, 200_000, replace=False)
            rx, ry = rx[sel], ry[sel]
        ax.hexbin(rx, ry, gridsize=40, cmap="viridis", bins="log", linewidths=0)
        ax.plot([0, 1], [0, 1], color="w", lw=0.8, ls="--")
        ax.set_title(f"{name}  $\\rho$={d['rho']:+.2f}", fontsize=9)
        ax.set_xlabel("context rank", fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("future rank", fontsize=8)
    fig.suptitle("A. context-distance rank vs future-distance rank", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{tag}_A_rank_scatter.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    pooled = np.concatenate([per_channel[n_]["topk_future_rank"][:, 0] for n_ in names])
    ax.hist(pooled, bins=40, range=(0, 1), color="#4c72b0", edgecolor="none")
    ax.axhline(len(pooled) / 40, color="crimson", ls="--", lw=1,
               label="uniform (= random retrieval)")
    ax.set_xlabel("future-distance percentile of the context-nearest window")
    ax.set_ylabel("count")
    ax.set_title("B. where the top-1 retrieved window lands on future distance", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{tag}_B_top1_future_rank.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for label, style in (("context", "-o"), ("oracle", "-s"), ("random", "--")):
        y = np.mean([per_channel[n_]["curves"][label] for n_ in names], axis=0)
        ax.plot(KS, y, style, label=label, ms=4)
    ax.set_xlabel("k")
    ax.set_ylabel("mean future distance / median")
    ax.set_title("C. how close the retrieved futures actually are", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{tag}_C_future_distance.png"), dpi=150)
    plt.close(fig)

    print(f"\nsaved 3 figures to {out_dir}/{tag}_*.png")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root-path", default="./Datasets/_parquet")
    p.add_argument("--dataset", default="ETTh1")
    p.add_argument("--channels", default="")
    p.add_argument("--seq-len", default=96, type=int)
    p.add_argument("--pred-len", default=64, type=int)
    p.add_argument("--eval-split", default="test")
    p.add_argument("--eval-stride", default=16, type=int)
    p.add_argument("--retrieval-split", default="train")
    p.add_argument("--retrieval-stride", default=1, type=int)
    p.add_argument("--metric", default="euclidean",
                   choices=["euclidean", "correlation", "cosine"])
    p.add_argument("--output-dir", default="./results/diagnostics")
    args = p.parse_args()

    print(f"{args.dataset}  metric={args.metric}")
    data, per_channel = analyse(args)
    overall = np.mean([d["rho"] for d in per_channel.values()])
    print(f"\n  mean spearman(d_x, d_y) over channels = {overall:+.3f}")
    plot(data, per_channel, args.output_dir, f"{args.dataset}_{args.metric}")


if __name__ == "__main__":
    main()
