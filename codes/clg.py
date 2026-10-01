"""Curriculum Latent Gradient selection of a task-level example set for Chronos-2.

Zhang et al. (ACL Findings 2025) select a fixed many-shot demonstration set by
gradient matching. Here the pieces map onto the parallel container:

  latent concept tokens   L learnable example rows of seq_len + pred_len values,
                          placed beside the history exactly where a real example
                          goes; Chronos-2 stays frozen and only the rows train
  training instance       a train window: its 2032-point history is the query,
                          its 64-point future the target; the same window cut to
                          its last 96 + 64 points is the candidate example
  curriculum gradient     the gradient of that window's loss with respect to the
                          latent rows, at initialization and after every epoch,
                          concatenated
  gradient matching       pick K windows whose mean curriculum gradient is
                          closest in L2 to the mean over all train windows,
                          greedily, then refined by single swaps

  python3 codes/clg.py learn  --channel OT            # latent rows + gradients
  python3 codes/clg.py select --top-k 8,32,128,256    # sets from saved gradients
  python3 codes/clg.py bestofn --top-k 8,32,128,256
"""
import argparse
import json
import os

import numpy as np
import torch

from dataset import TimeSeriesData
from sparse_rows import SparseRows, load_chronos2


def train_queries(data, ch, history, pool_stride):
    pool = data.windows("train", ch, pool_stride)
    starts = np.arange(len(pool), dtype=np.int64) * pool_stride
    hist = data.history("train", ch, starts, history)
    return pool, hist, pool[:, data.seq_len:]


def batch_loss(sr, hist, fut, rows, seq_len, per_sample=False):
    """rows (L, win) shared, or (B, L, win) one copy per query."""
    if rows.dim() == 2:
        rows = rows.unsqueeze(0)
    preds, ls = sr.forward(hist, rows[..., :seq_len], rows[..., seq_len:])
    loss = sr.loss(preds, ls, fut)
    return loss if per_sample else loss.mean()


def learn(args):
    data = TimeSeriesData(args.root_path, args.dataset, args.seq_len, args.pred_len)
    model = load_chronos2(threads=args.threads)
    sr = SparseRows(model, args.history, args.seq_len, args.pred_len)
    ch = data.channels.index(args.channel)
    pool, hist, fut = train_queries(data, ch, args.history, args.pool_stride)
    hist_t, fut_t = torch.as_tensor(hist), torch.as_tensor(fut)
    n, win = len(pool), args.seq_len + args.pred_len

    g = torch.Generator().manual_seed(args.seed)
    z = torch.randn(args.latent_rows, win, generator=g).requires_grad_()
    opt = (torch.optim.SGD([z], lr=args.lr) if args.optimizer == "sgd"
           else torch.optim.Adam([z], lr=args.lr))
    with torch.no_grad():
        no_ref = sum(float(batch_loss(sr, hist_t[lo:lo + 16], fut_t[lo:lo + 16],
                                      torch.zeros(1, 0, win), args.seq_len, True).sum())
                     for lo in range(0, n, 16)) / n
    print(f"{args.channel} no-reference train loss {no_ref:.4f}", flush=True)
    ckpts = [z.detach().clone()]
    log = []
    rng = np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        order = rng.permutation(n)
        total = 0.0
        for lo in range(0, n, args.batch_size):
            b = order[lo:lo + args.batch_size]
            opt.zero_grad()
            for mlo in range(0, len(b), args.micro_batch):
                mb = b[mlo:mlo + args.micro_batch]
                loss = batch_loss(sr, hist_t[mb], fut_t[mb], z, args.seq_len) * len(mb) / len(b)
                loss.backward()
                total += loss.item() * len(b)
            opt.step()
        ckpts.append(z.detach().clone())
        log.append(total / n)
        print(f"{args.channel} epoch {epoch + 1}: train loss {total / n:.4f}", flush=True)

    os.makedirs(args.work_dir, exist_ok=True)
    np.save(os.path.join(args.work_dir, f"latent_{args.channel}.npy"),
            torch.stack(ckpts).numpy())
    with open(os.path.join(args.work_dir, f"learn_{args.channel}.json"), "w") as f:
        json.dump({**vars(args), "train_loss": log, "no_ref_loss": no_ref, "n_pool": n}, f, indent=1)
    if args.skip_grads:
        return
    grads = np.zeros((n, len(ckpts), args.latent_rows * win), dtype=np.float32)
    for e, ze in enumerate(ckpts):
        for lo in range(0, n, args.grad_batch):
            hi = min(lo + args.grad_batch, n)
            zr = ze.unsqueeze(0).expand(hi - lo, -1, -1).clone().requires_grad_()
            loss = batch_loss(sr, hist_t[lo:hi], fut_t[lo:hi], zr, args.seq_len, per_sample=True)
            loss.sum().backward()
            grads[lo:hi, e] = zr.grad.reshape(hi - lo, -1).numpy()
        print(f"{args.channel} gradients at checkpoint {e}/{len(ckpts) - 1}", flush=True)

    np.save(os.path.join(args.work_dir, f"grads_{args.channel}.npy"), grads)


def match(grads: np.ndarray, k: int, swaps: int, sign: float = 1.0) -> np.ndarray:
    """Greedy then single-swap search for the subset whose mean gradient matches
    the full mean (sign 1) or is farthest from it (sign -1, the mismatch ablation)."""
    g = grads.astype(np.float64)
    target = g.mean(0)
    n = len(g)
    chosen, s = [], np.zeros_like(target)
    free = np.ones(n, dtype=bool)
    for i in range(1, k + 1):
        d = (((s[None] + g) / i - target[None]) ** 2).sum(1)
        d = np.where(free, sign * d, np.inf)
        j = int(np.argmin(d))
        chosen.append(j)
        free[j] = False
        s += g[j]
    sq = (g ** 2).sum(1)
    for _ in range(swaps):
        r = k * target - s
        cur = sign * (r @ r)
        gs = g[chosen]
        # residual after swapping out chosen a for candidate c: r + g_a - g_c
        cross = gs @ g.T
        val = (r @ r + sq[chosen][:, None] + sq[None, :] - 2 * cross
               + 2 * (r @ gs.T)[:, None] - 2 * (r @ g.T)[None, :])
        val = np.where(free[None, :], sign * val, np.inf)
        a, c = np.unravel_index(np.argmin(val), val.shape)
        if val[a, c] >= cur:
            break
        s += g[c] - g[chosen[a]]
        free[chosen[a]], free[c] = True, False
        chosen[a] = int(c)
    return np.array(chosen)


def select(args):
    data = TimeSeriesData(args.root_path, args.dataset, args.seq_len, args.pred_len)
    ks = [int(k) for k in args.top_k.split(",")]
    sets = {k: {} for k in ks}
    for name in data.channels:
        grads = np.load(os.path.join(args.work_dir, f"grads_{name}.npy"))
        if args.ckpts == "last":
            grads = grads[:, -1:]
        elif args.ckpts == "first":
            grads = grads[:, :1]
        grads = grads.reshape(len(grads), -1)
        for k in ks:
            sets[k][name] = match(grads, k, args.swaps, -1.0 if args.mismatch else 1.0)
        print(f"{name}: selected {ks}", flush=True)
    tag = "clgmis" if args.mismatch else "clg"
    if args.ckpts != "all":
        tag += f"_{args.ckpts}"
    for k in ks:
        np.savez(os.path.join(args.work_dir, f"{tag}_k{k}.npz"), **sets[k])
    latent = {name: np.load(os.path.join(args.work_dir, f"latent_{name}.npy"))[-1]
              for name in data.channels}
    np.savez(os.path.join(args.work_dir, "latent_final.npz"), **latent)


def bestofn(args):
    """Best of N random sets by mean loss over train windows, skipping any window
    whose future overlaps a window in the set being scored (it would see its answer)."""
    data = TimeSeriesData(args.root_path, args.dataset, args.seq_len, args.pred_len)
    model = load_chronos2(threads=args.threads)
    sr = SparseRows(model, args.history, args.seq_len, args.pred_len)
    ks = [int(k) for k in args.top_k.split(",")]
    sets = {k: {} for k in ks}
    rng = np.random.default_rng(args.seed)
    win = data.win_len
    for ch, name in enumerate(data.channels):
        pool, hist, fut = train_queries(data, ch, args.history, args.pool_stride)
        probe = rng.choice(len(pool), size=min(args.probe, len(pool)), replace=False)
        for k in ks:
            best, best_loss = None, np.inf
            for trial in range(args.n_sets):
                idx = rng.choice(len(pool), size=k, replace=False)
                lo_ = idx[:, None] * args.pool_stride
                q = probe[:, None] * args.pool_stride
                clash = (np.abs(q.T - lo_) < win).any(0)
                keep = probe[~clash]
                rows = torch.as_tensor(pool[idx])
                with torch.no_grad():
                    loss = sum(float(batch_loss(sr, torch.as_tensor(hist[keep[i:i + 32]]),
                                                torch.as_tensor(fut[keep[i:i + 32]]),
                                                rows, args.seq_len, True).sum())
                               for i in range(0, len(keep), 32)) / len(keep)
                if loss < best_loss:
                    best, best_loss = idx, loss
            sets[k][name] = best
            print(f"{name} k={k}: best train loss {best_loss:.4f}", flush=True)
    for k in ks:
        np.savez(os.path.join(args.work_dir, f"bestofn_k{k}.npz"), **sets[k])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("mode", choices=["learn", "select", "bestofn"])
    p.add_argument("--root-path", default="./Datasets/_parquet")
    p.add_argument("--dataset", default="ETTh1")
    p.add_argument("--channel", default="OT")
    p.add_argument("--seq-len", default=96, type=int)
    p.add_argument("--pred-len", default=64, type=int)
    p.add_argument("--history", default=2032, type=int)
    p.add_argument("--pool-stride", default=8, type=int)
    p.add_argument("--latent-rows", default=8, type=int)
    p.add_argument("--epochs", default=10, type=int)
    p.add_argument("--batch-size", default=64, type=int)
    p.add_argument("--optimizer", default="adam", choices=["sgd", "adam"])
    p.add_argument("--lr", default=0.05, type=float)
    p.add_argument("--micro-batch", default=8, type=int,
                   help="queries per backward pass; gradients accumulate up to --batch-size")
    p.add_argument("--grad-batch", default=8, type=int)
    p.add_argument("--skip-grads", action="store_true", help="train the latent rows only")
    p.add_argument("--top-k", default="8,32,128,256")
    p.add_argument("--swaps", default=32, type=int)
    p.add_argument("--mismatch", action="store_true")
    p.add_argument("--ckpts", default="all", choices=["all", "last", "first"])
    p.add_argument("--n-sets", default=5, type=int)
    p.add_argument("--probe", default=256, type=int)
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--threads", default=0, type=int)
    p.add_argument("--work-dir", default="./results/clg")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    {"learn": learn, "select": select, "bestofn": bestofn}[a.mode](a)
