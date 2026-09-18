"""A learned aggregator: constructed analogue with a learned projection.

The retrieval pipeline has three layers -- which windows to pick, how to turn
their futures into one forecast, and how to combine that with the backbone.
This is the middle one, and it can be trained and evaluated without ever running
the backbone, which makes it fast to iterate on.

Averaging the K nearest futures is a local-constant estimate and carries the
bias J(x_bar - x_q): the neighbours' contexts sit off-centre from the query,
pulled toward the bulk of the corpus, so the averaged future is pulled too.
Softmax attention does not escape it -- its weights are non-negative and sum to
one, so the estimate stays inside the convex hull of the retrieved futures and
remains local-constant. Constructed analogue escapes it by solving for weights
that reconstruct the query context, which may be negative.

This generalizes constructed analogue. A learned projection decides which part
of the context the reconstruction has to match:

    w = Phi (Phi^T Phi + lam I)^-1 phi(x_q),    y_hat = Y^T w

An identity projection recovers constructed analogue exactly. The dual form
keeps the solve at rank x rank instead of pool x pool, so a wide candidate pool
costs almost nothing.

Trained directly on forecast error. That is the point: the earlier contrastive
encoder optimized a proxy ("windows whose futures correlate") and the gain did
not survive to the forecast.

  python3 codes/aggregator.py --dataset ETTh1
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from dataset import TimeSeriesData, SPLITS


class LearnedCA(nn.Module):
    """Solve reconstruction weights in a learned space, apply them to futures."""

    def __init__(self, seq_len: int = 96, rank: int = 32, hidden: int = 128):
        super().__init__()
        self.seq_len, self.rank, self.hidden = seq_len, rank, hidden
        self.proj = nn.Sequential(nn.Linear(seq_len, hidden), nn.GELU(),
                                  nn.Linear(hidden, rank))
        self.log_lam = nn.Parameter(torch.zeros(()))

    @property
    def config(self) -> dict:
        return dict(seq_len=self.seq_len, rank=self.rank, hidden=self.hidden)

    def forward(self, cand_ctx, cand_fut, query_ctx):
        """cand_ctx (B,M,S), cand_fut (B,M,P), query_ctx (B,S) -> (B,P).

        Everything arrives as anomalies; the caller adds the query level back.
        """
        phi = self.proj(cand_ctx)                                  # (B, M, r)
        q = self.proj(query_ctx)                                   # (B, r)
        gram = torch.einsum("bmr,bms->brs", phi, phi)
        scale = torch.diagonal(gram, dim1=1, dim2=2).mean(-1).clamp_min(1e-6)
        eye = torch.eye(self.rank, device=phi.device)
        gram = gram + (torch.exp(self.log_lam) * scale)[:, None, None] * eye
        alpha = torch.linalg.solve(gram, q.unsqueeze(-1)).squeeze(-1)
        w = torch.einsum("bmr,br->bm", phi, alpha)                 # (B, M)
        return torch.einsum("bmp,bm->bp", cand_fut, w)


class Pools:
    """Candidate pools for one split, stored as indices into per-channel arrays."""

    def __init__(self, data, split, db_stride, eval_stride, pool_size):
        V = data.values.astype(np.float32)
        S, P, W = data.seq_len, data.pred_len, data.win_len
        base = data.borders[0][SPLITS[split]]
        pos = np.arange(0, len(V) - W, db_stride)

        self.C, self.F = {}, {}
        idx, qc, qf, ch = [], [], [], []
        for ci in range(data.n_channels):
            self.C[ci] = V[pos[:, None] + np.arange(S)[None, :], ci]
            self.F[ci] = V[pos[:, None] + np.arange(S, W)[None, :], ci]
            q = data.windows(split, ci, eval_stride)
            starts = np.arange(len(q)) * eval_stride
            for i, st in enumerate(starts):
                allowed = np.flatnonzero(pos + W <= base + st)
                if len(allowed) < pool_size + 1:
                    continue
                d = ((self.C[ci][allowed] - q[i, :S]) ** 2).sum(1)
                idx.append(allowed[np.argsort(d)[:pool_size]])
                qc.append(q[i, :S]); qf.append(q[i, S:S + P]); ch.append(ci)
        self.idx = np.stack(idx)
        self.qc = np.stack(qc).astype(np.float32)
        self.qf = np.stack(qf).astype(np.float32)
        self.ch = np.array(ch)
        self.var = {ci: float(self.qf[self.ch == ci].var()) for ci in set(ch)}

    def __len__(self):
        return len(self.idx)

    def take(self, rows, device):
        """-> anomaly candidate contexts/futures, anomaly query, level, variance."""
        cc = np.stack([self.C[self.ch[r]][self.idx[r]] for r in rows])
        cf = np.stack([self.F[self.ch[r]][self.idx[r]] for r in rows])
        mx = cc.mean(-1, keepdims=True)
        qm = self.qc[rows].mean(-1, keepdims=True)
        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
        return (t(cc - mx), t(cf - mx), t(self.qc[rows] - qm), t(qm),
                t(np.array([self.var[c] for c in self.ch[rows]])))


def evaluate(model, pool, device, chunk=256):
    model.eval()
    err = np.zeros(len(pool)); ch = pool.ch
    with torch.no_grad():
        for lo in range(0, len(pool), chunk):
            rows = np.arange(lo, min(lo + chunk, len(pool)))
            cc, cf, qc, qm, var = pool.take(rows, device)
            pred = model(cc, cf, qc) + qm
            e = ((pred - torch.as_tensor(pool.qf[rows], device=device)) ** 2).mean(-1)
            err[rows] = (e / var).cpu().numpy()
    model.train()
    return float(np.mean([err[ch == c].mean() for c in np.unique(ch)]))


def train(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    data = TimeSeriesData(args.root_path, args.dataset, args.seq_len, args.pred_len)
    device = torch.device("cpu")

    pools = {s: Pools(data, s, args.db_stride, args.eval_stride, args.pool_size)
             for s in ("train", "val", "test")}
    print(f"{args.dataset}: pools " +
          " ".join(f"{s}={len(p)}" for s, p in pools.items()) +
          f", pool_size={args.pool_size}")

    model = LearnedCA(args.seq_len, args.rank, args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    tr = pools["train"]

    best, best_state, best_step = np.inf, None, 0
    running = 0.0
    for step in range(1, args.steps + 1):
        rows = rng.choice(len(tr), args.batch_size, replace=False)
        cc, cf, qc, qm, var = tr.take(rows, device)
        pred = model(cc, cf, qc) + qm
        loss = ((((pred - torch.as_tensor(tr.qf[rows])) ** 2).mean(-1)) / var).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        running += loss.item()

        if step % args.log_every == 0:
            v = evaluate(model, pools["val"], device)
            mark = ""
            if v < best:
                best, best_step = v, step
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
                mark = "  <- best"
            print(f"  step {step:5d}  train {running / args.log_every:.4f}  val {v:.4f}{mark}")
            running = 0.0

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\n  best val {best:.4f} at step {best_step}")
    print(f"  test     {evaluate(model, pools['test'], device):.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": model.config,
                "dataset": args.dataset, "val": best}, args.out)
    print(f"  saved {args.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root-path", default="./Datasets/_parquet")
    p.add_argument("--dataset", default="ETTh1")
    p.add_argument("--seq-len", default=96, type=int)
    p.add_argument("--pred-len", default=64, type=int)
    p.add_argument("--db-stride", default=8, type=int)
    p.add_argument("--eval-stride", default=16, type=int)
    p.add_argument("--pool-size", default=200, type=int)
    p.add_argument("--rank", default=32, type=int)
    p.add_argument("--hidden", default=128, type=int)
    p.add_argument("--steps", default=1500, type=int)
    p.add_argument("--batch-size", default=64, type=int)
    p.add_argument("--lr", default=3e-4, type=float)
    p.add_argument("--log-every", default=100, type=int)
    p.add_argument("--seed", default=2021, type=int)
    p.add_argument("--out", default="")
    args = p.parse_args()
    if not args.out:
        args.out = f"./checkpoints/{args.dataset}_agg.pt"
    train(args)


if __name__ == "__main__":
    main()
