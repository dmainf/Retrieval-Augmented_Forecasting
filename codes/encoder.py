"""A window encoder trained so that closeness in its space means a similar future.

Every retrieval method in the RAF literature ranks database windows by context
similarity and then concatenates their futures, which only helps to the extent
that the two orderings agree. `diagnose_retrieval.py` measures that agreement,
and on ETTh1 it is +0.49 once level and scale are divided out -- +0.12 on the
target column. The oracle runs put a number on what that costs: retrieving on
the true future beats no retrieval at all by 19.5%, while retrieving on context
loses to it.

So the fix is to stop hand-designing the distance and learn one. Positives are
defined by the *outcome*, not by the input: two windows are close when their
futures are correlated, whatever their contexts happen to look like. The encoder
only ever sees contexts, so it stays usable at inference; the futures appear only
in the training target.

  target_ij = pearson(y_i, y_j)          futures, only seen during training
  pred_ij   = cos(f(x_i), f(x_j))        contexts, what the retriever will use
  loss      = KL( softmax(target / t_t) || softmax(pred / t_p) )

Soft targets rather than binary positive/negative: "how similar" is a continuous
quantity here and thresholding it throws that away. This is the loss FASCL uses
for cross-sectional asset retrieval (arXiv:2602.10711), pointed at temporal
retrieval instead.

Contexts are z-scored per window, so the encoder matches on shape and cannot
fall back on level -- which is what the raw euclidean baseline is mostly doing.

Training only ever touches the retrieval split.

  python3 codes/encoder.py --dataset ETTh1
  python3 codes/predict.py --encoder learned --encoder-path checkpoints/ETTh1_enc.pt
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import TimeSeriesData

EPS = 1e-8


def zscore(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + EPS)


class WindowEncoder(nn.Module):
    """Dilated 1D conv stack over one z-scored context -> a unit embedding.

    The dilations reach the full 96-point context by the last layer, and mean
    plus max pooling keeps both the average shape and its sharpest excursion,
    which is the part that matters for the spikes this project is after.

    Kept deliberately small. Windows are cut at stride 1, so 8481 of them per
    channel hold only about 8481 / 96 independent samples; a wider net memorizes
    the training futures within a few hundred steps.
    """

    def __init__(self, seq_len: int = 96, embed_dim: int = 16, width: int = 32):
        super().__init__()
        self.seq_len, self.embed_dim, self.width = seq_len, embed_dim, width
        layers, c_in = [], 1
        for dilation in (1, 2, 4, 8):
            layers += [nn.Conv1d(c_in, width, 5, padding=2 * dilation, dilation=dilation),
                       nn.GELU()]
            c_in = width
        self.net = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(2 * width, 2 * width), nn.GELU(),
                                  nn.Linear(2 * width, embed_dim))

    @property
    def config(self) -> dict:
        return dict(seq_len=self.seq_len, embed_dim=self.embed_dim, width=self.width)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """context (B, seq_len) at raw scale -> (B, embed_dim), L2-normalized."""
        h = self.net(zscore(context).unsqueeze(1))
        h = torch.cat([h.mean(-1), h.amax(-1)], dim=-1)
        return F.normalize(self.head(h), dim=-1)


def future_correlation(future: torch.Tensor) -> torch.Tensor:
    """(B, pred_len) -> (B, B) Pearson correlation between every pair of futures."""
    y = future - future.mean(-1, keepdim=True)
    y = y / (y.norm(dim=-1, keepdim=True) + EPS)
    return y @ y.T


def soft_contrastive_loss(z: torch.Tensor, future: torch.Tensor,
                          tau_pred: float, tau_target: float) -> torch.Tensor:
    b = z.shape[0]
    off = ~torch.eye(b, dtype=torch.bool, device=z.device)
    mask = torch.zeros(b, b, device=z.device).masked_fill(~off, float("-inf"))

    target = F.softmax(future_correlation(future) / tau_target + mask, dim=-1)
    pred = F.log_softmax(z @ z.T / tau_pred + mask, dim=-1)
    # index the off-diagonal rather than multiplying it out: the masked diagonal
    # is 0 * -inf, which is NaN even though its gradient is well defined
    return -(target[off].view(b, b - 1) * pred[off].view(b, b - 1)).sum(-1).mean()


def spearman_context_future(model, windows: np.ndarray, seq_len: int,
                            device, sample: int = 400, seed: int = 0,
                            repeats: int = 3) -> tuple:
    """Rank correlation between embedding distance and future distance.

    The same quantity `diagnose_retrieval.py` reports, so a learned encoder can
    be compared against the raw baseline on the axis that actually matters.
    Returns (learned, raw) so the two are always read together.
    """
    rng = np.random.default_rng(seed)
    learned, raw = [], []
    for _ in range(repeats):
        idx = rng.choice(len(windows), min(sample, len(windows)), replace=False)
        w = torch.from_numpy(windows[idx]).to(device)
        ctx, fut = w[:, :seq_len], w[:, seq_len:]

        with torch.no_grad():
            z = model(ctx)
        d_fut = (1.0 - _cos(fut)).cpu().numpy()
        learned.append(_mean_rank_corr((1.0 - z @ z.T).cpu().numpy(), d_fut))
        raw.append(_mean_rank_corr((1.0 - _cos(zscore(ctx))).cpu().numpy(), d_fut))
    return float(np.mean(learned)), float(np.mean(raw))


def _cos(x: torch.Tensor) -> torch.Tensor:
    xc = x - x.mean(-1, keepdim=True)
    xn = xc / (xc.norm(dim=-1, keepdim=True) + EPS)
    return xn @ xn.T


def _mean_rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    n = len(a)
    keep = ~np.eye(n, dtype=bool)
    ra = np.argsort(np.argsort(np.where(keep, a, np.inf), axis=1), axis=1).astype(float)
    rb = np.argsort(np.argsort(np.where(keep, b, np.inf), axis=1), axis=1).astype(float)
    ra, rb = ra[keep].reshape(n, -1), rb[keep].reshape(n, -1)
    ra -= ra.mean(1, keepdims=True)
    rb -= rb.mean(1, keepdims=True)
    num = (ra * rb).sum(1)
    den = np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1)) + EPS
    return float((num / den).mean())


def train(args):
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    data = TimeSeriesData(args.root_path, args.dataset, args.seq_len, args.pred_len,
                          args.channels)
    train_w = [data.windows(args.train_split, c, args.stride)
               for c in range(data.n_channels)]
    val_w = [data.windows(args.val_split, c, args.stride)
             for c in range(data.n_channels)]
    print(f"{args.dataset}: {data.n_channels} channels x "
          f"{len(train_w[0])} train windows, device={device}")

    model = WindowEncoder(args.seq_len, args.embed_dim, args.width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)

    tensors = [torch.from_numpy(w).to(device) for w in train_w]
    best, best_state, best_step = -np.inf, None, 0
    running = 0.0
    for step in range(1, args.steps + 1):
        # one channel per batch: a future-correlation target is only meaningful
        # between windows the retriever would ever compare, and retrieval is
        # per channel. The weights are still shared across channels.
        w = tensors[rng.integers(data.n_channels)]
        pick = torch.from_numpy(
            rng.choice(len(w), args.batch_size, replace=False)).to(device)
        batch = w[pick]

        z = model(batch[:, :args.seq_len])
        loss = soft_contrastive_loss(z, batch[:, args.seq_len:],
                                     args.tau_pred, args.tau_target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        running += loss.item()
        if step % args.log_every == 0:
            model.eval()
            rhos = [spearman_context_future(model, v, args.seq_len, device,
                                            seed=args.seed) for v in val_w]
            model.train()
            learned = float(np.mean([r[0] for r in rhos]))
            raw = float(np.mean([r[1] for r in rhos]))
            mark = ""
            if learned > best:
                best, best_step = learned, step
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                mark = "  <- best"
            print(f"  step {step:5d}  loss {running / args.log_every:.4f}   "
                  f"val spearman: learned {learned:+.3f}  vs  zscore-raw {raw:+.3f}{mark}")
            running = 0.0

    # keep the step that generalized best, not the last one: val spearman peaks
    # early and then falls below the untrained baseline as the futures memorize
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  best val spearman {best:+.3f} at step {best_step}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": model.config,
                "dataset": args.dataset, "train_split": args.train_split,
                "val_spearman": best, "step": best_step}, args.out)
    print(f"saved {args.out}")


def load_encoder(path: str):
    """-> (model in eval mode on cpu, config dict). Used by the retriever."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = WindowEncoder(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.requires_grad_(False)
    return model, ckpt["config"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root-path", default="./Datasets/_parquet")
    p.add_argument("--dataset", default="ETTh1")
    p.add_argument("--channels", default="")
    p.add_argument("--seq-len", default=96, type=int)
    p.add_argument("--pred-len", default=64, type=int)
    p.add_argument("--train-split", default="train",
                   help="must match --retrieval-split at predict time")
    p.add_argument("--val-split", default="val")
    p.add_argument("--stride", default=1, type=int)
    p.add_argument("--embed-dim", default=16, type=int)
    p.add_argument("--width", default=32, type=int)
    p.add_argument("--batch-size", default=256, type=int)
    p.add_argument("--steps", default=2000, type=int)
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--weight-decay", default=1e-2, type=float)
    p.add_argument("--tau-pred", default=0.1, type=float,
                   help="temperature on the embedding cosine")
    p.add_argument("--tau-target", default=0.1, type=float,
                   help="temperature on the future correlation; smaller = peakier "
                        "target, closer to hard positives")
    p.add_argument("--log-every", default=100, type=int)
    p.add_argument("--seed", default=2021, type=int)
    p.add_argument("--out", default="")
    args = p.parse_args()
    if not args.out:
        args.out = f"./checkpoints/{args.dataset}_enc.pt"
    train(args)


if __name__ == "__main__":
    main()
