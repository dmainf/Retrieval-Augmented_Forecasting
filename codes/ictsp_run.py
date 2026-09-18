"""Train and evaluate ICTSP, writing the same prediction table as predict.py.

Unlike the Chronos runs there is a training stage here: ICTSP has no pretrained
weights, so the backbone is fitted on the train split of each dataset. That is
the price of changing the token format.

The evaluation windows are chosen to line up exactly with predict.py -- window r
of a split forecasts absolute positions [base + r*stride + 96, +64), so the
output parquet drops straight into evaluate.py beside the Chronos results.

  python3 codes/ictsp_run.py --dataset ETTh1 --run-name ictsp_ETTh1
  python3 codes/ictsp_run.py --load checkpoints/ictsp_ETTh1.pt \\
      --example-ablation shuffle-future --run-name ictsp_shuffle

The second form is the one that matters: it asks whether the model actually uses
the context-to-future pairing that this token format makes available.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from dataset import SPLITS, TimeSeriesData
from ictsp import (ABLATIONS, ICTSP, QUANTILES, ablate, build_tokens,
                   context_starts, count_params, quantile_loss)

SEQ_LEN = 96  # only sets where a window's future begins, matching predict.py


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Series:
    """Standardized values plus the origins each split is allowed to forecast."""

    def __init__(self, data: TimeSeriesData, hist_len: int, pred_len: int):
        starts, ends = data.borders
        train = data.values[starts[0]:ends[0]]
        self.mean = train.mean(axis=0)
        self.std = np.where(train.std(axis=0) < 1e-6, 1.0, train.std(axis=0))
        self.values = ((data.values - self.mean) / self.std).astype(np.float32)
        self.channels = data.channels
        self.borders = data.borders
        self.hist_len = hist_len
        self.pred_len = pred_len

    def train_origins(self) -> np.ndarray:
        """Forecast origins whose history and future both stay inside train."""
        start, end = self.borders[0][0], self.borders[1][0]
        lo = max(start + self.hist_len, self.hist_len)
        return np.arange(lo, end - self.pred_len + 1, dtype=np.int64)

    def eval_origins(self, split: str, stride: int) -> tuple[np.ndarray, np.ndarray]:
        """(origins, window starts) matching predict.py's window numbering.

        History may reach back across the split boundary: at test time every
        earlier value really is observed, and the split only decides which
        windows get scored.
        """
        i = SPLITS[split]
        base, end = self.borders[0][i], self.borders[1][i]
        win = SEQ_LEN + self.pred_len
        n = (end - base - win) // stride + 1
        rows = np.arange(n, dtype=np.int64) * stride
        return base + rows + SEQ_LEN, rows

    def batch(self, origins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        back = np.arange(-self.hist_len, 0, dtype=np.int64)
        fwd = np.arange(self.pred_len, dtype=np.int64)
        hist = self.values[origins[:, None] + back[None, :]]      # (B, hist, C)
        future = self.values[origins[:, None] + fwd[None, :]]      # (B, H, C)
        return hist.transpose(0, 2, 1), future.transpose(0, 2, 1)


def forward_batch(model, series, origins, starts, ablation, device):
    hist, future = series.batch(origins)
    tokens = build_tokens(torch.from_numpy(hist).to(device),
                          model.lookback, model.pred_len, starts)
    tokens = ablate(tokens, ablation, model.lookback)
    return model(tokens), torch.from_numpy(future).to(device)


@torch.no_grad()
def evaluate_split(model, series, split, stride, starts, ablation, device,
                   batch_size):
    model.eval()
    origins, rows = series.eval_origins(split, stride)
    preds, total, seen = [], 0.0, 0
    for lo in range(0, len(origins), batch_size):
        chunk = origins[lo:lo + batch_size]
        out, true = forward_batch(model, series, chunk, starts, ablation, device)
        total += quantile_loss(out, true, model.quantiles).item() * len(chunk)
        seen += len(chunk)
        preds.append(out.float().cpu().numpy())
    return np.concatenate(preds, axis=0), rows, total / max(seen, 1)


def train(model, series, args, starts, device):
    origins = series.train_origins()
    print(f"training on {len(origins)} origins, {count_params(model):,} parameters")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    rng = np.random.default_rng(args.seed)

    best, best_state, running, t0 = float("inf"), None, 0.0, time.time()
    for step in range(1, args.steps + 1):
        model.train()
        picked = rng.choice(origins, size=args.batch_size, replace=False)
        out, true = forward_batch(model, series, picked, starts, "none", device)
        loss = quantile_loss(out, true, model.quantiles)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        running += loss.item()

        if step % args.val_every == 0:
            _, _, val = evaluate_split(model, series, "val", args.val_stride,
                                       starts, "none", device, args.batch_size)
            mark = ""
            if val < best:
                best, mark = val, "  *"
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"  step {step:5d}  train {running / args.val_every:.4f}  "
                  f"val {val:.4f}  {time.time() - t0:5.0f}s{mark}")
            running = 0.0

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"best val quantile loss {best:.4f}")
    return best


def write_table(preds, rows, series, model, split, out_dir, stem, args, val_loss):
    """Same schema as predict.py: one row per (channel, window), channel-major."""
    origins, _ = series.eval_origins(split, args.eval_stride)
    fwd = np.arange(model.pred_len, dtype=np.int64)
    frames_meta, pred_blocks, true_blocks = [], [], []
    for c, name in enumerate(series.channels):
        scale, loc = series.std[c], series.mean[c]
        pred_blocks.append(preds[:, c] * scale + loc)              # (n, Q, H)
        raw = series.values[origins[:, None] + fwd[None, :], c]
        true_blocks.append(raw * scale + loc)
        frames_meta.append(pd.DataFrame({
            "channel": name,
            "window": np.arange(len(origins), dtype=np.int64),
            "start": rows,
        }))

    pred = np.concatenate(pred_blocks, axis=0)
    true = np.concatenate(true_blocks, axis=0)
    frames = [pd.concat(frames_meta, ignore_index=True)]
    for i, q in enumerate(model.quantiles):
        frames.append(pd.DataFrame(
            pred[:, i, :], columns=[f"pred_q{q:g}_{t}" for t in range(model.pred_len)]))
    frames.append(pd.DataFrame(
        true, columns=[f"true_{t}" for t in range(model.pred_len)]))

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{stem}.parquet")
    pd.concat(frames, axis=1).to_parquet(path, compression="zstd", index=False)
    print(f"saved {path}  ({len(pred)} rows)")

    args_path = os.path.join(out_dir, f"{stem}_args.json")
    with open(args_path, "w") as f:
        json.dump({**vars(args), "model": model.config,
                   "channels_used": series.channels,
                   "n_context_tokens": None, "val_quantile_loss": val_loss}, f, indent=2)
    print(f"saved {args_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root-path", default="./Datasets/_parquet")
    p.add_argument("--dataset", default="ETTh1")
    p.add_argument("--channels", default="")
    p.add_argument("--eval-split", default="test", choices=list(SPLITS))
    p.add_argument("--eval-stride", default=16, type=int)
    p.add_argument("--val-stride", default=64, type=int,
                   help="coarser stride for the validation checks during training")

    p.add_argument("--hist-len", default=2032, type=int,
                   help="points of history each sample sees; context tokens are "
                        "cut from it. Matches the Chronos runs' 2032 budget")
    p.add_argument("--lookback", default=96, type=int,
                   help="L_b, the input half of a token. 96 + 64 = the 160-point "
                        "window the retrieval database already uses")
    p.add_argument("--pred-len", default=64, type=int, help="L_P, forecast horizon")
    p.add_argument("--token-step", default=32, type=int,
                   help="m: sample one context token every m steps. Smaller means "
                        "more examples and more compute")

    p.add_argument("--d-model", default=128, type=int)
    p.add_argument("--n-heads", default=8, type=int)
    p.add_argument("--n-layers", default=3, type=int)
    p.add_argument("--d-ff", default=256, type=int)
    p.add_argument("--dropout", default=0.1, type=float)
    p.add_argument("--max-tokens", default=512, type=int)

    p.add_argument("--steps", default=3000, type=int)
    p.add_argument("--batch-size", default=16, type=int)
    p.add_argument("--lr", default=3e-4, type=float)
    p.add_argument("--val-every", default=250, type=int)

    p.add_argument("--example-ablation", default="none", choices=ABLATIONS,
                   help="damage the context tokens at evaluation time. "
                        "shuffle-future permutes the future halves among the "
                        "context tokens, breaking only the pairing")
    p.add_argument("--load", default="", help="skip training, evaluate this checkpoint")
    p.add_argument("--checkpoint", default="", help="where to write the trained weights")
    p.add_argument("--output-dir", default="./results")
    p.add_argument("--run-name", default="ictsp")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--seed", default=2021, type=int)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)

    data = TimeSeriesData(args.root_path, args.dataset, SEQ_LEN, args.pred_len,
                          args.channels)
    series = Series(data, args.hist_len, args.pred_len)
    starts = context_starts(args.hist_len, args.lookback, args.pred_len,
                            args.token_step)

    model = ICTSP(lookback=args.lookback, pred_len=args.pred_len,
                  d_model=args.d_model, n_heads=args.n_heads,
                  n_layers=args.n_layers, d_ff=args.d_ff, dropout=args.dropout,
                  max_tokens=args.max_tokens, quantiles=QUANTILES).to(device)

    print(f"{args.dataset}: {len(series.channels)} channels, device={device}")
    print(f"history={args.hist_len}, token={args.lookback}+{args.pred_len}, "
          f"step={args.token_step} -> {len(starts)} context tokens/series "
          f"({len(starts) + 1} incl. target, x{len(series.channels)} channels = "
          f"{(len(starts) + 1) * len(series.channels)} tokens)")

    val_loss = None
    if args.load:
        blob = torch.load(args.load, map_location=device, weights_only=False)
        model.load_state_dict(blob["state"])
        val_loss = blob.get("val")
        print(f"loaded {args.load} (val {val_loss})")
    else:
        val_loss = train(model, series, args, starts, device)
        path = args.checkpoint or f"checkpoints/{args.run_name}.pt"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"state": model.state_dict(), "config": model.config,
                    "val": val_loss, "token_step": args.token_step,
                    "hist_len": args.hist_len}, path)
        print(f"saved {path}")

    preds, rows, loss = evaluate_split(model, series, args.eval_split,
                                       args.eval_stride, starts,
                                       args.example_ablation, device,
                                       args.batch_size)
    print(f"{args.eval_split} quantile loss (standardized) {loss:.4f}"
          f"  ablation={args.example_ablation}")
    write_table(preds, rows, series, model, args.eval_split, args.output_dir,
                args.run_name, args, val_loss)


if __name__ == "__main__":
    main()
