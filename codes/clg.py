"""Curriculum latent gradients (Zhang et al., ACL Findings 2025) for Chronos-2.

K latent example windows Z (seq_len + pred_len values each, the same shape as a
real example) are trained on the train windows of one channel with Chronos-2
frozen. Every train window, taken as a query (its history in, its future as the
target), then yields the gradient of its loss with respect to Z at the initial
Z and after each epoch; concatenated, that is its curriculum latent gradient.
selection.py picks the windows whose mean gradient matches the mean over all.

Z = mean + std * U with the channel's train mean and std, and U is what trains
and what gradients are taken against, so one learning rate fits every channel.
U starts from N(0, 1), as the original starts its prefix from a random init.
The loss is Chronos-2's own, in the scale the model normalizes its input to; for
clg that input includes Z, as in the concat container.

Parallel-CLG is meant for the parallel container, where many-shot sets fit; the
selected windows can still be passed with --fusion concat (up to 35).

The two methods differ only in where Z sits, i.e. which attention carries the
signal:

  clg    Z is spliced in front of the history along the time axis, separated as
         in the concat container -- the prefix of the original method, read
         through time attention
  pclg   Z is K covariate rows beside the history, as in the parallel container
         -- read through group attention, the way real examples are read (Parallel-CLG)

Both methods start from the same U. Hyperparameters are the CLG_* constants in run.py.

  python3 codes/clg.py --method pclg
  python3 codes/clg.py --method clg --channels OT
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from chronos2 import Chronos2Rows, load_chronos2
from dataset import TimeSeriesData
from run import (CLG_BATCH, CLG_EPOCHS, CLG_LATENT, CLG_LR, CLG_MICRO_BATCH, DATASET, DB_STRIDE, DEVICE,
                 ETT_MONTHS, HISTORY, MODEL_NAME, PRED_LEN, REPORT_QUANTILES, SEED, SEP_LEN,
                 SEQ_LEN, TEST_FRAC, TOKEN_BUDGET, TRAIN_FRAC, channel_rng)

METHODS = ("clg", "pclg")


def grads_path(output_dir: str, dataset: str, method: str, seed: int, channel: str) -> str:
    return os.path.join(output_dir, "clg", f"{dataset}_{method}_s{seed}_{channel}.npz")


def serial(history: torch.Tensor, z: torch.Tensor, sep_len: int) -> torch.Tensor:
    """history (B, H), z (K, W) or (B, K, W) -> (B, K * (W + sep_len) + H), laid out
    like concat_context: [z_K | sep] ... [z_1 | sep] [history], sep the query mean."""
    b = history.shape[0]
    z = z.expand(b, -1, -1) if z.dim() == 2 else z
    sep = torch.nanmean(history, 1, keepdim=True).expand(-1, sep_len)
    parts = []
    for j in reversed(range(z.shape[1])):
        parts += [z[:, j], sep]
    return torch.cat(parts + [history], 1)


def per_query_loss(rows, method, hist, fut, z, s):
    if method == "pclg":
        z = z.unsqueeze(0) if z.dim() == 2 else z
        preds, ls = rows.forward(hist, z[..., :s], z[..., s:])
        return rows.loss(preds, ls, fut)
    lead = torch.isnan(hist).to(torch.int32).cumprod(1).sum(1)
    if not lead.any():
        return rows.loss(*rows.forward(serial(hist, z, SEP_LEN)), fut)
    # A history padded with NaN would share the model's instance norm with Z, and
    # the NaN reach Z's gradient through it (0 * NaN); the padding holds no data,
    # so drop it and run those queries one at a time.
    z = z.expand(len(hist), -1, -1) if z.dim() == 2 else z
    return torch.cat([rows.loss(*rows.forward(serial(hist[i:i + 1, n:], z[i:i + 1], SEP_LEN)),
                                fut[i:i + 1]) for i, n in enumerate(lead.tolist())])


def learn_channel(rows, data, ch, method, seed, db_stride, history):
    """Train U on the channel's train windows; return its checkpoints, the train losses,
    the no-reference loss, the (n_windows, epochs + 1, K * W) curriculum latent
    gradients with respect to U, and the (mean, std) that map U to Z."""
    s, name = data.seq_len, data.channels[ch]
    pool = data.windows("train", ch, db_stride)
    starts = np.arange(len(pool), dtype=np.int64) * db_stride
    dev = rows.device
    hist = torch.as_tensor(data.history("train", ch, starts, history), device=dev)
    fut = torch.as_tensor(pool[:, s:], device=dev)
    n = len(pool)
    rng = channel_rng(seed, name)
    lo, hi = data.borders[0][0], data.borders[1][0]
    mean, std = float(data.values[lo:hi, ch].mean()), float(data.values[lo:hi, ch].std())
    to_z = lambda u: mean + std * u

    u = torch.as_tensor(rng.standard_normal((CLG_LATENT, data.win_len)), dtype=torch.float32, device=dev)
    u.requires_grad_()
    opt = torch.optim.Adam([u], lr=CLG_LR)
    with torch.no_grad():
        base = float(sum(rows.loss(*rows.forward(hist[i:i + 16]), fut[i:i + 16]).sum()
                         for i in range(0, n, 16)) / n)
    print(f"  {name} {method}: no-reference train loss {base:.4f}", flush=True)
    ckpts, losses = [u.detach().clone()], []
    for epoch in range(CLG_EPOCHS):
        t0, total = time.time(), 0.0
        order = rng.permutation(n)
        for b0 in range(0, n, CLG_BATCH):
            batch = order[b0:b0 + CLG_BATCH]
            opt.zero_grad()
            for mlo in range(0, len(batch), CLG_MICRO_BATCH):
                mb = batch[mlo:mlo + CLG_MICRO_BATCH]
                loss = per_query_loss(rows, method, hist[mb], fut[mb], to_z(u), s).sum() / len(batch)
                loss.backward()
                total += loss.item() * len(batch)
            opt.step()
        ckpts.append(u.detach().clone())
        losses.append(total / n)
        print(f"  {name} {method} epoch {epoch + 1}/{CLG_EPOCHS}: train loss {total / n:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    grads = np.zeros((n, len(ckpts), u.numel()), dtype=np.float32)
    for e, ue in enumerate(ckpts):
        for b0 in range(0, n, CLG_MICRO_BATCH):
            b1 = min(b0 + CLG_MICRO_BATCH, n)
            ur = ue.unsqueeze(0).expand(b1 - b0, -1, -1).clone().requires_grad_()
            per_query_loss(rows, method, hist[b0:b1], fut[b0:b1], to_z(ur), s).sum().backward()
            grads[b0:b1, e] = ur.grad.reshape(b1 - b0, -1).cpu().numpy()
    return torch.stack(ckpts).cpu().numpy(), np.array(losses), base, grads, (mean, std)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--root-path", default="./Datasets/_parquet")
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--channels", default="", help="comma-separated; empty = every column")
    p.add_argument("--seq-len", default=SEQ_LEN, type=int)
    p.add_argument("--pred-len", default=PRED_LEN, type=int)
    p.add_argument("--history", default=HISTORY, type=int)
    p.add_argument("--db-stride", default=DB_STRIDE, type=int)
    p.add_argument("--seed", default=SEED, type=int)
    p.add_argument("--threads", default=0, type=int)
    p.add_argument("--output-dir", default="./results")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    data = TimeSeriesData(a.root_path, a.dataset, a.seq_len, a.pred_len, a.channels,
                          ett_months=ETT_MONTHS, train_frac=TRAIN_FRAC, test_frac=TEST_FRAC)
    rows = Chronos2Rows(load_chronos2(MODEL_NAME, DEVICE, a.threads), a.seq_len, a.pred_len,
                        REPORT_QUANTILES, TOKEN_BUDGET)
    meta = {"method": a.method, "dataset": a.dataset, "seq_len": a.seq_len, "pred_len": a.pred_len,
            "history": a.history, "db_stride": a.db_stride, "seed": a.seed, "latent": CLG_LATENT,
            "epochs": CLG_EPOCHS, "lr": CLG_LR, "batch": CLG_BATCH, "sep_len": SEP_LEN}
    for ch, name in enumerate(data.channels):
        path = grads_path(a.output_dir, a.dataset, a.method, a.seed, name)
        if os.path.exists(path) and not a.overwrite:
            print(f"{path} exists; skipping (pass --overwrite)", flush=True)
            continue
        t0 = time.time()
        ckpts, losses, base, grads, (mean, std) = learn_channel(rows, data, ch, a.method, a.seed,
                                                                a.db_stride, a.history)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, grads=grads, latent_u=ckpts, z_mean=mean, z_std=std, train_loss=losses,
                 base_loss=base, meta=json.dumps(meta))
        print(f"[{ch + 1}/{data.n_channels}] {name}: saved {path} ({time.time() - t0:.0f}s)",
              flush=True)


if __name__ == "__main__":
    main()
