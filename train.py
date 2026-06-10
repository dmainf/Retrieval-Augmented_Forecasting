"""Training entry point.

  python3 train.py --method cross_raf --dataset ETTh1 --root-path ./datasets/ETT-small/ --data-path ETTh1.csv
  python3 train.py --method raf --raf-mode advanced --dataset weather ...

For methods that need no training (``none`` and ``raf --raf-mode naive``) this
script does nothing but tell you to run evaluate.py directly.
"""
import _bootstrap  # noqa: F401  (must precede torch/faiss; sets OpenMP safety)

import json
import os
import random
import time

import numpy as np
import torch
import torch.optim as optim

from config import train_args
from data.loaders import build_tsdata, window_loader, build_retriever
from methods.registry import get_method
from utils.tools import pick_device


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = train_args()
    set_seed(args.seed)
    device = pick_device(args.gpu)
    print(f"Device: {device} | method={args.method}"
          + (f" ({args.raf_mode})" if args.method == "raf" else ""))

    method = get_method(args)
    if not method.needs_training:
        print(f"[{args.method}] requires no training. Run evaluate.py directly.")
        return

    tsd = build_tsdata(args)
    model = method.build_model(device)
    if method.needs_retrieval:
        method.set_retriever(build_retriever(tsd, args, device))

    train_loader = window_loader(tsd, args, "train", stride=args.train_stride, shuffle=True)
    n_trainable = sum(p.numel() for p in method.trainable_parameters(model) if p.requires_grad)
    print(f"Data: {args.dataset} | channels={len(tsd.channels)} | "
          f"train_windows={len(train_loader.dataset)} | trainable_params={n_trainable}")

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.tmax, eta_min=1e-8)

    save_dir = os.path.join(args.output_dir, f"{args.method}_{args.dataset}")
    os.makedirs(save_dir, exist_ok=True)
    # human-readable record of the run's configuration
    with open(os.path.join(save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    def save_ckpt(path, step):
        # self-describing + resumable: weights, optimizer/scheduler state, step, args
        torch.save({
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "args": vars(args),
        }, path)

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            start_step = ckpt.get("step", 0)
        else:  # backward-compat: a raw state_dict (weights only)
            model.load_state_dict(ckpt)
            print("[warning] checkpoint has no optimizer/step; resuming weights only.")
        print(f"Resumed from {args.resume} at step {start_step}")

    model.train()
    step = start_step
    t0 = time.time()
    losses = []
    stop = False
    for epoch in range(args.epochs):
        if stop:
            break
        for context, target in train_loader:
            loss = method.compute_loss(model, context, target, device)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.grad_clip)
            optimizer.step()

            losses.append(loss.item())
            step += 1
            if step % 50 == 0:
                avg = sum(losses[-50:]) / min(len(losses), 50)
                print(f"  [epoch {epoch}][step {step}/{args.train_steps}] "
                      f"loss={avg:.5f} lr={optimizer.param_groups[0]['lr']:.2e}")
            if step % args.save_freq == 0:
                scheduler.step()  # step before snapshot so resume continues the LR schedule exactly
                path = os.path.join(save_dir, f"model_step{step}.pth")
                save_ckpt(path, step)
                print(f"  saved {path}")
            if step >= args.train_steps:
                stop = True
                break

    final = os.path.join(save_dir, "best.pth")
    save_ckpt(final, step)
    print(f"Training done in {time.time()-t0:.1f}s. Saved {final}")


if __name__ == "__main__":
    main()
