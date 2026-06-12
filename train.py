"""Training entry point — Cross-RAF fusion pretraining on the general corpus.

Cross-RAF is zero-shot: the fusion modules are trained **once** on a large,
dataset-agnostic corpus (never on the target), then applied unchanged at eval.
So training takes no ``--dataset``; it streams precomputed pairs and gathers
retrieved windows by their precomputed indices (no faiss search at train time).

  python3 train.py --method cross_raf \
      --corpus-dir ./corpus/pretrain_pairs_ctx512 \
      --retrieval-db-path ./corpus/retrieval_database_512.parquet

``none`` and ``raf`` are zero-shot baselines that are not trained here — run
evaluate.py directly.
"""
import _bootstrap  # noqa: F401  (must precede torch/faiss; sets OpenMP safety)

import json
import os
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from config import train_args
from data.corpus import CorpusPairs, RetrievalDB
from methods.registry import get_method
from utils.tools import pick_device, set_seed


def save_loss_curve(save_dir, steps, losses):
    csv_path = os.path.join(save_dir, "loss_curve.csv")
    with open(csv_path, "w") as f:
        f.write("step,loss\n")
        for s, l in zip(steps, losses):
            f.write(f"{s},{l:.6f}\n")
    print(f"  saved {csv_path}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(steps, losses, lw=0.8)
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title(os.path.basename(save_dir))
        plt.tight_layout()
        png_path = os.path.join(save_dir, "loss_curve.png")
        plt.savefig(png_path, dpi=120)
        plt.close()
        print(f"  saved {png_path}")
    except Exception as e:
        print(f"  [warning] could not plot loss curve: {e}")


def main():
    args = train_args()
    set_seed(args.seed)
    device = pick_device(args.gpu)

    if args.method != "cross_raf":
        print(f"[{args.method}] is a zero-shot baseline and is not trained here. "
              "Run evaluate.py directly.")
        return

    print(f"Device: {device} | Cross-RAF fusion pretraining on general corpus")
    method = get_method(args)
    model = method.build_model(device)  # builds fusion head, freezes backbone

    # retrieval KB (gathered by precomputed indices) + streamed corpus pairs
    db = RetrievalDB(args.retrieval_db_path, seq_len=args.seq_len, pred_len=args.pred_len)
    corpus = CorpusPairs(
        args.corpus_dir,
        context_length=args.seq_len,
        prediction_length=args.pred_len,
        retrieve_lookback_length=args.seq_len,
        top_k=args.top_k,
        drop_prob=args.drop_prob,
    ).shuffle(buffer_len=args.shuffle_buffer)
    train_loader = DataLoader(corpus, batch_size=args.batch_size, num_workers=0)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"KB windows={db.whole_seq.shape[0]} | trainable_params={n_trainable} "
          f"| top_k={args.top_k} | batch_size={args.batch_size}")

    optimizer = optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    save_dir = os.path.join(args.output_dir, "cross_raf_pretrain")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    def save_ckpt(path, step):
        torch.save({
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
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
            start_step = ckpt.get("step", 0)
        else:
            model.load_state_dict(ckpt)
            print("[warning] checkpoint has no optimizer/step; resuming weights only.")
        print(f"Resumed from {args.resume} at step {start_step}")

    model.train()
    t0 = time.time()
    steps, losses = [], []
    step = start_step
    for batch in train_loader:
        if step >= args.train_steps:
            break
        step += 1

        x = batch["x"].float().to(device)
        y = batch["y"].float().to(device)
        dist = batch["distances"].float().to(device)
        retrieved = db.gather(batch["indices"]).float().to(device)  # (B, k, win)

        out = model(context=x, target=y, retrieved_seq=retrieved, distances=dist)
        loss = out.loss.mean()

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()

        steps.append(step)
        losses.append(loss.item())
        if step % 50 == 0:
            avg = sum(losses[-50:]) / min(len(losses), 50)
            print(f"  [step {step}/{args.train_steps}] loss={avg:.5f} "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")
        if step % args.save_freq == 0:
            path = os.path.join(save_dir, f"model_step{step}.pth")
            save_ckpt(path, step)
            print(f"  saved {path}")

    save_loss_curve(save_dir, steps, losses)
    final = os.path.join(save_dir, "best.pth")
    save_ckpt(final, step)
    print(f"Training done in {time.time()-t0:.1f}s. Saved {final}")


if __name__ == "__main__":
    main()
