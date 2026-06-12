"""Zero-shot / trained evaluation entry point.

  python3 evaluate.py --method none      --dataset ETTh1 --root-path ./datasets/ETT-small/ --data-path ETTh1.csv
  python3 evaluate.py --method raf       --dataset ETTh1 ...
  python3 evaluate.py --method cross_raf --dataset ETTh1 --checkpoint ./checkpoints/cross_raf_ETTh1/best.pth ...

Forecasts and targets are compared in the dataset-standardized space; MSE/MAE
are reported and appended to ``--output-dir/--result-file``.
"""
import _bootstrap  # noqa: F401  (must precede torch/faiss; sets OpenMP safety)

import os

import numpy as np
import torch

from config import eval_args
from data.loaders import build_tsdata, window_loader, build_retriever
from methods.registry import get_method
from utils.metrics import metric
from utils.tools import pick_device, save_predictions, save_results, set_seed


def main():
    args = eval_args()
    set_seed(args.seed)
    device = pick_device(args.gpu)
    eval_stride = args.eval_stride if args.eval_stride is not None else args.pred_len
    tag = f"{args.method}" + (f"_{args.raf_mode}" if args.method == "raf" else "")
    print(f"Device: {device} | method={tag} | dataset={args.dataset}")

    method = get_method(args)
    tsd = build_tsdata(args)
    model = method.build_model(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state, train_args = ckpt["state_dict"], ckpt.get("args", {})
        else:  # backward-compat: a raw state_dict
            state, train_args = ckpt, {}
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {args.checkpoint}")
        if train_args:
            keys = ["method", "raf_mode", "chronos_model", "seq_len", "pred_len",
                    "top_k", "retrieval_metric", "augment_mode", "mix_lambda",
                    "dataset", "lr", "train_steps", "seed"]
            print("  trained with: "
                  + ", ".join(f"{k}={train_args[k]}" for k in keys if k in train_args))
            # warn if architecture-critical eval settings differ from training
            for k in ["chronos_model", "seq_len", "pred_len", "augment_mode", "method"]:
                if k in train_args and getattr(args, k) != train_args[k]:
                    print(f"  [warning] {k}: eval={getattr(args, k)} != train={train_args[k]} "
                          "(may load incorrectly or compare unfairly)")
        else:
            print("  [note] checkpoint has no embedded args (old format); "
                  "ensure eval settings match the training run.")
    elif method.needs_training:
        print("[warning] this method is normally trained; no --checkpoint given "
              "(using initial fusion weights).")

    if method.needs_retrieval:
        method.set_retriever(build_retriever(tsd, args, device))

    test_loader = window_loader(tsd, args, "test", stride=eval_stride, shuffle=False)
    print(f"test_windows={len(test_loader.dataset)} | eval_stride={eval_stride}")

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for context, target in test_loader:
            point = method.predict(model, context, device)  # (B, pred_len)
            preds.append(point.cpu().numpy())
            trues.append(target.numpy())
    preds = np.concatenate(preds, 0)
    trues = np.concatenate(trues, 0)

    mae, mse, rmse = metric(preds, trues)
    print(f"\n[{tag}] {args.dataset}  mse={mse:.4f}  mae={mae:.4f}  rmse={rmse:.4f}")

    result_path = os.path.join(args.output_dir, args.result_file)
    chronos_size = args.chronos_model.rstrip("/").split("-")[-1]
    model_id = f"{tag}_{args.dataset}_{chronos_size}"
    save_results(result_path, model_id, mse, mae)
    print(f"Result appended to {result_path}")

    pred_path = os.path.join(args.output_dir, f"{model_id}_preds.csv")
    save_predictions(pred_path, preds, trues)
    print(f"Predictions saved to {pred_path}")


if __name__ == "__main__":
    main()
