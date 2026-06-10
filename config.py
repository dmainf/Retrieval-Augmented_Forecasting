"""Unified argparse configuration for train.py and evaluate.py.

A single ``--method`` switch selects one of three retrieval-augmented forecasting
setups that share the same Chronos-Bolt backbone:

    none       : plain frozen Chronos (zero-shot, no retrieval)
    raf        : context-augmented RAF (Tire et al. 2026); naive=frozen / advanced=fine-tune
    cross_raf  : Cross-RAG (cross + self attention fusion of top-k retrievals)
"""
import argparse


def add_common_args(p: argparse.ArgumentParser):
    # method
    p.add_argument("--method", default="none", choices=["none", "raf", "cross_raf"],
                   help="forecasting setup to use")
    p.add_argument("--raf-mode", default="naive", choices=["naive", "advanced"],
                   help="(--method raf) naive: frozen backbone / advanced: fine-tune backbone")

    # data
    p.add_argument("--root-path", default="./Datasets/ETT-small/", help="directory holding the CSV")
    p.add_argument("--data-path", default="ETTh1.csv", help="CSV filename")
    p.add_argument("--dataset", default="ETTh1", help="dataset name (controls train/val/test borders)")
    p.add_argument("--features", default="M", choices=["M", "S"],
                   help="M: every column as an independent series / S: only --target")
    p.add_argument("--target", default="OT", help="target column when --features S")
    p.add_argument("--seq-len", default=512, type=int, help="context length")
    p.add_argument("--pred-len", default=64, type=int, help="forecast horizon")
    p.add_argument("--no-scale", action="store_true", help="disable per-channel standardization")

    # backbone
    p.add_argument("--chronos-model", default="amazon/chronos-bolt-small",
                   help="HF id or local path of the Chronos-Bolt backbone")

    # retrieval
    p.add_argument("--top-k", default=15, type=int,
                   help="retrieved neighbours (cross_raf fusion; raf concatenates k motifs, "
                        "auto-capped to the backbone context; use small k e.g. 1-3 for raf)")
    p.add_argument("--retrieval-metric", default="cosine",
                   choices=["cosine", "euclidean", "correlation"],
                   help="faiss similarity: cosine / euclidean (L2) / correlation (Pearson)")
    p.add_argument("--retrieval-split", default="train",
                   help="split used to build the retrieval database")
    p.add_argument("--retrieval-stride", default=1, type=int,
                   help="sliding-window stride for the retrieval database "
                        "(1 = paper default / full KB; larger = fewer windows, less memory)")

    # cross_raf fusion
    p.add_argument("--augment-mode", default="moe", help="cross_raf fusion mode (moe = cross+self)")
    p.add_argument("--mix-lambda", default=0.7, type=float, help="cross_raf cross/self mix weight")

    # io / runtime
    p.add_argument("--output-dir", default="./checkpoints", help="checkpoint / result directory")
    p.add_argument("--gpu", default=0, type=int)
    p.add_argument("--batch-size", default=256, type=int)
    p.add_argument("--num-workers", default=2, type=int)
    p.add_argument("--seed", default=2021, type=int)
    return p


def train_args():
    p = argparse.ArgumentParser(description="RAF / Cross-RAF / Chronos — training")
    add_common_args(p)
    p.add_argument("--train-steps", default=10000, type=int,
                   help="number of optimizer steps (Cross-RAG paper Table A.2: 10,000). "
                        "Training is step-driven: the loader is cycled until this many steps.")
    p.add_argument("--lr", default=3e-4, type=float,
                   help="constant learning rate (Cross-RAG paper Table A.2: 3e-4, no scheduler; "
                        "for raf --raf-mode advanced fine-tuning the full backbone, try 1e-5~1e-4)")
    p.add_argument("--weight-decay", default=0.01, type=float)
    p.add_argument("--grad-clip", default=1.0, type=float)
    p.add_argument("--train-stride", default=1, type=int, help="sliding-window stride for train windows")
    p.add_argument("--save-freq", default=2000, type=int, help="checkpoint every N steps")
    p.add_argument("--resume", default="", type=str)
    return p.parse_args()


def eval_args():
    p = argparse.ArgumentParser(description="RAF / Cross-RAF / Chronos — evaluation")
    add_common_args(p)
    p.add_argument("--checkpoint", default="", help="trained weights (cross_raf / advanced raf)")
    p.add_argument("--eval-stride", default=None, type=int,
                   help="test-window stride (default: pred_len for non-overlapping)")
    p.add_argument("--result-file", default="result.txt")
    return p.parse_args()
