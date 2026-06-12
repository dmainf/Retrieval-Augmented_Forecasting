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
    p.add_argument(
        "--method",
        default="none",
        choices=["none", "raf", "cross_raf"],
        help="forecasting setup to use",
    )
    p.add_argument(
        "--raf-mode",
        default="naive",
        choices=["naive", "advanced"],
        help="(--method raf) naive: frozen backbone / advanced: fine-tune backbone",
    )

    # window geometry (shared by train corpus and eval target)
    p.add_argument("--seq-len", default=512, type=int, help="context length")
    p.add_argument("--pred-len", default=64, type=int, help="forecast horizon")

    # backbone
    p.add_argument(
        "--chronos-model",
        default="amazon/chronos-bolt-base",
        help="HF id or local path of the Chronos-Bolt backbone",
    )

    # retrieval
    p.add_argument(
        "--top-k",
        default=15,
        type=int,
        help="retrieved neighbours (cross_raf fusion; raf concatenates k motifs, "
        "auto-capped to the backbone context; use small k e.g. 1-3 for raf)",
    )
    p.add_argument(
        "--retrieval-metric",
        default="cosine",
        choices=["cosine", "euclidean", "correlation"],
        help="faiss similarity: cosine / euclidean (L2) / correlation (Pearson)",
    )

    # cross_raf fusion
    p.add_argument(
        "--augment-mode", default="moe", help="cross_raf fusion mode (moe = cross+self)"
    )
    p.add_argument(
        "--mix-lambda", default=0.7, type=float, help="cross_raf cross/self mix weight"
    )

    # io / runtime
    p.add_argument(
        "--output-dir", default="./checkpoints", help="checkpoint / result directory"
    )
    p.add_argument("--gpu", default=0, type=int)
    p.add_argument("--batch-size", default=256, type=int)
    p.add_argument("--num-workers", default=2, type=int)
    p.add_argument("--seed", default=2021, type=int)
    return p


def add_eval_dataset_args(p: argparse.ArgumentParser):
    """Target-dataset args. Evaluation only — Cross-RAF is trained dataset-agnostic
    on the general corpus, so training never selects a dataset."""
    p.add_argument(
        "--root-path", default="./Datasets/ETT-small/", help="directory holding the CSV"
    )
    p.add_argument("--data-path", default="ETTh1.csv", help="CSV filename")
    p.add_argument(
        "--dataset",
        default="ETTh1",
        help="dataset name (controls train/val/test borders)",
    )
    p.add_argument(
        "--features",
        default="M",
        choices=["M", "S"],
        help="M: every column as an independent series / S: only --target",
    )
    p.add_argument("--target", default="OT", help="target column when --features S")
    p.add_argument(
        "--no-scale", action="store_true", help="disable per-channel standardization"
    )
    p.add_argument(
        "--retrieval-split",
        default="train",
        help="split used to build the (target-own) retrieval database for zero-shot eval",
    )
    p.add_argument(
        "--retrieval-stride",
        default=1,
        type=int,
        help="sliding-window stride for the retrieval database "
        "(1 = paper default / full KB; larger = fewer windows, less memory)",
    )
    return p


def train_args():
    """Training = Cross-RAF fusion pretraining on the general corpus (no dataset)."""
    p = argparse.ArgumentParser(description="Cross-RAF fusion pretraining (general corpus)")
    add_common_args(p)
    # general training corpus (Hugging Face nkh/TS-RAG-Data)
    p.add_argument(
        "--corpus-dir",
        default="./corpus/pretrain_pairs_ctx512",
        help="dir of precomputed pair parquets (target=[x|y], precomputed indices/distances)",
    )
    p.add_argument(
        "--retrieval-db-path",
        default="./corpus/retrieval_database_512.parquet",
        help="retrieval knowledge base parquet (x/y columns gathered by precomputed indices)",
    )
    p.add_argument(
        "--shuffle-buffer", default=10000, type=int, help="stream-shuffle reservoir size"
    )
    p.add_argument(
        "--drop-prob", default=0.0, type=float, help="target NaN-masking probability (paper: 0.0)"
    )
    p.add_argument(
        "--train-steps",
        default=10000,
        type=int,
        help="number of optimizer steps (Cross-RAG paper Table A.2: 10,000).",
    )
    p.add_argument(
        "--lr",
        default=3e-4,
        type=float,
        help="constant learning rate (Cross-RAG paper Table A.2: 3e-4, no scheduler).",
    )
    p.add_argument("--weight-decay", default=0.01, type=float)
    p.add_argument("--grad-clip", default=1.0, type=float)
    p.add_argument(
        "--save-freq", default=5000, type=int, help="checkpoint every N steps"
    )
    p.add_argument("--resume", default="", type=str)
    return p.parse_args()


def eval_args():
    p = argparse.ArgumentParser(description="RAF / Cross-RAF / Chronos — zero-shot evaluation")
    add_common_args(p)
    add_eval_dataset_args(p)
    p.add_argument(
        "--checkpoint", default="", help="trained weights (cross_raf / advanced raf)"
    )
    p.add_argument(
        "--eval-stride",
        default=1,
        type=int,
        help="test-window stride (1 = paper protocol / every timestep; "
        "pred_len = non-overlapping)",
    )
    p.add_argument("--result-file", default="result.txt")
    return p.parse_args()
