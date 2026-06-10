"""Glue between TSData, the retriever, and torch DataLoaders."""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data.dataset import TSData
from retrieval.retriever import Retriever


def build_tsdata(args) -> TSData:
    return TSData(
        root_path=args.root_path,
        data_path=args.data_path,
        dataset_name=args.dataset,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        features=args.features,
        target=args.target,
        scale=not args.no_scale,
    )


def window_loader(tsd: TSData, args, split: str, stride: int, shuffle: bool) -> DataLoader:
    windows, _ = tsd.make_windows(split, stride=stride)
    ctx = torch.from_numpy(windows[:, : args.seq_len])
    tgt = torch.from_numpy(windows[:, args.seq_len : args.seq_len + args.pred_len])
    ds = TensorDataset(ctx, tgt)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.num_workers, drop_last=False)


def build_retriever(tsd: TSData, args, device) -> Retriever:
    db_windows, _ = tsd.make_windows(args.retrieval_split, stride=args.retrieval_stride)
    return Retriever(
        db_windows,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        metric=args.retrieval_metric,
        device=str(device),
    )
