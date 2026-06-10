"""faiss-based top-k retriever over time-series windows (X-space).

The retrieval database is the set of ``[context | future]`` windows from the
train split. A query is a context of length ``seq_len``. All metrics operate on
min-max normalized contexts (cross-rag X-space matching), then build a faiss
flat index:

  * cosine       : L2-normalize, IndexFlatIP   (distance = 1 - cosine)
  * euclidean    : raw,           IndexFlatL2   (distance = squared L2)
  * correlation  : mean-center + L2-normalize, IndexFlatIP  (Pearson; distance = 1 - corr)

faiss runs on CPU here (GPU index is used automatically when available). The full
retrieved windows (context + future) and their distances are returned as torch
tensors so the rest of the pipeline is unchanged.
"""
from typing import Tuple

import _bootstrap  # noqa: F401  (sets OpenMP safety before faiss/torch load)
import faiss
import numpy as np
import torch

METRICS = ("cosine", "euclidean", "correlation")


def _minmax(x: np.ndarray) -> np.ndarray:
    x_min = x.min(axis=1, keepdims=True)
    x_max = x.max(axis=1, keepdims=True)
    rng = np.where((x_max - x_min) == 0, 1.0, x_max - x_min)
    return (x - x_min) / rng


def _make_keys(x: np.ndarray, metric: str) -> np.ndarray:
    """Map min-max normalized windows into the space matching the chosen metric.

    The per-window min-max gives shape-based matching; the metric-specific
    normalization (L2 / mean-center) below is always applied because faiss only
    computes a raw inner product / L2. Only the search keys are normalized — the
    stored windows returned by ``gather`` are left untouched.
    """
    x = _minmax(x.astype(np.float32))
    if metric == "euclidean":
        keys = x
    elif metric == "cosine":
        keys = x.copy()
        faiss.normalize_L2(keys)
    elif metric == "correlation":
        keys = x - x.mean(axis=1, keepdims=True)
        faiss.normalize_L2(keys)
    else:
        raise ValueError(f"Unknown metric: {metric}. Choices: {METRICS}")
    return np.ascontiguousarray(keys, dtype=np.float32)


class Retriever:
    def __init__(
        self,
        db_windows,
        seq_len: int,
        pred_len: int,
        metric: str = "cosine",
        device: str = "cpu",
    ):
        assert metric in METRICS, f"metric must be one of {METRICS}"
        self.metric = metric
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.device = device

        db = np.asarray(db_windows, dtype=np.float32)
        self.db_windows = torch.as_tensor(db, device=device)  # (N, win)
        self.n_db, dim = db.shape[0], seq_len

        keys = _make_keys(db[:, :seq_len], metric)
        if metric == "euclidean":
            index = faiss.IndexFlatL2(dim)
        else:  # cosine / correlation use inner product on normalized vectors
            index = faiss.IndexFlatIP(dim)

        # use a GPU index when a faiss-gpu build is available
        if device.startswith("cuda") and hasattr(faiss, "StandardGpuResources"):
            res = faiss.StandardGpuResources()
            gpu_id = int(device.split(":")[1]) if ":" in device else 0
            index = faiss.index_cpu_to_gpu(res, gpu_id, index)

        index.add(keys)
        self.index = index
        self._larger_is_closer = metric in ("cosine", "correlation")

    @torch.no_grad()
    def search(self, query_ctx, top_k: int, exclude_self: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """query_ctx: (B, seq_len). Returns (indices (B,k), distances (B,k))."""
        if torch.is_tensor(query_ctx):
            query_ctx = query_ctx.detach().cpu().numpy()
        q = _make_keys(np.asarray(query_ctx, dtype=np.float32), self.metric)

        k_fetch = min(top_k + (1 if exclude_self else 0), self.n_db)
        scores, idx = self.index.search(q, k_fetch)  # IP: larger=closer / L2: smaller=closer

        # convert to a uniform "smaller is closer" distance
        dist = (1.0 - scores) if self._larger_is_closer else scores

        if exclude_self:
            is_self = dist[:, 0] < 1e-6
            dist = np.where(is_self[:, None], dist[:, 1:], dist[:, :-1])
            idx = np.where(is_self[:, None], idx[:, 1:], idx[:, :-1])
        idx = idx[:, :top_k]
        dist = dist[:, :top_k]

        return (
            torch.as_tensor(idx, dtype=torch.long),
            torch.as_tensor(dist, dtype=torch.float32),
        )

    def gather(self, indices) -> torch.Tensor:
        """indices: (B, k) -> retrieved full windows (B, k, win_len)."""
        idx = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        return self.db_windows[idx]
