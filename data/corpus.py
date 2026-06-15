"""General training corpus for Cross-RAF (paper-faithful, dataset-agnostic).

Cross-RAF trains its fusion modules **once** on a large, dataset-agnostic corpus
and is then applied zero-shot to unseen targets (ETTh1, weather, ...). It never
trains on the target dataset. The two artifacts (hosted on Hugging Face
``nkh/TS-RAG-Data``) are:

  * ``pretrain_pairs_ctx512/*.parquet`` -- one row per training pair, columns
    ``target = [x(512) | y(64)]`` (List<double>), plus the **precomputed** top-k
    retrieval ``indices`` (List<int64>) / ``distances`` (List<double>) into the
    retrieval database below. 30 chunks x 1M rows ~= the paper's 26M pairs.
  * ``retrieval_database_512.parquet``  -- the retrieval knowledge base, columns
    ``x(512)`` / ``y(64)`` (List<float>) (plus a 768-d ``embedding`` used only to
    build the offline index, which we skip here since indices are precomputed).

Because the indices are precomputed, training does **no** faiss search: each step
just gathers ``whole_seq[indices]`` and feeds it to the fusion model. We read the
parquets directly with pyarrow (the reference uses GluonTS FileDataset, but its
pydantic models are incompatible with this environment's Python).
"""
import gc
from pathlib import Path

import numpy as np
import torch
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset


class _PseudoShuffled(IterableDataset):
    """Stream-shuffle an iterable through a fixed-size reservoir buffer."""

    def __init__(self, base, buffer_len: int = 10000):
        super().__init__()
        self.base = base
        self.buffer_len = buffer_len
        self.gen = torch.Generator()

    def __iter__(self):
        buf = []
        for el in self.base:
            buf.append(el)
            if len(buf) >= self.buffer_len:
                i = torch.randint(len(buf), size=(), generator=self.gen)
                yield buf.pop(i)
        while buf:
            i = torch.randint(len(buf), size=(), generator=self.gen)
            yield buf.pop(i)


class CorpusPairs(IterableDataset):
    """Cyclically stream ``(x, y, distances, indices)`` from the pair parquets.

    ``target = [x | y]`` is split into ``x`` (tail of length ``context_length``)
    and ``y`` (head of length ``prediction_length``); the retrieval fields are
    truncated to ``top_k``. ``drop_prob`` optionally NaN-masks the target (paper
    default 0.0). The stream is infinite so the loader never runs dry before
    ``train_steps``.
    """

    def __init__(self, data_dir, context_length=512, prediction_length=64,
                 retrieve_lookback_length=512, top_k=15, drop_prob=0.0,
                 read_batch=8192):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.retrieve_lookback_length = retrieve_lookback_length
        self.top_k = top_k
        self.drop_prob = drop_prob
        self.read_batch = read_batch

        if not self.data_dir.is_dir():
            raise ValueError(f"corpus dir not found: {self.data_dir}")
        self.files = sorted(str(f) for f in self.data_dir.iterdir()
                            if f.is_file() and f.suffix == ".parquet")
        if not self.files:
            raise ValueError(f"no parquet files in {self.data_dir}")

    def shuffle(self, buffer_len: int = 10000):
        return _PseudoShuffled(self, buffer_len)

    @staticmethod
    def _mat(col):
        """ListArray (uniform length) -> 2D numpy via zero-copy child reshape."""
        n = len(col)
        flat = col.values.to_numpy(zero_copy_only=False)
        return flat.reshape(n, -1)

    def _file_batches(self, path):
        """Infinite-cyclic stream of pyarrow batches from one parquet file."""
        while True:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(
                batch_size=self.read_batch,
                columns=["target", "indices", "distances"],
            ):
                yield batch

    def _rows(self, batch, x0, lb, pl, k):
        tgt = self._mat(batch.column("target"))           # (n, lb+pl)
        idx = self._mat(batch.column("indices"))[:, :k]    # (n, k)
        dist = self._mat(batch.column("distances"))[:, :k]
        X = tgt[:, x0:lb].astype(np.float32, copy=False)   # (n, ctx)
        Y = tgt[:, lb:lb + pl].astype(np.float32, copy=False)
        idx = np.ascontiguousarray(idx, dtype=np.int64)
        dist = np.ascontiguousarray(dist, dtype=np.float32)
        for i in range(len(tgt)):
            y = Y[i]
            if self.drop_prob > 0:
                p = np.random.uniform(0.0, self.drop_prob)
                mask = np.random.choice([True, False], size=pl, p=[p, 1 - p])
                y = y.copy()
                y[mask] = np.nan
            yield {"x": X[i], "y": y, "distances": dist[i], "indices": idx[i]}

    def __iter__(self):
        lb, ctx, pl, k = (self.retrieve_lookback_length, self.context_length,
                          self.prediction_length, self.top_k)
        x0 = max(0, lb - ctx) if ctx > 0 else 0
        # Round-robin one batch from each file so the stream interleaves all
        # chunks: with sequential reading a short run (< 1 epoch) would only
        # touch the first few of ~30 heterogeneous chunks. Each file iterator is
        # infinite, so the stream is cyclic. The downstream shuffle buffer then
        # de-correlates the interleaved batches.
        batch_iters = [self._file_batches(p) for p in self.files]
        while True:
            for bi in batch_iters:
                yield from self._rows(next(bi), x0, lb, pl, k)


class RetrievalDB:
    """Holds ``whole_seq = [x | y]`` of the retrieval KB for index-based gather.

    Only the ``x`` / ``y`` columns are read (the 768-d ``embedding`` is skipped,
    saving ~8 GB of RAM) because the corpus pairs already carry precomputed
    neighbour indices, so no faiss search happens at train time.
    """

    def __init__(self, parquet_path, seq_len=512, pred_len=64):
        path = str(parquet_path)
        n = pq.ParquetFile(path).metadata.num_rows
        win = seq_len + pred_len
        self.whole_seq = np.empty((n, win), dtype=np.float32)

        self._fill_column(path, "x", 0, seq_len)
        self._fill_column(path, "y", seq_len, win)

    def _fill_column(self, path, col, start, stop):
        pf = pq.ParquetFile(path)
        row = 0
        for batch in pf.iter_batches(batch_size=200000, columns=[col]):
            arr = batch.column(col)
            block = arr.values.to_numpy(zero_copy_only=False).reshape(len(arr), -1)
            self.whole_seq[row:row + len(block), start:stop] = block
            row += len(block)
            del arr, block
        gc.collect()

    def gather(self, indices) -> torch.Tensor:
        idx = np.asarray(indices, dtype=np.int64)
        return torch.from_numpy(self.whole_seq[idx])  # (B, k, win)
