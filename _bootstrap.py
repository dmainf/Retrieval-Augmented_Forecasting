"""Process bootstrap — MUST be imported before torch/faiss anywhere.

faiss-cpu and torch each ship their own OpenMP (libomp) on macOS. When both are
loaded and run parallel regions, the duplicate runtime segfaults. We make this
safe by allowing the duplicate runtime and pinning both libraries to a single
OpenMP thread. Override with ``OMP_NUM_THREADS`` (e.g. on Linux where the
runtimes are shared and multi-threading is fine).
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

_N = max(1, int(os.environ.get("OMP_NUM_THREADS", "1")))

try:  # import faiss first so its OpenMP loads before torch's
    import faiss

    faiss.omp_set_num_threads(_N)
except Exception:  # pragma: no cover
    pass

try:
    import torch

    torch.set_num_threads(_N)
except Exception:  # pragma: no cover
    pass
