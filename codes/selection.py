"""How the reference examples are chosen.

Every function returns examples of shape (n_queries or 1, K, seq_len + pred_len)
plus the indices they came from. A leading 1 means one set shared by every query.

task-level (one set per channel, drawn from the train windows)
  random    K windows at random
  kmeans    the window nearest each of K centroids of z-scored window shapes

instance-level (chosen per query from every window that ends before its context)
  l2        nearest contexts by raw L2 -- the RAF baseline
  oracle    nearest futures to the TRUE future; a diagnostic bound, not a method
  recent    the K windows immediately preceding the query
  placebo   l2 windows whose futures are replaced by noise with the query's mean and std

ablations
  shuffle-future   permute the futures among the K examples, keeping the set of futures
  truth-future     replace every future with the query's true future (diagnostic)
"""
import numpy as np

TASK_LEVEL = ("random", "kmeans")
INSTANCE_LEVEL = ("l2", "oracle", "recent", "placebo")


def nearest(keys_db: np.ndarray, keys_q: np.ndarray, k: int, limit: np.ndarray) -> np.ndarray:
    """(n, k) indices of the k nearest rows of keys_db by L2, using only rows <= limit[i]."""
    d = ((keys_q ** 2).sum(1)[:, None] + (keys_db ** 2).sum(1)[None, :]
         - 2.0 * keys_q @ keys_db.T)
    d[np.arange(len(keys_db))[None, :] > limit[:, None]] = np.inf
    if (limit + 1 < k).any():
        raise ValueError("a query has fewer than k windows before it")
    idx = np.argpartition(d, k - 1, axis=1)[:, :k]
    order = np.take_along_axis(d, idx, 1).argsort(1)
    return np.take_along_axis(idx, order, 1)


def instance_level(method: str, data, split: str, channel: int, starts: np.ndarray,
                   query: np.ndarray, k: int, db_stride: int, rng):
    s = data.seq_len
    if method == "recent":
        ex = data.preceding_windows(split, channel, starts, k)
        return ex, np.full((len(query), k), -1)
    db = data.all_windows(channel, db_stride)
    limit = data.past_limit(split, starts, db_stride)
    if method in ("l2", "placebo"):
        idx = nearest(db[:, :s].astype(np.float64), query[:, :s].astype(np.float64), k, limit)
    elif method == "oracle":
        idx = nearest(db[:, s:].astype(np.float64), query[:, s:].astype(np.float64), k, limit)
    else:
        raise ValueError(method)
    ex = db[idx].copy()
    if method == "placebo":
        mu = query[:, :s].mean(1)[:, None, None]
        sd = query[:, :s].std(1)[:, None, None]
        ex[:, :, s:] = mu + sd * rng.standard_normal(ex[:, :, s:].shape)
    return ex, idx


def kmeans_set(pool: np.ndarray, k: int, seed: int, seq_len: int) -> np.ndarray:
    from sklearn.cluster import KMeans
    ctx = pool[:, :seq_len]
    x = (pool - ctx.mean(1, keepdims=True)) / (ctx.std(1, keepdims=True) + 1e-8)
    km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(x)
    d = ((x[:, None, :] - km.cluster_centers_[None]) ** 2).sum(-1)
    picked = []
    for c in range(k):
        picked.append(next(int(i) for i in np.argsort(d[:, c]) if i not in picked))
    return np.array(picked)


def task_level(method: str, pool: np.ndarray, k: int, seed: int, seq_len: int, rng):
    if k > len(pool):
        raise ValueError(f"k={k} exceeds the {len(pool)} train windows")
    if method == "random":
        idx = rng.choice(len(pool), size=k, replace=False)
    elif method == "kmeans":
        idx = kmeans_set(pool, k, seed, seq_len)
    else:
        raise ValueError(method)
    return pool[idx][None].copy(), idx


def ablate(ex: np.ndarray, mode: str, seq_len: int, futures: np.ndarray, rng) -> np.ndarray:
    if mode == "none":
        return ex
    out = ex.copy()
    if mode == "shuffle-future":
        for i in range(len(out)):
            out[i, :, seq_len:] = out[i, rng.permutation(out.shape[1]), seq_len:]
        return out
    if mode == "truth-future":
        out = np.broadcast_to(out, (len(futures),) + out.shape[1:]).copy()
        out[:, :, seq_len:] = futures[:, None, :]
        return out
    raise ValueError(mode)
