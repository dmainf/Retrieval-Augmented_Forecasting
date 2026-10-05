"""How the reference examples are chosen.

Every method draws from the same pool, the train windows of one channel, and
returns examples of shape (n_queries or 1, K, seq_len + pred_len) plus their
indices in the pool. A leading 1 means one set shared by every query.

task-level (one set per channel)
  random    K windows at random
  clg       K windows whose mean curriculum latent gradient matches the mean over all
            pool windows, with the latent windows spliced in front of the history
  pclg      the same with the latent windows as parallel covariate rows (Parallel-CLG);
            gradients come from clg.py

instance-level (chosen per query among the pool windows that end before its context)
  l2        nearest contexts by raw L2 -- the RAF baseline
  oracle    nearest futures to the TRUE future; a diagnostic bound, not a method
  truth     K copies of the query itself, its context and its TRUE future; not from the
            pool. A diagnostic of how far a given future is read, not a method

ablation
  --reverse-match  for clg / pclg, the set farthest from the mean gradient (diagnostic)
  --match cosine   for clg / pclg, rank windows by gradient alignment instead of matching the mean
                   (the default for pclg; clg keeps the paper's mean matching -- see DEFAULT_MATCH)

recent target (--recent H)
  for clg / pclg, match (or align with) the mean gradient of the pool windows that end in
  the last H points before the evaluated split, instead of the mean over all of them

hybrid (--inst-k)
  a task-level set plus, per query, the nearest-context (l2) windows not already in the set,
  all as parallel rows -- the task-level many-shot + instance-level few-shot of Zhang et al.
  shuffle-future   permute the futures among the K examples, keeping the set of futures,
                   whatever the selection (a no-op for truth, whose K examples are equal)
"""
import numpy as np

TASK_LEVEL = ("random", "clg", "pclg")
INSTANCE_LEVEL = ("l2", "oracle", "truth")


def nearest(keys_db: np.ndarray, keys_q: np.ndarray, k: int, limit: np.ndarray,
            exclude=None) -> np.ndarray:
    """(n, k) indices of the k nearest rows of keys_db by L2, using only rows <= limit[i]
    and not in exclude."""
    d = ((keys_q ** 2).sum(1)[:, None] + (keys_db ** 2).sum(1)[None, :]
         - 2.0 * keys_q @ keys_db.T)
    d[np.arange(len(keys_db))[None, :] > limit[:, None]] = np.inf
    n_excl = 0
    if exclude is not None and len(exclude):
        d[:, exclude] = np.inf
        n_excl = int((np.asarray(exclude)[None, :] <= limit[:, None]).sum(1).max())
    if (limit + 1 - n_excl < k).any():
        raise ValueError("a query has fewer than k windows before it")
    idx = np.argpartition(d, k - 1, axis=1)[:, :k]
    order = np.take_along_axis(d, idx, 1).argsort(1)
    return np.take_along_axis(idx, order, 1)


def instance_level(method: str, pool: np.ndarray, query: np.ndarray, limit: np.ndarray,
                   k: int, s: int, exclude=None):
    """limit[i] is the highest pool index query i may use (see TimeSeriesData.past_limit);
    exclude: pool indices never to pick (a shared set the examples are added to).
    Returns indices None for truth, whose examples do not come from the pool."""
    if method == "truth":
        return np.repeat(query[:, None, :], k, axis=1), None
    if method == "l2":
        idx = nearest(pool[:, :s].astype(np.float64), query[:, :s].astype(np.float64), k, limit,
                      exclude)
    elif method == "oracle":
        idx = nearest(pool[:, s:].astype(np.float64), query[:, s:].astype(np.float64), k, limit)
    else:
        raise ValueError(method)
    return pool[idx].copy(), idx


def match_mean(grads: np.ndarray, k: int, swaps: int, reverse: bool = False,
               target: np.ndarray = None) -> np.ndarray:
    """Greedy search, then up to `swaps` single swaps, for the k rows of grads whose
    mean is closest in L2 to the mean of all rows (Algorithm 1 of Zhang et al.).
    reverse: greedily take the farthest instead and skip the swaps -- the paper's
    mismatch ablation (Sec. 4.7), as in the official code."""
    g = grads.astype(np.float64)
    target = g.mean(0) if target is None else target.astype(np.float64)
    n = len(g)
    chosen, total, free = [], np.zeros_like(target), np.ones(n, dtype=bool)
    for i in range(1, k + 1):
        d = (((total[None] + g) / i - target[None]) ** 2).sum(1)
        d = -d if reverse else d
        j = int(np.argmin(np.where(free, d, np.inf)))
        chosen.append(j)
        free[j] = False
        total += g[j]
    sq = (g ** 2).sum(1)
    for _ in range(0 if reverse else swaps):
        r = k * target - total
        gs = g[chosen]
        # squared residual after swapping chosen a out for candidate c: |r + g_a - g_c|^2
        val = (r @ r + sq[chosen][:, None] + sq[None, :] - 2 * gs @ g.T
               + 2 * (gs @ r)[:, None] - 2 * (g @ r)[None, :])
        val = np.where(free[None, :], val, np.inf)
        a, c = np.unravel_index(np.argmin(val), val.shape)
        if val[a, c] >= r @ r:
            break
        total += g[c] - g[chosen[a]]
        free[chosen[a]], free[c] = True, False
        chosen[a] = int(c)
    return np.array(chosen)


def align(grads: np.ndarray, k: int, max_gap: int, target: np.ndarray = None) -> np.ndarray:
    """The k rows of grads most aligned (cosine) with their mean -- the windows whose
    latent update most lowers the loss of the average query, summed over checkpoints
    as in TracIn -- with no two picks closer than a gap in pool index. The gap starts as
    wide as the pool allows (n // k - 1, at most max_gap = no time overlap at all) and
    narrows until k windows fit, since greedy packing can fall short of n // k."""
    g = grads.astype(np.float64)
    target = g.mean(0) if target is None else target.astype(np.float64)
    cos = (g @ target) / (np.linalg.norm(g, axis=1) * np.linalg.norm(target) + 1e-30)
    order = np.argsort(-cos)
    for gap in range(max(1, min(max_gap, len(g) // k - 1)), 0, -1):
        chosen, blocked = [], np.zeros(len(g), dtype=bool)
        for j in order:
            if blocked[j]:
                continue
            chosen.append(int(j))
            blocked[max(0, j - gap + 1):j + gap] = True
            if len(chosen) == k:
                return np.array(chosen)
    raise ValueError(f"k={k} exceeds the {len(g)} windows")


def task_level(method: str, pool: np.ndarray, k: int, rng, grads: np.ndarray = None,
               swaps: int = 0, reverse: bool = False, match: str = "mean", max_gap: int = 1,
               target_idx: np.ndarray = None):
    """grads (n_pool, d): curriculum latent gradients of the pool windows, for clg / pclg.
    match "mean" matches the mean gradient (CLG), "cosine" ranks by alignment (align);
    reverse picks the mismatching set (diagnostic). target_idx: pool windows whose mean
    gradient is the target instead of the mean over all (e.g. the most recent ones)."""
    if k > len(pool):
        raise ValueError(f"k={k} exceeds the {len(pool)} train windows")
    if method == "random":
        idx = rng.choice(len(pool), size=k, replace=False)
    elif method in ("clg", "pclg"):
        if grads is None or len(grads) != len(pool):
            raise ValueError(f"{method} needs one gradient per pool window")
        target = None if target_idx is None else grads[target_idx].mean(0)
        idx = (align(grads, k, max_gap, target) if match == "cosine"
               else match_mean(grads, k, swaps, reverse, target))
    else:
        raise ValueError(method)
    return pool[idx][None].copy(), idx


def ablate(ex: np.ndarray, mode: str, seq_len: int, rng) -> np.ndarray:
    if mode == "none":
        return ex
    if mode == "shuffle-future":
        out = ex.copy()
        for i in range(len(out)):
            out[i, :, seq_len:] = out[i, rng.permutation(out.shape[1]), seq_len:]
        return out
    raise ValueError(mode)
