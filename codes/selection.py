"""How the reference examples are chosen.

Every method picks K examples per query among the same candidates: train windows
of the query's channel that end at least GAP points before the forecast origin,
--candidates of them drawn at random per query (all of them when 0).

random    the first K candidates (the draw is already random)
match     <query side>=<window side>, nearest K by --metric:
            ctx=ctx   the query's last seq_len history points vs the window's context
            fc=fc     the query's no-reference forecast vs the window's no-reference
                      forecast from its own context only
            fc=fut    the query's forecast vs the window's future
            fut=fc    the query's TRUE future vs the window's forecast (diagnostic)
            fut=fut   the query's TRUE future vs the window's future (diagnostic bound)
          with --match-norm context every series is standardized on its own context
          (the query on its last seq_len points, the window on its context) first.
attn      group attention of chosen heads: per head and forecast column, a softmax
          of the query's query vector over its own key and the candidates' keys; the
          weights on the candidates are averaged over the forecast columns, renormed
          to sum to 1 and averaged over the heads, and the K largest are kept.
            precompute  every row runs on its own, so a candidate's key is fixed and
                        computed once per window (rows do not read each other)
            exact       the query and all its candidates in one forward
"""
import zlib

import numpy as np

MATCHES = ("ctx=ctx", "fc=fc", "fc=fut", "fut=fc", "fut=fut")
METRICS = ("l2", "cosine", "corr")


def candidates(limit: np.ndarray, n_pool: int, m: int, seed: int, channel: str):
    """(n, M) pool indices drawn per query at random among the windows <= limit[i]
    (all of them when m is 0) and their validity (shorter draws are padded)."""
    limit = np.minimum(limit, n_pool - 1)
    crc = zlib.crc32(channel.encode())
    draws = []
    for i, lim in enumerate(limit):
        perm = np.random.default_rng([seed, crc, i]).permutation(int(lim) + 1)
        draws.append(perm if m == 0 else perm[:m])
    width = max(len(d) for d in draws)
    cand = np.zeros((len(limit), width), dtype=np.int64)
    valid = np.zeros((len(limit), width), dtype=bool)
    for i, d in enumerate(draws):
        cand[i, :len(d)], valid[i, :len(d)] = d, True
    return cand, valid


def top_k(score: np.ndarray, cand: np.ndarray, valid: np.ndarray, k: int):
    """Highest-scoring k valid candidates per query -> pool indices and scores (n, k)."""
    if (valid.sum(1) < k).any():
        raise ValueError(f"a query has fewer than k={k} candidates")
    score = np.where(valid, score, -np.inf)
    order = np.argsort(-score, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(cand, order, 1), np.take_along_axis(score, order, 1)


def standardize(x: np.ndarray, ctx: np.ndarray) -> np.ndarray:
    mu = ctx.mean(-1, keepdims=True)
    sd = ctx.std(-1, keepdims=True)
    return (x - mu) / np.maximum(sd, 1e-8)


def distance(q: np.ndarray, d: np.ndarray, metric: str) -> np.ndarray:
    """q (L,), d (M, L) -> (M,) distances, smaller is nearer."""
    if metric == "l2":
        return np.sqrt(((d - q) ** 2).sum(-1))
    if metric == "corr":
        q, d = q - q.mean(), d - d.mean(-1, keepdims=True)
    elif metric != "cosine":
        raise ValueError(metric)
    return 1.0 - d @ q / np.maximum(np.linalg.norm(d, axis=-1) * np.linalg.norm(q), 1e-12)


def match(pair: str, query: dict, window: dict, cand, valid, k: int, metric: str, norm: str):
    """query / window: {"ctx", "fut", "fc"} arrays of the queries (n, L) and the pool (N, L)."""
    qs, ws = pair.split("=")
    q = query[qs].astype(np.float64)
    w = window[ws].astype(np.float64)
    if norm == "context":
        q = standardize(q, query["ctx"].astype(np.float64))
        w = standardize(w, window["ctx"].astype(np.float64))
    score = np.empty(cand.shape)
    for i in range(len(q)):
        score[i] = -distance(q[i], w[cand[i]], metric)
    return top_k(score, cand, valid, k)


def head_weights(att: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """att (heads, cols, 1 + M) softmax weights over [self, candidates] -> (heads, M): the
    weights on the candidates averaged over the columns and renormed to sum to 1 per head."""
    w = att[..., 1:].mean(1) * valid
    return w / np.maximum(w.sum(-1, keepdims=True), 1e-30)


def head_score(att: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """(M,) score of the candidates: head_weights averaged over the heads."""
    return head_weights(att, valid).mean(0)


def precomputed_attention(q: np.ndarray, k_self: np.ndarray, k_pool: np.ndarray,
                          cand: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """q, k_self (heads, cols, d) of one query, k_pool (N, heads, cols, d) ->
    (heads, cols, 1 + M) softmax over its own key and its candidates' keys."""
    logit = np.einsum("hcd,mhcd->hcm", q, k_pool[cand])
    logit = np.where(valid[None, None], logit, -np.inf)
    full = np.concatenate([(q * k_self).sum(-1)[..., None], logit], -1)
    full = np.exp(full - full.max(-1, keepdims=True))
    return full / full.sum(-1, keepdims=True)
