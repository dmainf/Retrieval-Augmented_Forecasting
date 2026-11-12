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


def candidates(limit: np.ndarray, n_pool: int, m: int, seed: int, channel: str, ids=None):
    """(n, M) pool indices drawn per query at random among the windows <= limit[i]
    (all of them when m is 0) and their validity (shorter draws are padded). The draw of
    query i is seeded by ids[i] (default i), its index among all the split's origins."""
    limit = np.minimum(limit, n_pool - 1)
    crc = zlib.crc32(channel.encode())
    draws = []
    ids = np.arange(len(limit)) if ids is None else ids
    for i, lim in zip(ids, limit):
        perm = np.random.default_rng([seed, crc, int(i)]).permutation(int(lim) + 1)
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


def head_score(att: np.ndarray, valid: np.ndarray, mix: np.ndarray = None) -> np.ndarray:
    """(M,) score of the candidates: head_weights averaged over the heads (weighted by mix)."""
    w = head_weights(att, valid)
    return w.mean(0) if mix is None else (mix[:, None] * w).sum(0) / mix.sum()


def past_head_mix(att_past: np.ndarray, valid_past: np.ndarray, dist: np.ndarray):
    """Per-query head weights from the query's own past. att_past (heads, cols, 1 + M): the query
    cut one patch back attending to the candidates shifted one patch back; dist (M,): distance of
    the query's last patch to each shifted candidate's first future patch. rho per head is the
    Spearman correlation of its candidate weights with -dist, pi is max(rho, 0) normed (uniform
    when no head is positive) -> pi, rho (heads,)."""
    w = head_weights(att_past, valid_past)[:, valid_past]
    rank = lambda x: np.argsort(np.argsort(x, -1, kind="stable"), -1, kind="stable").astype(np.float64)
    rw, rd = rank(w), rank(-dist[valid_past])
    rw -= rw.mean(-1, keepdims=True)
    rd -= rd.mean()
    rho = rw @ rd / np.maximum(np.linalg.norm(rw, axis=-1) * np.linalg.norm(rd), 1e-30)
    pi = np.maximum(rho, 0)
    return (pi / pi.sum() if pi.sum() > 0 else np.full_like(pi, 1 / len(pi))), rho


VALUE_SCORES = ("none", "av", "move", "align", "still")


def value_score(att: np.ndarray, v_self: np.ndarray, v_cand: np.ndarray, gram: np.ndarray,
                kind: str, cols, valid: np.ndarray, mix: np.ndarray = None) -> np.ndarray:
    """(M,) score of the candidates from the attention weights and the values, on the forecast
    columns cols. att (heads, cols, 1 + M) over [self, candidates], v_self (heads, cols, d),
    v_cand (M, heads, cols, d), gram (heads, d, d) of the output projection. Reading
    candidate j moves the head's output by alpha_j (v_j - v_0):
      none   alpha_j                       (as head_score, on cols only)
      av     alpha_j |v_j W_O|
      move   alpha_j |(v_j - v_0) W_O|
      align  alpha_j cos((v_j - v_0) W_O, m W_O), m = sum_l alpha_l (v_l - v_0)
      still  alpha_j / |(v_j - v_0) W_O|
    summed over the columns, normed per head and averaged over the heads (weighted by mix)."""
    a = np.moveaxis(att[:, cols, 1:], -1, 0) * valid[:, None, None]
    if kind == "none":
        s = a.sum(2)
    else:
        g = gram
        vs, vc = v_self[:, cols].astype(np.float64), v_cand[:, :, cols].astype(np.float64)
        norm = lambda x: np.sqrt(np.maximum(np.einsum("...hcd,hde,...hce->...hc", x, g, x), 1e-30))
        if kind == "av":
            s = (a * norm(vc)).sum(2)
        else:
            delta = vc - vs[None]
            nd = norm(delta)
            if kind == "move":
                s = (a * nd).sum(2)
            elif kind == "still":
                s = (a / nd).sum(2)
            elif kind == "align":
                m = (a[..., None] * delta).sum(0)
                dot = np.einsum("mhcd,hde,hce->mhc", delta, g, m)
                s = (a * dot / (nd * norm(m)[None])).sum(2)
            else:
                raise ValueError(kind)
    s = s / np.maximum(np.abs(s).sum(0, keepdims=True), 1e-30)
    return s.mean(1) if mix is None else (s * mix[None]).sum(1) / mix.sum()


def diverse_top_k(score: np.ndarray, keys: np.ndarray, valid: np.ndarray, k: int, lam: float):
    """Positions of k candidates of one query picked one by one (maximal marginal relevance):
    the next is the argmax of log score - lam * (its largest cosine to the picked ones' keys).
    score (M,) > 0, keys (M, D) unit-norm."""
    if valid.sum() < k:
        raise ValueError(f"a query has fewer than k={k} candidates")
    gain = np.where(valid, np.log(np.maximum(score, 1e-300)), -np.inf)
    near = np.zeros(len(score))
    picked = np.empty(k, dtype=np.int64)
    for r in range(k):
        j = int(np.argmax(gain - lam * near))
        picked[r] = j
        gain[j] = -np.inf
        c = keys @ keys[j]
        near = c if r == 0 else np.maximum(near, c)
    return picked


def spread_match(score: np.ndarray, valid: np.ndarray, fut: np.ndarray, sigma: np.ndarray, k: int, m: int):
    """Positions of k candidates of one query among its m best-scoring ones whose futures spread
    like the query's no-reference forecast: greedy swaps from the top k lower
    mean_h |sd_j(fut[j, h]) - sigma[h]|, each taking out the picked candidate farthest from (or
    nearest to) the picked mean and putting in the unpicked one nearest to (or farthest from) it
    while the spread is too wide (too narrow). fut (M, H) standardized on each window's context,
    sigma (H,) on the query's history."""
    order = np.argsort(-np.where(valid, score, -np.inf), kind="stable")[:m]
    if not valid[order].all() or len(order) < k:
        raise ValueError(f"a query has fewer than m={m} candidates")
    y = fut[order].astype(np.float64)
    picked = np.zeros(len(order), dtype=bool)
    picked[:k] = True
    s1, s2 = y[:k].sum(0), (y[:k] ** 2).sum(0)
    gap = lambda a, b: np.abs(np.sqrt(np.maximum(b / k - (a / k) ** 2, 0)) - sigma).mean()
    cur = gap(s1, s2)
    for _ in range(len(order) - k):
        dev = ((y - s1 / k) ** 2).mean(1)
        wide = np.sqrt(np.maximum(s2 / k - (s1 / k) ** 2, 0)).mean() > sigma.mean()
        o = np.flatnonzero(picked)[np.argmax(dev[picked]) if wide else np.argmin(dev[picked])]
        n = np.flatnonzero(~picked)[np.argmin(dev[~picked]) if wide else np.argmax(dev[~picked])]
        t1, t2 = s1 - y[o] + y[n], s2 - y[o] ** 2 + y[n] ** 2
        new = gap(t1, t2)
        if new >= cur:
            break
        picked[o], picked[n], s1, s2, cur = False, True, t1, t2, new
    return order[picked]


def precomputed_attention(q: np.ndarray, k_self: np.ndarray, k_pool: np.ndarray,
                          cand: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """q, k_self (heads, cols, d) of one query, k_pool (N, heads, cols, d) ->
    (heads, cols, 1 + M) softmax over its own key and its candidates' keys."""
    logit = np.einsum("hcd,mhcd->hcm", q, k_pool[cand])
    logit = np.where(valid[None, None], logit, -np.inf)
    full = np.concatenate([(q * k_self).sum(-1)[..., None], logit], -1)
    full = np.exp(full - full.max(-1, keepdims=True))
    return full / full.sum(-1, keepdims=True)
