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
          --value-score also runs every candidate with its future hidden (masked_score)
state     the candidates whose state with the future hidden is nearest to the query's
          (state_distance), every row run on its own as in precompute (the outputs of
          single heads are --select attn --value-score vpre):
            out    the output of a layer's group attention (all heads) at --layers
            in     the input of a layer's group attention (after its layer norm) at --layers
            final  the encoder's final state, which the forecast head reads
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


def precomputed_attention(q: np.ndarray, k_self: np.ndarray, k_pool: np.ndarray,
                          cand: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """q, k_self (heads, cols, d) of one query, k_pool (N, heads, cols, d) ->
    (heads, cols, 1 + M) softmax over its own key and its candidates' keys."""
    logit = np.einsum("hcd,mhcd->hcm", q, k_pool[cand])
    logit = np.where(valid[None, None], logit, -np.inf)
    full = np.concatenate([(q * k_self).sum(-1)[..., None], logit], -1)
    full = np.exp(full - full.max(-1, keepdims=True))
    return full / full.sum(-1, keepdims=True)


VALUE_SCORES = ("none", "q", "qpre", "qchange", "qchange_s", "k", "kpre", "kchange", "kchange_s", "v", "vpre",
                "vchange", "vchange_s", "qkpre", "qkchange", "qkv", "qkvpre", "qkvchange", "avpre", "avchpre",
                "qkvo", "qkbi", "jac", "jac_s", "vscale", "vstab", "vnorm", "cosvscale")
JOINT_SCORES = ("qkvo",)
RAW_SCORES = ("jac", "jac_s")
MASKED_ONLY = ("qkvo", "qkbi", "jac", "jac_s", "vscale", "vstab", "vnorm", "cosvscale")


def out_projection(gram: np.ndarray):
    """Map values (..., heads, cols, d) by the Cholesky factor of each head's W_O^T W_O (gram), so
    that plain dot products of the mapped values are those of the heads' outputs v W_O."""
    g = gram.astype(np.float64)
    ch = np.stack([np.linalg.cholesky(x + 1e-9 * np.trace(x) / len(x) * np.eye(len(x))) for x in g])
    return lambda v: np.einsum("...hcd,hde->...hce", v.astype(np.float64), ch).astype(np.float32)


def joint_projection(gram: np.ndarray, n_heads: int):
    """Map values (..., heads, cols, d) to (..., heads, cols, heads * d) by the rows of the Cholesky factor
    of the heads' joint W_O^T W_O (gram, (heads * d, heads * d)) that belong to each head, so that the norm
    of a sum over the heads of mapped values is that of the sum of the heads' outputs v W_O."""
    g = gram.astype(np.float64)
    ch = np.linalg.cholesky(g + 1e-9 * np.trace(g) / len(g) * np.eye(len(g)))
    ch = ch.reshape(n_heads, len(g) // n_heads, len(g))
    return lambda v: np.einsum("...hcd,hde->...hce", v.astype(np.float64), ch).astype(np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.exp(x - x.max(-1, keepdims=True))
    return x / x.sum(-1, keepdims=True)


def state_distance(x0: np.ndarray, x: np.ndarray) -> np.ndarray:
    """x0 (units, cols, d) of the query, x (M, units, cols, d) of the candidates -> (M,) sum over the
    units (heads or layers) and forecast columns of the Euclidean distance |x_j - x0|."""
    return np.sqrt(((x.astype(np.float64) - x0.astype(np.float64)[None]) ** 2).sum(-1)).sum((1, 2))


def candidate_terms(q0: np.ndarray, k0: np.ndarray, v0: np.ndarray, X: dict) -> dict:
    """The terms of one query's candidates, each (heads, cols, M). q0, k0, v0 (heads, cols, d) are the
    query's (its future hidden), X holds the candidates' vectors (M, heads, cols, d): q, k, v with their
    true future and qm, km, vm with it hidden (values mapped by out_projection), any subset of them.
      self        q0 k0, (heads, cols, 1)
      qk, qkpre   q0 k_j, q0 km_j
      qkchange    q0 (k_j - km_j)
      x           |x_j - x0|      x in q, k, v
      xpre        |xm_j - x0|
      xchange     |x_j - xm_j|"""
    t = {"self": (q0 * k0).sum(-1)[..., None].astype(np.float64)}
    norm = lambda a, b: np.sqrt(((a.astype(np.float64) - b) ** 2).sum(-1)).transpose(1, 2, 0)
    for x, x0 in (("q", q0), ("k", k0), ("v", v0)):
        x0 = x0.astype(np.float64)[None]
        if x in X:
            t[x] = norm(X[x], x0)
        if x + "m" in X:
            t[x + "pre"] = norm(X[x + "m"], x0)
        if x in X and x + "m" in X:
            t[x + "change"] = norm(X[x], X[x + "m"])
    logit = lambda k: np.einsum("hcd,mhcd->hcm", q0, k).astype(np.float64)
    if "k" in X:
        t["qk"] = logit(X["k"])
    if "km" in X:
        t["qkpre"] = logit(X["km"])
    if "k" in X and "km" in X:
        t["qkchange"] = logit(X["k"] - X["km"])
    return t


TERM_NEEDS = {"qk": ("k",), "q": ("q",), "qpre": ("qm",), "qchange": ("q", "qm"), "k": ("k",), "kpre": ("km",),
              "kchange": ("k", "km"), "v": ("v",), "vpre": ("vm",), "vchange": ("v", "vm"), "qkpre": ("km",),
              "qkchange": ("k", "km"), "qkv": ("k", "v"), "qkvpre": ("km", "vm"), "qkvchange": ("k", "v", "vm"),
              "avpre": ("k", "vm"), "avchpre": ("k", "v", "vm"), "qkvo": ("k", "v"),
              "qkbi": ("q", "k"), "jac": ("k", "v"), "vscale": ("k", "v"), "vstab": ("k", "v"),
              "vnorm": ("k", "v"), "cosvscale": ("k", "v")}
MASKED_NEEDS = {k: TERM_NEEDS[k.removesuffix("_s")] for k in VALUE_SCORES if k != "none"}
_SMALL_FIRST = ("q", "qpre", "k", "kpre", "vpre")


def _weights(logit: np.ndarray, own: np.ndarray) -> np.ndarray:
    """softmax over [own, candidates] per head and column -> the weights on the candidates (heads, cols, M)."""
    return _softmax(np.concatenate([own, logit], -1))[..., 1:]


def _renorm(w: np.ndarray) -> np.ndarray:
    w = w.mean(1)
    return w / w.sum(-1, keepdims=True)


def head_terms(kind: str, t: dict) -> np.ndarray:
    """(heads, M) score of the candidates per head from candidate_terms t, larger first.
      qk          softmax of q0 k_j (with the query's own q0 k0), averaged over the columns and renormed to sum
                  to 1 (head_weights); the proposed method
      x, xpre     minus |x_j - x0|, |xm_j - x0| summed over the columns (x in q, k); v: plus |v_j - v0|
                  (how far the head's output moves), vpre: minus |vm_j - v0|
      xchange     |x_j - xm_j| summed over the columns, larger first (x in q, k, v); xchange_s smaller first
      qkpre       as qk from q0 km_j (with q0 k0)
      qkchange    as qk from q0 (k_j - km_j) (the query's own logit taken as 0)
      qkv         sum over the columns of softmax(q0 k_j) |v_j - v0|
      qkvpre      sum over the columns of softmax(q0 km_j) |vm_j - v0|
      qkvchange   sum over the columns of softmax(q0 k_j) |v_j - vm_j|
      avpre       qk / the sum of |vm_j - v0| over the columns
      avchpre     qk times the sum of |v_j - vm_j| over the columns / the sum of |vm_j - v0|
    qkvo (masked_score only) sums softmax(q0 k_j) (v_j - v0) W_O over the heads before the norm; qkbi (masked_score
    only) is qk times how much the candidate's row reads the query's, per head the mean over the columns of the
    softmax of [q_j k_j, q_j k0] on the query (both run with the candidate's future, as in the forecast); jac
    (masked_score only) is the norm of the first-order change of the forecast without reference, J sum over the
    heads and the columns the query reads of softmax(q0 k_j) (v_j - v0) W_O, larger first (jac_s smaller first)."""
    if kind.endswith("_s"):
        return -head_terms(kind[:-2], t)
    if kind in ("q", "qpre", "qchange", "k", "kpre", "kchange", "v", "vpre", "vchange"):
        return (-1 if kind in _SMALL_FIRST else 1) * t[kind].sum(1)
    if kind == "qk":
        return _renorm(_weights(t["qk"], t["self"]))
    if kind == "qkpre":
        return _renorm(_weights(t["qkpre"], t["self"]))
    if kind == "qkchange":
        return _renorm(_weights(t["qkchange"], np.zeros_like(t["self"])))
    if kind == "qkv":
        return (_weights(t["qk"], t["self"]) * t["v"]).sum(1)
    if kind == "qkvpre":
        return (_weights(t["qkpre"], t["self"]) * t["vpre"]).sum(1)
    if kind == "qkvchange":
        return (_weights(t["qk"], t["self"]) * t["vchange"]).sum(1)
    if kind == "avpre":
        return head_terms("qk", t) / np.maximum(t["vpre"].sum(1), 1e-30)
    if kind == "avchpre":
        return head_terms("qk", t) * t["vchange"].sum(1) / np.maximum(t["vpre"].sum(1), 1e-30)
    raise ValueError(kind)


def masked_score(kind: str, q: np.ndarray, k_self: np.ndarray, v_self: np.ndarray, cand: np.ndarray,
                 vecs: dict, jac: np.ndarray = None, cols: np.ndarray = None) -> np.ndarray:
    """(M,) score of one query's candidates cand (pool indices, all valid): head_terms averaged over the
    heads (avpre, avchpre: the heads' mean qk times / over the distances summed over the heads; qkvo: the
    norm, summed over the columns, of softmax(q0 k_j) (v_j - v0) summed over the heads, with the values mapped
    by joint_projection). q, k_self, v_self (heads, cols, d) are the query's, vecs the arrays of every pool
    window (N, heads, cols, d) named as in candidate_terms. jac, jac_s take the query's forecast Jacobian through
    each head's slice of W_O (heads, cols, pred_len, d), raw values, and cols (cols,) 1 where the query reads
    the examples. vscale: qk times 2|v0||v_j| / (|v0|^2 + |v_j|^2) of the heads' outputs, averaged over the
    columns, per head. vstab: qk / (1 + sum |v_j - v0| / sum |v0|), vnorm: qk / sum |v_j|
    (sums over the heads and columns), cosvscale: cos(q0, k_j) times the vscale factor, both averaged
    over the columns, per head."""
    X = {x: vecs[x][cand] for x in MASKED_NEEDS[kind]}
    if kind in ("jac", "jac_s"):
        t = candidate_terms(q, k_self, v_self, {"k": X["k"]})
        w = _weights(t["qk"], t["self"]) * cols[None, :, None]
        d = np.einsum("hcm,hctd,mhcd->mt", w, jac.astype(np.float64),
                      X["v"].astype(np.float64) - v_self.astype(np.float64)[None])
        return (-1 if kind == "jac_s" else 1) * np.sqrt((d ** 2).sum(-1))
    if kind == "qkvo":
        t = candidate_terms(q, k_self, v_self, {"k": X["k"]})
        w = _weights(t["qk"], t["self"])
        g = np.einsum("hcm,mhce->mce", w, X["v"].astype(np.float64) - v_self.astype(np.float64)[None])
        return np.sqrt((g ** 2).sum(-1)).sum(-1)
    if kind in ("vscale", "vstab", "vnorm", "cosvscale"):
        t = candidate_terms(q, k_self, v_self, {"k": X["k"]})
        V, v0 = X["v"].astype(np.float64), v_self.astype(np.float64)
        a = np.sqrt((v0 ** 2).sum(-1))[..., None]
        b = np.sqrt((V ** 2).sum(-1)).transpose(1, 2, 0)
        if kind == "vstab":
            d = np.sqrt(((V - v0[None]) ** 2).sum(-1)).sum((1, 2))
            return head_terms("qk", t).mean(0) / (1 + d / max(a.sum(), 1e-30))
        if kind == "vnorm":
            return head_terms("qk", t).mean(0) / np.maximum(b.sum((0, 1)), 1e-30)
        match = (2 * a * b / np.maximum(a ** 2 + b ** 2, 1e-30)).mean(1)
        if kind == "vscale":
            return (head_terms("qk", t) * match).mean(0)
        K, q0 = X["k"].astype(np.float64), q.astype(np.float64)
        cos = np.einsum("mhcd,hcd->hcm", K, q0) / np.maximum(
            np.sqrt((K ** 2).sum(-1)).transpose(1, 2, 0) * np.sqrt((q0 ** 2).sum(-1))[..., None], 1e-30)
        return (cos.mean(1) * match).mean(0)
    if kind == "qkbi":
        t = candidate_terms(q, k_self, v_self, {"k": X["k"]})
        Q = X["q"].astype(np.float64)
        rev = np.einsum("mhcd,hcd->hcm", Q, k_self.astype(np.float64))
        own = np.einsum("mhcd,mhcd->hcm", Q, X["k"].astype(np.float64))
        reads = (1 / (1 + np.exp(own - rev))).mean(1)
        return (head_terms("qk", t) * reads).mean(0)
    t = candidate_terms(q, k_self, v_self, X)
    if kind == "avpre":
        return head_terms("qk", t).mean(0) / np.maximum(t["vpre"].sum((0, 1)), 1e-30)
    if kind == "avchpre":
        return head_terms("qk", t).mean(0) * t["vchange"].sum((0, 1)) / np.maximum(t["vpre"].sum((0, 1)), 1e-30)
    return head_terms(kind, t).mean(0)


STATES = ("out", "in", "final")

