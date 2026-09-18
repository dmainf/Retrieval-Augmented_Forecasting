"""Retrieval over the sliding windows of a single channel, split into three axes.

The database holds full [context | future] windows. Picking which of them to put
in front of the query is three independent decisions, and this module keeps them
independent so any combination can be run as an ablation:

  encode   what a window is turned into before anything is compared
  pool     which database windows are even allowed to be candidates
  select   which K of those candidates end up in the context

The default `raw / all / topk` is plain nearest-neighbour retrieval and is what
every RAF baseline does. Everything else exists so the two open questions --
how to build a same-regime pool, and how to choose K out of it -- can be varied
one at a time.

encode (ENCODERS)
  raw       the context values as they are
  zscore    the context, per-window standardized; drops level and scale, so two
            windows match on shape alone
  spectral  |rFFT| of the centered context; matches on periodic structure and
            ignores phase
  future    an oracle. Keys come from the *future* half, and the query key is
            its true future, which is only knowable after the fact. Not a usable
            method: it upper-bounds what any context-side retriever can reach.
  predicted the runnable half of the oracle. Keys still come from the database
            futures, but the query key is a *forecast* of the query's future
            produced by a first pass with no retrieval -- never the true one.
            predict.py substitutes the forecast into the query window, so the
            retriever treats it exactly like `future`.
  learned   a checkpoint from encoder.py, trained so that closeness in its space
            means a correlated future. Sees only contexts, so it stays usable at
            inference; needs `encoder_path`.

  `key_dim` optionally follows the encoder with a PCA fit on the database keys.
  This matters for the eopt selector: lambda_min of a sum of K rank-one matrices
  is identically zero whenever the key dimension exceeds K, so the criterion is
  vacuous at the raw 96 dimensions and only becomes meaningful once the keys are
  projected below K.

pool (POOLS)
  all       no restriction; the pool is the whole database
  topm      the nearest `pool_size` windows -- a candidate pool in the sense
            SARAF uses, wide enough that a selector has something to choose from
  cluster   only the database windows sharing the query's k-means cluster, then
            the nearest `pool_size` of those. A crude stand-in for a regime.

select (SELECTORS)
  topk      the K nearest. What everyone does.
  mmr       greedy relevance-vs-diversity, the SARAF selector
  dpp       greedy MAP of a DPP, i.e. maximize log det of the selected kernel.
            That is D-optimality (volume), not E-optimality.
  eopt      greedy maximization of lambda_min(sum_i x_i x_i^T), i.e. E-optimal
            design. Greedy carries no approximation guarantee here; it is the
            baseline a guaranteed algorithm would have to beat.

Distances are always "smaller is closer" whatever the metric, so they stay
comparable across runs. The windows returned by `gather` keep their raw values,
which is what gets concatenated in front of the query.
"""
import faiss
import numpy as np

METRICS = ("cosine", "euclidean", "correlation")
ENCODERS = ("raw", "zscore", "spectral", "future", "predicted", "learned")
POOLS = ("all", "topm", "cluster")
SELECTORS = ("topk", "mmr", "dpp", "eopt")

EPS = 1e-8


def _encode(windows: np.ndarray, seq_len: int, pred_len: int, encoder: str,
            model=None, batch_size: int = 2048) -> np.ndarray:
    """(N, seq_len + pred_len) full windows -> (N, D) feature keys."""
    if encoder == "learned":
        # torch is imported here, not at module scope: faiss and torch each ship
        # their own OpenMP and faiss has to load first (see predict.py)
        import torch
        out = []
        with torch.no_grad():
            for lo in range(0, len(windows), batch_size):
                chunk = torch.from_numpy(
                    np.ascontiguousarray(windows[lo:lo + batch_size, :seq_len],
                                         dtype=np.float32))
                out.append(model(chunk).numpy())
        return np.ascontiguousarray(np.concatenate(out), dtype=np.float32)

    if encoder in ("future", "predicted"):
        x = windows[:, seq_len:seq_len + pred_len]
    elif encoder in ENCODERS:
        x = windows[:, :seq_len]
    else:
        raise ValueError(f"unknown encoder {encoder}; choose from {ENCODERS}")
    x = x.astype(np.float32)

    if encoder in ("raw", "future", "predicted"):
        keys = x
    elif encoder == "zscore":
        keys = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + EPS)
    else:
        keys = np.abs(np.fft.rfft(x - x.mean(axis=1, keepdims=True), axis=1))
    return np.ascontiguousarray(keys, dtype=np.float32)


def _metric_keys(keys: np.ndarray, metric: str) -> np.ndarray:
    """Map keys into the space faiss needs, since it only does inner product or L2."""
    if metric == "euclidean":
        out = keys.copy()
    elif metric == "cosine":
        out = keys.copy()
        faiss.normalize_L2(out)
    elif metric == "correlation":
        out = keys - keys.mean(axis=1, keepdims=True)
        faiss.normalize_L2(out)
    else:
        raise ValueError(f"unknown metric {metric}; choose from {METRICS}")
    return np.ascontiguousarray(out, dtype=np.float32)


def _pairwise(q: np.ndarray, db: np.ndarray, metric: str) -> np.ndarray:
    """(B, D) against (B, M, D) -> (B, M) distances, smaller is closer.

    Squared L2 for euclidean, matching what IndexFlatL2 reports.
    """
    if metric == "euclidean":
        return np.sum((db - q[:, None, :]) ** 2, axis=-1)
    return 1.0 - np.einsum("bd,bmd->bm", q, db)


def _cosine_gram(keys: np.ndarray) -> np.ndarray:
    unit = keys / (np.linalg.norm(keys, axis=-1, keepdims=True) + EPS)
    return np.einsum("bmd,bnd->bmn", unit, unit)


def _select_topk(dist, k, **_):
    return np.argsort(dist, axis=1)[:, :k]


def _select_mmr(dist, k, keys, diversity, **_):
    """Greedy max of (1-w) * relevance - w * max similarity to what is already chosen."""
    b, m = dist.shape
    sim = _cosine_gram(keys)
    rel = -dist / (np.median(dist, axis=1, keepdims=True) + EPS)

    chosen = np.empty((b, k), dtype=np.int64)
    chosen[:, 0] = np.argmin(dist, axis=1)
    worst = sim[np.arange(b), chosen[:, 0], :]
    for j in range(1, k):
        score = (1.0 - diversity) * rel - diversity * worst
        np.put_along_axis(score, chosen[:, :j], -np.inf, axis=1)
        chosen[:, j] = np.argmax(score, axis=1)
        worst = np.maximum(worst, sim[np.arange(b), chosen[:, j], :])
    return chosen


def _select_dpp(dist, k, keys, diversity, **_):
    """Greedy DPP MAP on L = q S q, with q a relevance weight and S the key cosine.

    Maximizes log det, i.e. the volume the selected keys span: D-optimal, not
    E-optimal. See the module docstring.
    """
    b, m = dist.shape
    sim = _cosine_gram(keys).astype(np.float64)
    scale = dist / (np.median(dist, axis=1, keepdims=True) + EPS)
    quality = np.exp(-(1.0 - diversity) * scale)
    L = quality[:, :, None] * sim * quality[:, None, :]
    L[:, np.arange(m), np.arange(m)] += 1e-6

    rows = np.arange(b)
    chosen = np.empty((b, k), dtype=np.int64)
    gain = L[:, np.arange(m), np.arange(m)].copy()
    cis = np.zeros((b, k, m))
    for j in range(k):
        chosen[:, j] = np.argmax(gain, axis=1)
        idx = chosen[:, j]
        di = np.sqrt(np.maximum(gain[rows, idx], 1e-12))
        prev = np.einsum("bjm,bj->bm", cis[:, :j, :], cis[:, :j, :][rows, :, idx])
        ei = (L[rows, idx, :] - prev) / di[:, None]
        cis[:, j, :] = ei
        gain = np.maximum(gain - ei ** 2, 0.0)
        np.put_along_axis(gain, chosen[:, :j + 1], -np.inf, axis=1)
    return chosen


def _select_eopt(dist, k, keys, diversity, **_):
    """Greedy max of lambda_min(sum_i x_i x_i^T) minus a relevance penalty.

    The sum of k rank-one matrices has rank at most k, so lambda_min is exactly
    zero while the key dimension exceeds k and the criterion says nothing.
    Project the keys below k first (`key_dim`).
    """
    b, m, d = keys.shape
    if d >= k:
        raise ValueError(
            f"eopt needs the key dimension ({d}) to be below k ({k}); lambda_min "
            f"is identically zero otherwise. Set --key-dim below --top-k.")

    x = keys.astype(np.float64)
    outer = np.einsum("bmi,bmj->bmij", x, x)
    penalty = diversity * dist / (np.median(dist, axis=1, keepdims=True) + EPS)

    rows = np.arange(b)
    chosen = np.empty((b, k), dtype=np.int64)
    chosen[:, 0] = np.argmin(dist, axis=1)
    acc = outer[rows, chosen[:, 0]]
    for j in range(1, k):
        lam = np.linalg.eigvalsh(acc[:, None, :, :] + outer)[..., 0]
        score = lam - penalty
        np.put_along_axis(score, chosen[:, :j], -np.inf, axis=1)
        chosen[:, j] = np.argmax(score, axis=1)
        acc = acc + outer[rows, chosen[:, j]]
    return chosen


_SELECTORS = {"topk": _select_topk, "mmr": _select_mmr,
              "dpp": _select_dpp, "eopt": _select_eopt}


class WindowRetriever:
    """One channel's database windows, addressed as encode / pool / select."""

    def __init__(self, db_windows: np.ndarray, seq_len: int, pred_len: int,
                 metric: str = "euclidean", encoder: str = "raw", key_dim: int = 0,
                 encoder_path: str = "", pool: str = "all", pool_size: int = 100,
                 n_clusters: int = 8, selector: str = "topk", diversity: float = 0.5,
                 vol_match: float = 0.0, hub_penalty: float = 0.0, rank_offset: float = 0.0,
                 seed: int = 0):
        if pool not in POOLS:
            raise ValueError(f"unknown pool {pool}; choose from {POOLS}")
        if selector not in SELECTORS:
            raise ValueError(f"unknown selector {selector}; choose from {SELECTORS}")
        if pool == "all" and selector != "topk":
            raise ValueError(
                f"selector={selector} builds an M x M kernel per query, so it needs a "
                f"bounded pool. Use --pool topm (or cluster) with --pool-size.")
        self.pool_size = pool_size

        self.windows = np.ascontiguousarray(db_windows, dtype=np.float32)
        self.seq_len, self.pred_len = seq_len, pred_len
        self.metric, self.encoder = metric, encoder
        self.pool, self.pool_size = pool, pool_size
        self.selector, self.diversity = selector, diversity
        self.vol_match, self.hub_penalty = vol_match, hub_penalty
        self.size = len(self.windows)

        # Start the top-K at a rank other than 0, as a fraction of the candidates
        # available to that query. With encoder=future the candidates are ordered
        # by how close their futures are to the query's true future, so this is a
        # dial on retrieval quality measured in the space that actually matters:
        # 0 is the oracle, larger values reach windows whose futures resemble the
        # answer less and less. Sweeping it traces the curve that says whether a
        # partly-good retriever buys a partly-good result, or whether the gain
        # only exists at the answer itself.
        self.rank_offset = float(rank_offset)
        if not 0.0 <= self.rank_offset < 1.0:
            raise ValueError(f"rank_offset must be in [0, 1), found {rank_offset}")

        # how volatile each database window's future actually was. Known at
        # retrieval time -- these are past windows -- so matching against it is
        # legitimate; what has to be estimated is the *query's* future volatility.
        self.db_vol = self.windows[:, seq_len:seq_len + pred_len].std(axis=1)

        self.model = None
        if encoder == "learned":
            if not encoder_path:
                raise ValueError("encoder=learned needs --encoder-path")
            from encoder import load_encoder
            self.model, config = load_encoder(encoder_path)
            if config["seq_len"] != seq_len:
                raise ValueError(
                    f"checkpoint was trained for seq_len {config['seq_len']}, "
                    f"but this run uses {seq_len}")

        keys = _encode(self.windows, seq_len, pred_len, encoder, self.model)
        self.pca = None
        if key_dim > 0:
            self.pca = faiss.PCAMatrix(keys.shape[1], key_dim)
            self.pca.train(keys)
            keys = self.pca.apply(keys)
        self.keys = _metric_keys(keys, metric)
        self.key_dim = self.keys.shape[1]

        self.index = self._new_index()
        self.index.add(self.keys)

        # hubness: in high dimensions a few windows sit near everything and are
        # returned for every query regardless of the query. Their mean distance
        # to a random sample of the database measures that, and subtracting it
        # discounts "close to everyone" in favour of "close to this query".
        self.typicality = None
        if hub_penalty > 0:
            rng = np.random.default_rng(seed)
            probe = self.keys[rng.choice(self.size, min(512, self.size), replace=False)]
            d = _pairwise(probe, np.broadcast_to(self.keys, (len(probe),) + self.keys.shape),
                          metric) if False else None
            if metric == "euclidean":
                d = ((self.keys[None, :, :] - probe[:, None, :]) ** 2).sum(-1)
            else:
                d = 1.0 - probe @ self.keys.T
            self.typicality = d.mean(axis=0)

        if pool == "cluster":
            km = faiss.Kmeans(self.key_dim, n_clusters, niter=25, seed=seed)
            km.train(self.keys)
            self.centroids = np.ascontiguousarray(km.centroids)
            labels = km.index.search(self.keys, 1)[1].ravel()
            self.members = [np.flatnonzero(labels == c) for c in range(n_clusters)]
            self.cluster_index = []
            for member in self.members:
                sub = self._new_index()
                sub.add(np.ascontiguousarray(self.keys[member]))
                self.cluster_index.append(sub)
            smallest = min(len(m) for m in self.members)
            if smallest < self.pool_size:
                print(f"    pool_size {self.pool_size} -> {smallest} "
                      f"(smallest of {n_clusters} clusters holds {smallest} windows)")
                self.pool_size = smallest

    def _rescore(self, cand: np.ndarray, dist: np.ndarray,
                 target_vol: np.ndarray) -> np.ndarray:
        """Combine raw distance with the two anti-averaging corrections."""
        score = dist / (np.median(dist, axis=1, keepdims=True) + EPS)
        if self.vol_match > 0 and target_vol is not None:
            t = np.asarray(target_vol, dtype=np.float64)[:, None]
            gap = np.abs(self.db_vol[cand] - t) / (t + EPS)
            score = score + self.vol_match * gap
        if self.hub_penalty > 0 and self.typicality is not None:
            typ = self.typicality[cand]
            score = score - self.hub_penalty * typ / (np.median(typ, axis=1, keepdims=True) + EPS)
        return score

    def _new_index(self):
        return (faiss.IndexFlatL2(self.key_dim) if self.metric == "euclidean"
                else faiss.IndexFlatIP(self.key_dim))

    def _query_keys(self, query_windows: np.ndarray) -> np.ndarray:
        keys = _encode(np.asarray(query_windows, dtype=np.float32),
                       self.seq_len, self.pred_len, self.encoder, self.model)
        if self.pca is not None:
            keys = self.pca.apply(keys)
        return _metric_keys(keys, self.metric)

    def _candidates(self, qk: np.ndarray) -> np.ndarray:
        """(B, D) query keys -> (B, M) global database indices."""
        if self.pool == "all":
            return np.tile(np.arange(self.size, dtype=np.int64), (len(qk), 1))
        if self.pool == "topm":
            return self.index.search(qk, min(self.pool_size, self.size))[1].astype(np.int64)

        assign = faiss.knn(qk, self.centroids, 1)[1].ravel()
        out = np.empty((len(qk), self.pool_size), dtype=np.int64)
        for c in np.unique(assign):
            rows = np.flatnonzero(assign == c)
            local = self.cluster_index[c].search(
                np.ascontiguousarray(qk[rows]), self.pool_size)[1]
            out[rows] = self.members[c][local]
        return out

    def _skip(self, available: int, keep: int = 0) -> int:
        """How many of the nearest candidates to step over before taking K."""
        if self.rank_offset == 0.0:
            return 0
        return max(0, min(int(self.rank_offset * available), available - keep - 1))

    def retrieve_past(self, query_windows: np.ndarray, top_k: int,
                      limit: np.ndarray, target_vol=None):
        """Like `retrieve`, but row i may only use database indices <= limit[i].

        Over-searching and filtering does not work here: at stride 1 the windows
        overlapping the query are near-duplicates of it, so they monopolize the
        nearest neighbours and are exactly the ones that must be excluded. The
        cut has to happen inside the search, which is what IDSelectorRange does.
        """
        if self.pool == "cluster":
            raise ValueError("pool=cluster is not supported with an expanding "
                             "database; cluster membership would have to be "
                             "recomputed per query")
        qk = self._query_keys(query_windows)
        rescoring = self.vol_match > 0 or self.hub_penalty > 0
        plain = self.selector == "topk" and not rescoring
        n_keep = top_k if plain else min(self.pool_size, self.size)

        cand = np.empty((len(qk), n_keep), dtype=np.int64)
        dist = np.empty((len(qk), n_keep), dtype=np.float32)
        for row in range(len(qk)):
            params = faiss.SearchParameters()
            available = int(limit[row]) + 1
            params.sel = faiss.IDSelectorRange(0, available)
            skip = self._skip(available, n_keep)
            scores, indices = self.index.search(qk[row:row + 1], skip + n_keep,
                                                params=params)
            cand[row] = indices[0, skip:]
            dist[row] = (scores[0, skip:] if self.metric == "euclidean"
                         else 1.0 - scores[0, skip:])
        if plain:
            return cand, dist

        scored = self._rescore(cand, dist, target_vol)
        picks = _SELECTORS[self.selector](dist=scored, k=top_k, keys=self.keys[cand],
                                          diversity=self.diversity)
        indices = np.take_along_axis(cand, picks, axis=1)
        distances = np.take_along_axis(dist, picks, axis=1)
        order = np.argsort(distances, axis=1)
        return (np.take_along_axis(indices, order, axis=1),
                np.take_along_axis(distances, order, axis=1))

    def retrieve(self, query_windows: np.ndarray, top_k: int, target_vol=None):
        """query_windows (B, seq_len + pred_len) -> (indices, distances), both (B, k).

        Full windows go in, not just contexts, because the encoder decides which
        part it needs and the oracle encoder needs the future half. Results come
        back sorted by distance, so column 0 is the closest match whatever order
        the selector picked them in.
        """
        qk = self._query_keys(query_windows)
        k = min(top_k, self.size)

        if (self.pool == "all" and self.selector == "topk"
                and self.vol_match == 0 and self.hub_penalty == 0):
            skip = self._skip(self.size)
            scores, indices = self.index.search(qk, skip + k)
            distances = scores if self.metric == "euclidean" else 1.0 - scores
            return indices[:, skip:].astype(np.int64), distances[:, skip:]

        cand = self._candidates(qk)
        if cand.shape[1] < k:
            raise ValueError(f"pool holds {cand.shape[1]} candidates, fewer than k={k}")
        cand_keys = self.keys[cand]
        dist = _pairwise(qk, cand_keys, self.metric)
        scored = self._rescore(cand, dist, target_vol)

        picks = _SELECTORS[self.selector](dist=scored, k=k, keys=cand_keys,
                                          diversity=self.diversity)
        indices = np.take_along_axis(cand, picks, axis=1)
        distances = np.take_along_axis(dist, picks, axis=1)
        order = np.argsort(distances, axis=1)
        return (np.take_along_axis(indices, order, axis=1),
                np.take_along_axis(distances, order, axis=1))

    def constructed_analogue(self, contexts: np.ndarray, pool_size: int, lam: float,
                             limit: np.ndarray = None) -> np.ndarray:
        """(B, seq_len) query contexts -> (B, pred_len) constructed-analogue forecast.

        Solves for weights whose combination of retrieved *contexts* reconstructs
        the query context, then applies those same weights to the futures:

            w = argmin ||X^T w - x_q||^2 + lam ||w||^2 ,   y_CA = Y^T w

        Averaging the k nearest futures is a local-constant (Nadaraya-Watson)
        estimate, whose bias is J(x_bar - x_q) -- the neighbours' contexts sit
        off-centre from the query, pulled toward the bulk of the corpus, so the
        averaged future is pulled toward the bulk too. That is the regression to
        the mean this project keeps running into. Solving for weights that
        reconstruct x_q cancels that first-order term, and because the weights
        may be negative the result can leave the convex hull of the retrieved
        futures -- it can be sharper than any of them.

        Everything is done on anomalies (each window minus its own context mean)
        so that level drift does not dominate the reconstruction.
        """
        ctx = np.asarray(contexts, dtype=np.float64)
        b = len(ctx)
        m = min(pool_size, self.size)
        keys = self._query_keys(np.concatenate(
            [ctx, np.zeros((b, self.windows.shape[1] - self.seq_len))], axis=1))

        if limit is None:
            cand = self.index.search(keys, m)[1]
        else:
            cand = np.empty((b, m), dtype=np.int64)
            for row in range(b):
                params = faiss.SearchParameters()
                params.sel = faiss.IDSelectorRange(0, int(limit[row]) + 1)
                cand[row] = self.index.search(keys[row:row + 1], m, params=params)[1][0]

        win = self.windows[cand].astype(np.float64)
        X = win[:, :, :self.seq_len]
        Y = win[:, :, self.seq_len:self.seq_len + self.pred_len]
        mx = X.mean(axis=-1, keepdims=True)
        Xa, Ya = X - mx, Y - mx

        qm = ctx.mean(axis=-1, keepdims=True)
        xqa = ctx - qm

        G = np.einsum("bmd,bnd->bmn", Xa, Xa)
        ridge = lam * np.trace(G, axis1=1, axis2=2) / m
        G[:, np.arange(m), np.arange(m)] += ridge[:, None]
        rhs = np.einsum("bmd,bd->bm", Xa, xqa)[:, :, None]
        w = np.linalg.solve(G, rhs)[:, :, 0]
        return (np.einsum("bmd,bm->bd", Ya, w) + qm).astype(np.float32)

    def gather(self, indices: np.ndarray) -> np.ndarray:
        """indices (B, k) -> retrieved windows (B, k, win_len) at raw scale."""
        return self.windows[indices]
