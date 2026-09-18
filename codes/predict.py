"""Entry point: forecast every query window and save predictions with ground truth.

No metrics are computed here. The output tables hold the raw forecasts, the true
future values, and which windows were retrieved, so any metric or analysis can be
done afterwards.
"""
import json
import os

# faiss and torch each ship their own OpenMP; allow the duplicate and pin both to
# one thread. Must happen before either library is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from arguments import parse_args          # pulls in faiss (before torch)
from dataset import TimeSeriesData
from retriever import WindowRetriever
from raf_model import RAFModel, concat_retrieved

import torch


def _ablate(windows: np.ndarray, mode: str, seq_len: int, pred_len: int,
            db: np.ndarray, rng, query_ctx: np.ndarray = None,
            query_future: np.ndarray = None) -> np.ndarray:
    """Replace the future half of retrieved windows, leaving their contexts intact.

    The contexts are what retrieval matched on; the futures are the part that
    carries "and then this happened". If the backbone is doing anything like
    in-context learning, breaking the futures must cost accuracy.

    `truth-future` runs the other way. It writes the query's own true future into
    every retrieved window, which is not a retrieval a search could ever perform
    -- the point is that it sets the retrieved-future distance to exactly zero.
    The oracle only reaches 0.2 of the median, because no database window matches
    perfectly, so the two measure different ceilings: the oracle bounds what
    *selection* can buy from this database, and this bounds what the *container*
    can transmit at all. If even a perfect reference moves little, no retriever
    is worth building.
    """
    out = np.array(windows, dtype=np.float32)
    fut = slice(seq_len, seq_len + pred_len) if out.shape[-1] > pred_len else slice(0, pred_len)
    pred_len = min(pred_len, out.shape[-1])
    b, k, _ = out.shape
    if mode == "shuffle-future":
        for i in range(b):
            out[i, :, fut] = out[i, rng.permutation(k), fut]
    elif mode == "random-future":
        pick = rng.integers(0, len(db), size=(b, k))
        out[:, :, fut] = db[pick][:, :, fut]
    elif mode == "drop-future":
        out[:, :, fut] = np.nan
    elif mode == "shuffle-time":
        # keeps the exact multiset of values in each future, destroys the waveform
        for i in range(b):
            for j in range(k):
                out[i, j, fut] = out[i, j, fut][rng.permutation(pred_len)]
    elif mode == "gauss-future":
        block = out[:, :, fut]
        mu = np.nanmean(block, axis=-1, keepdims=True)
        sd = np.nanstd(block, axis=-1, keepdims=True)
        out[:, :, fut] = mu + sd * rng.standard_normal(block.shape)
    elif mode == "truth-future":
        if query_future is None:
            raise ValueError("truth-future needs the query's future")
        out[:, :, fut] = query_future[:, None, :pred_len]
    elif mode == "self-truth":
        # the query's own window, answer included, handed over as the example.
        # Not a retrieval at all -- it is the perfect reference, and it asks the
        # prior question: if the example were exactly right, would it help?
        if query_future is None or query_ctx is None:
            raise ValueError("self-truth needs the query's context and future")
        if out.shape[-1] != seq_len + pred_len:
            raise ValueError("self-truth needs whole windows (--example-part window)")
        out[:, :, :seq_len] = query_ctx[:, None, :seq_len]
        out[:, :, fut] = query_future[:, None, :pred_len]
    elif mode == "gauss-local":
        # statistics taken from the query itself: retrieval contributes nothing
        mu = np.nanmean(query_ctx, axis=-1)[:, None, None]
        sd = np.nanstd(query_ctx, axis=-1)[:, None, None]
        out[:, :, fut] = mu + sd * rng.standard_normal(out[:, :, fut].shape)
    else:
        raise ValueError(mode)
    return out


def _align(windows: np.ndarray, query_ctx: np.ndarray, mode: str,
           seq_len: int, pred_len: int) -> np.ndarray:
    """Put each retrieved future on the query's current level.

    A retrieved window carries whatever level the series had when it happened.
    Concatenated raw, its shape is unusable and only its mean and spread survive.
    The shift is computed from the retrieved *context*, which is observed, so
    nothing about the retrieved future is used to place it.
    """
    out = np.array(windows, dtype=np.float32)
    ctx = out[:, :, :seq_len]
    fut = slice(seq_len, seq_len + pred_len)
    q = query_ctx.astype(np.float32)
    eps = 1e-8
    if mode == "last":
        out[:, :, fut] += q[:, -1][:, None, None] - ctx[:, :, -1:]
    elif mode == "mean":
        out[:, :, fut] += q.mean(-1)[:, None, None] - ctx.mean(-1, keepdims=True)
    elif mode == "mean-scale":
        mu_r = ctx.mean(-1, keepdims=True)
        sd_r = ctx.std(-1, keepdims=True) + eps
        out[:, :, fut] = ((out[:, :, fut] - mu_r) / sd_r
                          * q.std(-1)[:, None, None] + q.mean(-1)[:, None, None])
    else:
        raise ValueError(mode)
    return out


def pick_device(name: str, gpu: int) -> torch.device:
    if name != "auto":
        return torch.device(f"cuda:{gpu}" if name == "cuda" else name)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_run_name(args) -> str:
    if args.run_name:
        return args.run_name
    stem = f"{args.dataset}_c{args.seq_len}_h{args.pred_len}_k{args.top_k}"
    if args.retrieval_mode != "search":
        stem += f"_{args.retrieval_mode}"
    if "chronos-2" in args.chronos_model:
        stem += "_c2"
    if args.inject != "concat" and args.top_k > 0:
        stem += f"_{args.inject}"
    if args.query_context > 0:
        stem += f"_qc{args.query_context}"
    if args.stage1_context > 0:
        stem += f"_s1c{args.stage1_context}"
    if args.top_k > 0:
        stem += f"_{args.retrieval_metric}"
        if args.encoder != "raw":
            stem += f"_{args.encoder}"
        if args.key_dim > 0:
            stem += f"_d{args.key_dim}"
        if args.pool != "all":
            stem += f"_{args.pool}{args.pool_size}"
        if args.selector != "topk":
            stem += f"_{args.selector}{args.diversity:g}"
        if args.oracle_rank > 0:
            stem += f"_r{args.oracle_rank:g}"
        if args.ca_shift > 0:
            stem += f"_ca{args.ca_shift:g}"
        if args.align != "none":
            stem += f"_al{args.align}"
        if args.example_part != "window":
            stem += f"_{args.example_part}only"
        if args.example_ablation != "none":
            stem += f"_{args.example_ablation}"
        if args.vol_match > 0:
            stem += f"_vm{args.vol_match:g}"
        if args.hub_penalty > 0:
            stem += f"_hub{args.hub_penalty:g}"
        if args.separator != "none":
            stem += f"_sep{args.separator}{args.separator_len or ''}"
    return stem


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rng = np.random.default_rng(args.seed)
    device = pick_device(args.device, args.gpu)
    data = TimeSeriesData(args.root_path, args.dataset, args.seq_len, args.pred_len,
                          args.channels)
    model = RAFModel(args.chronos_model, args.pred_len, device)

    eval_stride = args.eval_stride if args.eval_stride > 0 else data.win_len
    use_retrieval = args.top_k > 0
    parallel = args.inject == "parallel" and use_retrieval
    sep_len = 0
    if args.separator != "none" and not parallel:
        sep_len = args.separator_len if args.separator_len > 0 else model.patch_size
    query_context = args.query_context or args.seq_len
    example_len = {"window": data.win_len, "future": args.pred_len,
                   "context": args.seq_len}[args.example_part]
    if parallel:
        # examples sit beside the history rather than inside it, so the only
        # budget they spend is rows; top_k is bounded by memory, not by
        # context_length, and there is no gap to fill with a separator
        if not model.is_chronos2:
            raise ValueError(
                "--inject parallel needs the chronos-2 backbone; chronos-bolt is "
                "univariate and takes no covariates"
            )
        if args.example_part != "window":
            raise ValueError(
                "--inject parallel places each example's context in the history "
                "and its future in the horizon, so it needs both halves; use "
                "--example-part window"
            )
        if query_context > model.context_length:
            raise ValueError(
                f"query_context {query_context} exceeds the backbone "
                f"context_length {model.context_length}."
            )
    elif use_retrieval:
        limit = model.max_top_k(query_context, example_len, sep_len)
        if args.top_k > limit:
            raise ValueError(
                f"top_k {args.top_k} does not fit: {args.top_k} x "
                f"({example_len} + {sep_len}) + {query_context} exceeds the backbone "
                f"context_length {model.context_length}. Maximum is {limit}."
            )

    print(f"{args.dataset}: {data.n_channels} channels, device={device}, "
          f"quantiles={model.quantiles}")
    print(f"window={args.seq_len}+{args.pred_len}, eval_stride={eval_stride}"
          + (f", top_k={args.top_k}, {args.encoder}/{args.pool}/{args.selector}"
             f", metric={args.retrieval_metric}" if use_retrieval else "")
          + (f", separator={args.separator} x{sep_len}" if sep_len else "")
          + (f", inject=parallel ({args.top_k + 1} rows x {query_context})"
             if parallel else ""))

    pred_rows, true_rows, meta_rows, log_rows = [], [], [], []
    for channel, name in enumerate(data.channels):
        query = data.windows(args.eval_split, channel, eval_stride)
        contexts = query[:, :args.seq_len]
        futures = query[:, args.seq_len:args.seq_len + args.pred_len]
        starts = np.arange(len(query), dtype=np.int64) * eval_stride

        recent = None
        if use_retrieval and args.retrieval_mode == "recent":
            recent = data.preceding_windows(args.eval_split, channel, starts, args.top_k)

        retriever, past_limit = None, None
        if use_retrieval and args.retrieval_mode == "search":
            if args.retrieval_split == "past":
                db = data.all_windows(channel, args.retrieval_stride)
                past_limit = data.past_limit(args.eval_split, starts,
                                             args.retrieval_stride)
            else:
                db = data.windows(args.retrieval_split, channel, args.retrieval_stride)
            retriever = WindowRetriever(
                db, args.seq_len, args.pred_len, metric=args.retrieval_metric,
                encoder=args.encoder, key_dim=args.key_dim,
                encoder_path=args.encoder_path, pool=args.pool,
                pool_size=args.pool_size, n_clusters=args.n_clusters,
                selector=args.selector, diversity=args.diversity,
                vol_match=args.vol_match, hub_penalty=args.hub_penalty,
                rank_offset=args.oracle_rank, seed=args.seed)

        # best observable proxy for the query's future volatility (corr 0.425);
        # the forecast's quantile spread is weaker (0.226 at context 96)
        target_vol = contexts.std(axis=1)

        model_context = contexts
        if args.query_context > 0:
            model_context = data.history(args.eval_split, channel, starts,
                                         args.query_context)

        stage1_input = contexts
        if args.stage1_context > 0:
            stage1_input = data.history(args.eval_split, channel, starts,
                                        args.stage1_context)

        def forecast(search_windows, keep_log: bool, plain=None):
            """search_windows (n, win_len) or None -> quantile forecasts (n, q, pred_len).

            Only the future half of search_windows is ever read, and only by the
            future-keyed encoders; everything else keys off the context.
            """
            out = []
            source = contexts if plain is None else plain

            def run(context, retrieved):
                """Hand one batch to the backbone through the chosen container."""
                if retrieved is None:
                    return model.predict(context)
                if parallel:
                    return model.predict_parallel(context, retrieved, args.seq_len)
                return model.predict(concat_retrieved(
                    context, retrieved, args.concat_order, args.separator, sep_len))

            for lo in range(0, len(query), args.batch_size):
                hi = min(lo + args.batch_size, len(query))

                if retriever is None and recent is None:
                    out.append(run(torch.from_numpy(source[lo:hi]), None).cpu().numpy())
                    continue
                if recent is not None:
                    out.append(run(torch.from_numpy(model_context[lo:hi]),
                                   torch.from_numpy(recent[lo:hi])).cpu().numpy())
                    continue
                context = torch.from_numpy(model_context[lo:hi])
                if past_limit is None:
                    indices, distances = retriever.retrieve(
                        search_windows[lo:hi], args.top_k, target_vol[lo:hi])
                else:
                    indices, distances = retriever.retrieve_past(
                        search_windows[lo:hi], args.top_k, past_limit[lo:hi],
                        target_vol[lo:hi])
                gathered = retriever.gather(indices)
                if args.ca_shift > 0:
                    y_ca = retriever.constructed_analogue(
                        contexts[lo:hi], args.ca_pool, args.ca_shift,
                        None if past_limit is None else past_limit[lo:hi])
                    f = slice(args.seq_len, args.seq_len + args.pred_len)
                    gathered[:, :, f] += (
                        y_ca[:, None, :] - gathered[:, :, f].mean(axis=1, keepdims=True))
                if args.align != "none":
                    gathered = _align(gathered, contexts[lo:hi], args.align,
                                      args.seq_len, args.pred_len)
                if args.example_part == "future":
                    gathered = gathered[:, :, args.seq_len:args.seq_len + args.pred_len]
                elif args.example_part == "context":
                    gathered = gathered[:, :, :args.seq_len]
                if args.example_ablation != "none":
                    gathered = _ablate(gathered, args.example_ablation,
                                       args.seq_len, args.pred_len,
                                       retriever.windows, rng, contexts[lo:hi],
                                       futures[lo:hi])
                retrieved = torch.from_numpy(gathered)
                if keep_log and args.save_retrieval:
                    b, k = indices.shape
                    log_rows.append(pd.DataFrame({
                        "channel": name,
                        "window": np.repeat(np.arange(lo, hi, dtype=np.int64), k),
                        "start": np.repeat(starts[lo:hi], k),
                        "rank": np.tile(np.arange(k, dtype=np.int64), b),
                        "db_index": indices.reshape(-1).astype(np.int64),
                        "db_start": (indices.reshape(-1).astype(np.int64)
                                     * args.retrieval_stride),
                        "distance": distances.reshape(-1).astype(np.float32),
                    }))
                out.append(run(context, retrieved).cpu().numpy())
            return np.concatenate(out, axis=0)

        if use_retrieval and args.encoder == "predicted" and recent is None:
            # Two-stage, optionally repeated. Round 0 forecasts with no retrieval
            # at all; each later round hands the retriever the previous round's
            # forecast in place of the future it is not allowed to know, so a
            # better forecast buys a better query. The true future is never read.
            median = model.quantiles.index(0.5)
            saved, retriever = retriever, None
            guess = forecast(None, keep_log=False, plain=stage1_input)[:, median, :]
            retriever = saved
            if args.oracle_mix > 0:
                guess = (1.0 - args.oracle_mix) * guess + args.oracle_mix * futures
            for round_i in range(args.retrieval_rounds):
                search_windows = query.copy()
                search_windows[:, args.seq_len:args.seq_len + args.pred_len] = guess
                last = round_i == args.retrieval_rounds - 1
                quantiles = forecast(search_windows, keep_log=last)
                guess = quantiles[:, median, :]
        else:
            quantiles = forecast(query, keep_log=True, plain=stage1_input)

        pred_rows.append(quantiles)  # (n, n_quantiles, pred_len)
        true_rows.append(futures)
        meta_rows.append(pd.DataFrame({
            "channel": name,
            "window": np.arange(len(query), dtype=np.int64),
            "start": starts,
        }))
        print(f"  [{channel + 1}/{data.n_channels}] {name}: {len(query)} windows"
              + (f", db={retriever.size}" if retriever else "")
              + (", mode=recent" if recent is not None else ""))

    preds = np.concatenate(pred_rows, axis=0)
    trues = np.concatenate(true_rows, axis=0)
    frames = [pd.concat(meta_rows, ignore_index=True)]
    for i, q in enumerate(model.quantiles):
        frames.append(pd.DataFrame(
            preds[:, i, :], columns=[f"pred_q{q:g}_{t}" for t in range(args.pred_len)]))
    frames.append(pd.DataFrame(
        trues, columns=[f"true_{t}" for t in range(args.pred_len)]))
    table = pd.concat(frames, axis=1)

    os.makedirs(args.output_dir, exist_ok=True)
    stem = build_run_name(args)
    path = os.path.join(args.output_dir, f"{stem}.parquet")
    table.to_parquet(path, compression="zstd", index=False)
    print(f"saved {path}  ({len(table)} windows x {len(model.quantiles)} quantiles "
          f"x {args.pred_len} steps)")

    # the file stem only encodes a few settings; keep the full run condition
    args_path = os.path.join(args.output_dir, f"{stem}_args.json")
    with open(args_path, "w") as f:
        json.dump({**vars(args), "eval_stride_used": eval_stride,
                   "separator_len_used": sep_len,
                   "channels_used": data.channels}, f, indent=2)
    print(f"saved {args_path}")

    if log_rows:
        log_path = os.path.join(args.output_dir, f"{stem}_retrieval.parquet")
        pd.concat(log_rows, ignore_index=True).to_parquet(
            log_path, compression="zstd", index=False)
        print(f"saved {log_path}")


if __name__ == "__main__":
    main()
