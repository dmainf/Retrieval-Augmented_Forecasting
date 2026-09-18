"""Command line arguments. Every hyperparameter has a default and can be overridden."""
import argparse

from dataset import SPLITS
from retriever import METRICS, ENCODERS, POOLS, SELECTORS  # imports faiss, before torch
from raf_model import CONCAT_ORDERS, INJECTIONS, SEPARATORS


def parse_args():
    p = argparse.ArgumentParser(
        description="RAF: retrieve similar windows, concatenate them in front of "
                    "the query context, and forecast with a frozen Chronos-Bolt."
    )

    # data
    p.add_argument("--root-path", default="./Datasets/_parquet",
                   help="directory holding <dataset>.parquet")
    p.add_argument("--dataset", default="ETTh1", help="dataset name")
    p.add_argument("--channels", default="",
                   help="comma-separated column names; empty = every column")
    p.add_argument("--eval-split", default="test", choices=list(SPLITS),
                   help="split the query windows are taken from")
    p.add_argument("--eval-stride", default=0, type=int,
                   help="stride over query windows; 0 = window length (non-overlapping)")

    # window geometry
    p.add_argument("--seq-len", default=96, type=int, help="query context length")
    p.add_argument("--pred-len", default=64, type=int,
                   help="forecast horizon (<= backbone prediction_length)")

    # retrieval
    p.add_argument("--top-k", default=1, type=int,
                   help="number of retrieved windows to concatenate; 0 = no retrieval (baseline)")
    p.add_argument("--retrieval-split", default="train", choices=list(SPLITS) + ["past"],
                   help="split the retrieval database is built from. past = every "
                        "window that ends before the query context begins, so the "
                        "database grows with time and the query's own recent history "
                        "is a candidate")
    p.add_argument("--retrieval-stride", default=1, type=int,
                   help="stride when cutting the retrieval database windows")
    p.add_argument("--retrieval-metric", default="euclidean", choices=METRICS,
                   help="distance between search keys; cosine and correlation "
                        "normalize by definition, euclidean does not")

    # retrieval axis 1: encode -- what a window is turned into before comparing
    p.add_argument("--encoder", default="raw", choices=ENCODERS,
                   help="raw = the context values. zscore = shape only. spectral = "
                        "|rFFT|. learned = a trained embedding (--encoder-path). "
                        "predicted = two-stage; forecast the query future with no "
                        "retrieval, then search the database futures with it. "
                        "future = an ORACLE keyed on the TRUE future; not a usable "
                        "method, it bounds what any retriever could reach")
    p.add_argument("--encoder-path", default="",
                   help="checkpoint for --encoder learned, written by encoder.py")
    p.add_argument("--retrieval-mode", default="search", choices=["search", "recent"],
                   help="search = retrieve by similarity. recent = skip retrieval and "
                        "take the K blocks immediately preceding the query. With "
                        "--separator none the latter reassembles the contiguous "
                        "history exactly, so it isolates what the container costs "
                        "when the content is already ideal")
    p.add_argument("--query-context", default=0, type=int,
                   help="history length fed to the backbone as the query context; "
                        "0 = seq_len. Retrieval still keys off seq_len, so this only "
                        "changes how much real history the model reads. Retrieved "
                        "windows and history compete for the same 2048-point budget: "
                        "each window costs win_len + separator_len points")
    p.add_argument("--stage1-context", default=0, type=int,
                   help="history length for the retrieval-free forecast; 0 = seq_len. "
                        "With --top-k 0 that forecast is the output, so this gives the "
                        "long-context baseline. With --encoder predicted it is the "
                        "forecast that becomes the retrieval key, and the RAF input "
                        "itself still uses only seq_len")
    p.add_argument("--oracle-mix", default=0.0, type=float,
                   help="DIAGNOSTIC for --encoder predicted: blend the true future "
                        "into the search key by this fraction. 0 = the honest "
                        "two-stage method, 1 = the oracle. Anything in between is "
                        "not a usable method; it traces how final accuracy depends "
                        "on the accuracy of the key")
    p.add_argument("--retrieval-rounds", default=1, type=int,
                   help="for --encoder predicted: how many times to re-search with "
                        "the previous round's forecast as the query. 1 = plain "
                        "two-stage; higher iterates. Ignored by other encoders")
    p.add_argument("--key-dim", default=0, type=int,
                   help="PCA the keys to this many dimensions; 0 = no projection. "
                        "Must be below --top-k for the eopt selector to mean anything")

    # retrieval axis 2: pool -- which windows are allowed to be candidates
    p.add_argument("--pool", default="all", choices=POOLS,
                   help="all = the whole database. topm = the nearest --pool-size. "
                        "cluster = only the query's own k-means cluster")
    p.add_argument("--pool-size", default=100, type=int,
                   help="candidates handed to the selector when --pool is not all")
    p.add_argument("--n-clusters", default=8, type=int,
                   help="k-means clusters for --pool cluster")

    # retrieval axis 3: select -- which K candidates end up in the context
    p.add_argument("--selector", default="topk", choices=SELECTORS,
                   help="topk = the K nearest. mmr = relevance vs diversity (SARAF). "
                        "dpp = greedy log-det, i.e. D-optimal. eopt = greedy "
                        "lambda_min, i.e. E-optimal")
    p.add_argument("--ca-shift", default=0.0, type=float,
                   help="ridge strength for the constructed-analogue correction; "
                        "0 disables. Solves weights that reconstruct the query "
                        "context from the retrieved contexts, applies them to the "
                        "futures, and shifts the injected futures so their mean "
                        "matches that estimate. Corrects the first-order averaging "
                        "bias while leaving the spread alone")
    p.add_argument("--ca-pool", default=200, type=int,
                   help="candidates used for the constructed-analogue solve")
    p.add_argument("--align", default="none",
                   choices=["none", "last", "mean", "mean-scale"],
                   help="shift each retrieved future onto the query's current level "
                        "before concatenating. Retrieved windows carry the raw level "
                        "they had months ago, which is why only their mean and spread "
                        "survive; aligning is what makes the waveform comparable. "
                        "last matches the final context value, mean the context mean, "
                        "mean-scale also matches the context standard deviation. "
                        "This is the alignment Tire et al. call Retrieval w/ Alignment")
    p.add_argument("--example-part", default="window",
                   choices=["window", "future", "context"],
                   help="which half of each retrieved window to concatenate. "
                        "window = both (the usual RAF layout). future = only the "
                        "continuation, which costs 64 points instead of 160. "
                        "Worth trying because shuffling the futures among the "
                        "retrieved windows costs nothing, i.e. the context-to-future "
                        "pairing is not being used")
    p.add_argument("--example-ablation", default="none",
                   choices=["none", "shuffle-future", "random-future", "drop-future",
                            "shuffle-time", "gauss-future", "gauss-local",
                            "truth-future", "self-truth"],
                   help="damage the retrieved examples to test whether the backbone "
                        "actually reads them. shuffle-future permutes the futures "
                        "among the K retrieved, so the same futures are present but "
                        "paired with the wrong contexts; random-future replaces them "
                        "with unrelated ones; drop-future blanks them out. "
                        "shuffle-time permutes the points *inside* each future, which "
                        "keeps its exact multiset of values -- so loc, scale and the "
                        "whole histogram survive -- and destroys only the waveform. "
                        "gauss-future replaces each future with noise matched to its "
                        "mean and standard deviation. gauss-local goes further and "
                        "takes that mean and standard deviation from the QUERY's own "
                        "context instead of the retrieved windows -- if that matches, "
                        "retrieval is contributing nothing at all. If accuracy does "
                        "not move, the futures are not being used. "
                        "Two run the other way and are diagnostics, not damage: "
                        "truth-future writes the query's true future into each "
                        "retrieved window, keeping the retrieved contexts; "
                        "self-truth replaces the whole example with the query's own "
                        "window, answer included. The latter is a perfect reference "
                        "rather than a perfect search, so it bounds what the "
                        "container can transmit at all -- if even that moves little, "
                        "no retriever is worth building")
    p.add_argument("--vol-match", default=0.0, type=float,
                   help="penalise candidates whose future volatility differs from the "
                        "query's estimated future volatility. Counters the pull toward "
                        "average-looking windows; 0 disables")
    p.add_argument("--hub-penalty", default=0.0, type=float,
                   help="discount windows that sit close to everything (hubs). "
                        "Rewards being close to THIS query rather than to any query; "
                        "0 disables")
    p.add_argument("--oracle-rank", default=0.0, type=float,
                   help="take the K windows starting at this fraction of the "
                        "candidate ranking instead of at the top. With "
                        "--encoder future the ranking is by how close a window's "
                        "future is to the query's TRUE future, so this dials "
                        "retrieval quality continuously: 0 is the oracle, larger "
                        "values retrieve futures that resemble the answer less. "
                        "Diagnostic only -- it needs the answer to order the "
                        "candidates, exactly like the oracle does")
    p.add_argument("--diversity", default=0.5, type=float,
                   help="0 = relevance only, 1 = diversity only; unused by topk")

    # augmentation
    p.add_argument("--inject", default="concat", choices=INJECTIONS,
                   help="how the retrieved examples reach the backbone. concat = "
                        "splice them into the time axis in front of the query, the "
                        "usual RAF layout, which spends context budget on them and "
                        "labels them as recent history. parallel = hand each one to "
                        "chronos-2 as its own covariate row aligned on the same time "
                        "axis, mixed by group attention; the history is kept whole, "
                        "every row is instance-normed on its own, and no separator is "
                        "needed. chronos-2 only")
    p.add_argument("--concat-order", default="nearest-last", choices=CONCAT_ORDERS,
                   help="nearest-last: closest match sits next to the query")
    p.add_argument("--separator", default="nan", choices=SEPARATORS,
                   help="what fills the gap between examples; none = plain concatenation. "
                        "nan is the default: it is the only variant that does not invent "
                        "observations, and its meaning does not shift with top_k")
    p.add_argument("--separator-len", default=0, type=int,
                   help="separator length in points; 0 = one backbone patch (16), "
                        "which keeps example boundaries aligned with patch boundaries")

    # backbone
    p.add_argument("--chronos-model", default="amazon/chronos-2",
                   help="HF id or local path of the backbone. chronos-2 reads the "
                        "injected examples far more strongly than chronos-bolt "
                        "(oracle 0.3397 vs 0.4615 on ETTh1), so the retrieval "
                        "ceiling is much higher there")
    p.add_argument("--batch-size", default=256, type=int)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--gpu", default=0, type=int, help="cuda device index")

    # output
    p.add_argument("--output-dir", default="./results", help="where to write the tables")
    p.add_argument("--run-name", default="",
                   help="output file stem; empty = built from the settings")
    p.add_argument("--save-retrieval", default=1, type=int,
                   help="1 = also write which windows were retrieved")
    p.add_argument("--seed", default=2021, type=int)
    return p.parse_args()
