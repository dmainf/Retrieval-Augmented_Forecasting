import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REQUIRED = {"window", "t", "pred", "true"}


def load_csv(path):
    df = pd.read_csv(path)
    assert REQUIRED.issubset(df.columns)
    return df.sort_values(["window", "t"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Compare per-step error distributions across methods")
    parser.add_argument("csvs", nargs="*",
                        help="*_preds.csv files (default: all *_preds.csv next to this script)")
    parser.add_argument("--save", type=str, default="error_dist.png", help="PNG path")
    parser.add_argument("--baseline", type=str, default=None,
                        help="label substring used as reference for the decile panel (default: 'none' if present)")
    parser.add_argument("--clip", type=float, default=3.0, help="x-range for the signed-error histogram")
    parser.add_argument("--bins", type=int, default=120)
    args = parser.parse_args()

    csvs = args.csvs
    if not csvs:
        pred_dir = Path(__file__).resolve().parent.parent
        csvs = sorted(str(p) for p in pred_dir.glob("*_preds.csv"))
        if not csvs:
            parser.error(f"no *_preds.csv found in {pred_dir}")

    preds, true_ref = {}, None
    for path in csvs:
        p = Path(path)
        df = load_csv(p)
        label = p.stem.replace("_preds", "")
        if true_ref is None:
            true_ref = df["true"].values
        else:
            assert len(df) == len(true_ref), f"{label}: row count differs from the first file"
            assert np.allclose(df["true"].values, true_ref, atol=1e-6), \
                f"{label}: targets differ — these are not the same evaluation set"
        preds[label] = df["pred"].values

    err = {m: preds[m] - true_ref for m in preds}
    aerr = {m: np.abs(e) for m, e in err.items()}

    print(f"{'method':<22}{'MAE':>9}{'MSE':>9}{'RMSE':>9}{'p95|e|':>9}{'p99|e|':>9}{'max|e|':>9}")
    for m in preds:
        a = aerr[m]
        print(f"{m:<22}{a.mean():>9.4f}{(a ** 2).mean():>9.4f}{np.sqrt((a ** 2).mean()):>9.4f}"
              f"{np.percentile(a, 95):>9.4f}{np.percentile(a, 99):>9.4f}{a.max():>9.4f}")

    base = None
    if args.baseline:
        base = next((m for m in preds if args.baseline in m), None)
    if base is None:
        base = next((m for m in preds if "none" in m), list(preds)[0])

    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    rng = (-args.clip, args.clip)
    for i, m in enumerate(preds):
        ax.hist(np.clip(err[m], *rng), bins=args.bins, range=rng, density=True,
                histtype="step", linewidth=1.5, color=colors[i], label=m)
    ax.axvline(0, color="k", lw=0.8, ls=":")
    ax.set_title("Signed error (pred − true) — density")
    ax.set_xlabel("error")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)

    amax = min(args.clip, max(np.percentile(aerr[m], 99.9) for m in preds))
    ax = axes[0, 1]
    for i, m in enumerate(preds):
        ax.hist(aerr[m], bins=args.bins, range=(0, amax), histtype="step",
                linewidth=1.5, color=colors[i], label=m)
    ax.set_yscale("log")
    ax.set_title("|error| histogram (log count — tail visible)")
    ax.set_xlabel("|error|")
    ax.set_ylabel("count (log)")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    xs = np.linspace(0, amax, 300)
    for i, m in enumerate(preds):
        a = np.sort(aerr[m])
        surv = 1.0 - np.searchsorted(a, xs, side="right") / len(a)
        ax.plot(xs, surv, color=colors[i], lw=1.5, label=m)
    ax.set_yscale("log")
    ax.set_title("Tail: fraction of steps with |error| > x")
    ax.set_xlabel("x")
    ax.set_ylabel("P(|error| > x)")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    edges = np.percentile(aerr[base], np.arange(0, 101, 10))
    binid = np.clip(np.digitize(aerr[base], edges[1:-1]), 0, 9)
    base_mse = np.array([(err[base][binid == d] ** 2).mean() for d in range(10)])
    others = [m for m in preds if m != base]
    width = 0.8 / max(1, len(others))
    x = np.arange(10)
    for i, m in enumerate(others):
        se = err[m] ** 2
        diff = np.array([se[binid == d].mean() for d in range(10)]) - base_mse
        ci = list(preds).index(m)
        tot = diff.mean()
        ax.bar(x + i * width, diff, width=width, color=colors[ci],
               label=f"{m}  (Σ→{tot:+.4f} total MSE)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(f"MSE − '{base}' within each |error| decile  (<0 = better)")
    ax.set_xlabel("decile by |error| (0 = easiest steps, 9 = hardest)")
    ax.set_ylabel("MSE difference vs baseline")
    ax.set_xticks(x + (len(others) - 1) * width / 2)
    ax.set_xticklabels(range(10))
    ax.legend(fontsize=8)

    fig.suptitle("Per-step error distribution comparison", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
