import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_csv(path):
    df = pd.read_csv(path)
    assert {"window", "t", "pred", "true"}.issubset(df.columns)
    if "channel" not in df.columns:
        df["channel"] = 0
    return df.sort_values(["channel", "window", "t"]).reset_index(drop=True)


def tile_series(df):
    """Stitch non-overlapping windows (step = pred_len) into one continuous series."""
    pred_len = int(df["t"].max()) + 1
    n_windows = int(df["window"].max()) + 1
    keep = set(range(0, n_windows, pred_len))
    sel = df[df["window"].isin(keep)]
    return sel["pred"].to_numpy(), sel["true"].to_numpy()


def plot_channel(dfs: dict, ch, save_dir, seg):
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    series = {label: tile_series(df[df["channel"] == ch]) for label, df in dfs.items()}
    true = next(iter(series.values()))[1]
    n = len(true)
    seg = seg or n
    n_seg = (n + seg - 1) // seg

    for s in range(n_seg):
        a, b = s * seg, min((s + 1) * seg, n)
        x = np.arange(a, b)
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(x, true[a:b], color="black", linewidth=1.2, label="true")
        for i, (label, (pred, _)) in enumerate(series.items()):
            ax.plot(x, pred[a:b], color=colors[i % len(colors)],
                    linewidth=0.9, linestyle="--", label=label)
        ax.set_title(f"channel {ch} — continuous test series  [{a}:{b}]")
        ax.set_xlabel("t")
        ax.set_ylabel("value")
        ax.legend(fontsize=8)
        plt.tight_layout()

        out = Path(save_dir) / f"ch{ch:02d}_seg{s:03d}.png"
        plt.savefig(out, dpi=150)
        plt.close(fig)
    return n_seg


def plot_series(dfs: dict, save_dir, seg):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    channels = sorted(next(iter(dfs.values()))["channel"].unique())
    total = 0
    for n, ch in enumerate(channels, 1):
        total += plot_channel(dfs, ch, save_dir, seg)
        print(f"\r[{n}/{len(channels)}] channels", end="", flush=True)
    print(f"\rSaved {total} figures ({len(channels)} channels) to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot pred vs true as a continuous test series, per channel")
    parser.add_argument("csvs", nargs="*",
                        help="CSV file(s) to plot (default: all *_preds.csv in cached/)")
    parser.add_argument("--seg", type=int, default=0,
                        help="points per figure (0 = whole channel in one figure)")
    parser.add_argument("--save", type=str,
                        default=str(Path(__file__).resolve().parent / "plots"),
                        help="Directory to save figures (default: plots/ next to this script)")
    args = parser.parse_args()

    csvs = args.csvs
    if not csvs:
        pred_dir = Path(__file__).resolve().parent / "cached"
        csvs = sorted(str(p) for p in pred_dir.glob("*_preds.csv"))
        if not csvs:
            parser.error(f"no *_preds.csv found in {pred_dir}")

    dfs = {}
    for path in csvs:
        p = Path(path)
        df = load_csv(p)
        label = p.stem.replace("_preds", "")
        dfs[label] = df

    plot_series(dfs, save_dir=args.save, seg=args.seg)


if __name__ == "__main__":
    main()
