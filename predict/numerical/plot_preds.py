import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_csv(path):
    df = pd.read_csv(path)
    assert {"window", "t", "pred", "true"}.issubset(df.columns)
    return df


def plot_windows(dfs: dict, windows, save_dir=None):
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

    for win in windows:
        fig, ax = plt.subplots(figsize=(8, 4))
        first = True
        for i, (label, df) in enumerate(dfs.items()):
            w = df[df["window"] == win].sort_values("t")
            if w.empty:
                continue
            if first:
                ax.plot(w["t"], w["true"], color="black", linewidth=1.5, label="true")
                first = False
            ax.plot(w["t"], w["pred"], color=colors[i % len(colors)],
                    linewidth=1, linestyle="--", label=label)
        ax.set_title(f"window {win}")
        ax.set_xlabel("t")
        ax.set_ylabel("value")
        ax.legend(fontsize=8)
        plt.tight_layout()

        if save_dir:
            out = save_path / f"window_{win:04d}.png"
            plt.savefig(out, dpi=150)
            print(f"Saved to {out}")
            plt.close(fig)
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot pred vs true per window")
    parser.add_argument("csvs", nargs="+", help="CSV file(s) to plot")
    parser.add_argument("--windows", nargs="*", type=int, default=None,
                        help="Window indices to plot (default: first 12)")
    parser.add_argument("--save", type=str, default=None,
                        help="Directory to save figure (default: show interactively)")
    args = parser.parse_args()

    dfs = {}
    for path in args.csvs:
        p = Path(path)
        df = load_csv(p)
        label = p.stem.replace("_preds", "")
        dfs[label] = df

    all_windows = sorted(next(iter(dfs.values()))["window"].unique())
    windows = args.windows if args.windows is not None else all_windows

    plot_windows(dfs, windows, save_dir=args.save)


if __name__ == "__main__":
    main()
