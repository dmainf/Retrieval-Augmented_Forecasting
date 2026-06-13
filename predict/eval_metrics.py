import pandas as pd
import numpy as np
import glob
import os
import re

pred_dir = os.path.dirname(os.path.abspath(__file__))

rows = []
for path in sorted(glob.glob(os.path.join(pred_dir, "*_preds.csv"))):
    fname = os.path.basename(path)
    model = re.sub(r"_preds\.csv$", "", fname)
    df = pd.read_csv(path)
    mae = (df["pred"] - df["true"]).abs().mean()
    mse = ((df["pred"] - df["true"]) ** 2).mean()
    rows.append({"Model": model, "MAE": round(mae, 6), "MSE": round(mse, 6)})

result = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)

BOLD = "\033[1m"
RESET = "\033[0m"

columns = ["Model", "MAE", "MSE"]
best = {c: result[c].min() for c in ["MAE", "MSE"]}

table = []
for _, row in result.iterrows():
    cells = []
    for c in columns:
        if c == "Model":
            cells.append((str(row[c]), False))
        else:
            cells.append((f"{row[c]:.6f}", row[c] == best[c]))
    table.append(cells)

widths = [
    max(len(columns[i]), max(len(table[r][i][0]) for r in range(len(table))))
    for i in range(len(columns))
]

header = "  ".join(columns[i].rjust(widths[i]) for i in range(len(columns)))
print(header)
for cells in table:
    parts = []
    for i, (text, is_best) in enumerate(cells):
        padded = text.rjust(widths[i])
        if is_best:
            padded = padded.replace(text, f"{BOLD}{text}{RESET}")
        parts.append(padded)
    print("  ".join(parts))
