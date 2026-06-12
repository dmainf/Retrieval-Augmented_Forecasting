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
print(result.to_string(index=False))
