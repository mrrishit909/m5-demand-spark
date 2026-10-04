"""Step 6a: how much of each method's squared error sits in a handful of series-origins (RMSE fragility).

    ./venv/bin/python concentration.py   (after analyze.py)  -> results/concentration.csv

For every method: the 15,000 series-origins (3,000 series x 5 origins) are ranked by their sum of squared errors over the 28 days; the table gives
the share of the total squared error held by the worst 20 (0.13% of them) and the RMSE before / after dropping each method's own worst 20.
"""
import csv
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ORIGINS = [1801, 1829, 1857, 1885, 1913]
H = 28
METHODS = ["naive", "seasonal_naive_7", "moving_avg_28", "zero", "hgb", "hgb_gap28"]


def main():
    w = pd.read_csv(HERE / "data" / "sales_train_evaluation.csv").set_index("id")
    p = pd.read_csv(HERE / "results" / "predictions.csv.gz")
    ids = sorted(p["id"].unique())
    Y = w.loc[ids, [f"d_{i}" for i in range(1, 1942)]].to_numpy(dtype=np.float64)
    sse = {m: [] for m in METHODS}
    for d in ORIGINS:
        act = Y[:, d:d + H]
        hist = Y[:, :d]
        q = p[p["origin"] == d].sort_values(["id", "h"])
        pr = {"naive": np.repeat(hist[:, -1:], H, axis=1), "moving_avg_28": np.repeat(hist[:, -28:].mean(axis=1, keepdims=True), H, axis=1), "zero": np.zeros_like(act),
              "seasonal_naive_7": np.stack([hist[:, d + h - 1 - 7 * int(np.ceil(h / 7))] for h in range(1, H + 1)], axis=1),
              "hgb": q["hgb"].to_numpy().reshape(len(ids), H), "hgb_gap28": q["hgb_gap28"].to_numpy().reshape(len(ids), H)}
        for m in METHODS:
            sse[m].append(((pr[m] - act) ** 2).sum(axis=1))
    with open(HERE / "results" / "concentration.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["method", "series_origins", "share_sse_worst20", "rmse_all", "rmse_without_worst20"])
        for m in METHODS:
            s = np.sort(np.concatenate(sse[m]))
            n = len(s)
            wr.writerow([m, n, f"{s[-20:].sum() / s.sum():.10g}", f"{np.sqrt(s.sum() / (n * H)):.10g}", f"{np.sqrt(s[:-20].sum() / ((n - 20) * H)):.10g}"])
    print(open(HERE / "results" / "concentration.csv").read())


if __name__ == "__main__":
    main()
