"""Step 6a: how much of each method's squared error sits in a handful of series-origins (RMSE fragility).

    ./venv/bin/python concentration.py   (after analyze.py)  -> results/concentration.csv

For every method: the 15,000 series-origins (3,000 series x 5 origins) are ranked by their sum of squared errors over the 28 days; the table gives
the share of the total squared error held by the worst 20 (0.13% of them) and the RMSE before / after dropping each method's own worst 20.
Also results/worst_case.csv (each method's single worst series-origin, its largest forecast and the series' largest sale before the origin) and
results/without_worst_case.csv (pooled RMSE and RMSE by forecast days, with and without the boosted model's single worst series-origin, removed for all methods).
"""
import csv
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ORIGINS = [1801, 1829, 1857, 1885, 1913]
H = 28
BANDS = [(1, 7), (8, 14), (15, 21), (22, 28)]
METHODS = ["naive", "seasonal_naive_7", "moving_avg_28", "zero", "hgb", "hgb_gap28"]


def main():
    w = pd.read_csv(HERE / "data" / "sales_train_evaluation.csv").set_index("id")
    p = pd.read_csv(HERE / "results" / "predictions.csv.gz")
    ids = sorted(p["id"].unique())
    Y = w.loc[ids, [f"d_{i}" for i in range(1, 1942)]].to_numpy(dtype=np.float64)
    sse = {m: [] for m in METHODS}
    band = {m: {b: [] for b in BANDS} for m in METHODS}
    fmax = {m: [] for m in METHODS}
    hmax = []
    mb, ma = [], []
    fmean = {m: [] for m in METHODS}
    keys = []
    for d in ORIGINS:
        act = Y[:, d:d + H]
        hist = Y[:, :d]
        q = p[p["origin"] == d].sort_values(["id", "h"])
        pr = {"naive": np.repeat(hist[:, -1:], H, axis=1), "moving_avg_28": np.repeat(hist[:, -28:].mean(axis=1, keepdims=True), H, axis=1), "zero": np.zeros_like(act),
              "seasonal_naive_7": np.stack([hist[:, d + h - 1 - 7 * int(np.ceil(h / 7))] for h in range(1, H + 1)], axis=1),
              "hgb": q["hgb"].to_numpy().reshape(len(ids), H), "hgb_gap28": q["hgb_gap28"].to_numpy().reshape(len(ids), H)}
        keys += [(d, i) for i in ids]
        hmax.append(hist.max(axis=1))
        mb.append(hist[:, -28:].mean(axis=1))
        ma.append(act.mean(axis=1))
        for m in METHODS:
            e2 = (pr[m] - act) ** 2
            sse[m].append(e2.sum(axis=1))
            fmax[m].append(pr[m].max(axis=1))
            fmean[m].append(pr[m].mean(axis=1))
            for b in BANDS:
                band[m][b].append(e2[:, b[0] - 1:b[1]].sum(axis=1))
    hmax, mb, ma = np.concatenate(hmax), np.concatenate(mb), np.concatenate(ma)
    with open(HERE / "results" / "concentration.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["method", "series_origins", "share_sse_worst20", "rmse_all", "rmse_without_worst20"])
        for m in METHODS:
            s = np.sort(np.concatenate(sse[m]))
            n = len(s)
            wr.writerow([m, n, f"{s[-20:].sum() / s.sum():.10g}", f"{np.sqrt(s.sum() / (n * H)):.10g}", f"{np.sqrt(s[:-20].sum() / ((n - 20) * H)):.10g}"])
    with open(HERE / "results" / "worst_case.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["method", "id", "origin", "share_of_method_sse", "max_forecast", "max_sale_before_origin", "mean_sales_28d_before", "mean_actual", "mean_forecast"])
        for m in METHODS:
            s = np.concatenate(sse[m]); k = int(np.argmax(s))
            wr.writerow([m, keys[k][1], keys[k][0], f"{s[k] / s.sum():.10g}", f"{np.concatenate(fmax[m])[k]:.10g}", f"{hmax[k]:.10g}", f"{mb[k]:.10g}", f"{ma[k]:.10g}", f"{np.concatenate(fmean[m])[k]:.10g}"])
    kh = int(np.argmax(np.concatenate(sse["hgb"])))                  # the boosted model's worst series-origin, removed for every method below
    with open(HERE / "results" / "without_worst_case.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["scope", "method", "rmse_all", "rmse_without_case"])
        for scope in ["pooled"] + [f"{a}-{b}" for a, b in BANDS]:
            for m in METHODS:
                arrs = [np.concatenate(band[m][b]) for b in BANDS] if scope == "pooled" else [np.concatenate(band[m][tuple(int(x) for x in scope.split("-"))])]
                days = 28 if scope == "pooled" else int(scope.split("-")[1]) - int(scope.split("-")[0]) + 1
                tot = sum(a.sum() for a in arrs); n = len(arrs[0])
                rest = sum(a.sum() - a[kh] for a in arrs)
                wr.writerow([scope, m, f"{np.sqrt(tot / (n * days)):.10g}", f"{np.sqrt(rest / ((n - 1) * days)):.10g}"])
    print(open(HERE / "results" / "worst_case.csv").read()); print(open(HERE / "results" / "without_worst_case.csv").read())


if __name__ == "__main__":
    main()
