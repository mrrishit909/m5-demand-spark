"""Steps 4-6: time-aware evaluation of four benchmarks and a gradient-boosted model on the Spark-built features.

    ./venv/bin/python analyze.py   (after features.py)  -> results/*.csv, results/predictions.csv.gz

Protocol. Five forecast origins d = 1801, 1829, 1857, 1885, 1913 (28 days apart; the last is the M5 'validation' window d_1914..d_1941,
whose labels are public in sales_train_evaluation.csv, so it is NOT the hidden test of the competition). At each origin every method sees only
days <= d and forecasts days d+1..d+28. No random splits anywhere; no hyper-parameter was tuned (one fixed setting, below).
Methods: naive (last day repeated), seasonal naive 7 (last week repeated), moving average (mean of the last 28 days), zero (always 0; a reference that
shows how MAE rewards the median on intermittent series), and HistGradientBoosting (sklearn, Poisson loss, 300 iterations, learning rate 0.1,
63 leaves, min 200 rows per leaf) trained per origin on the target days d-729..d only, using the features from features.py.
Scores: RMSE and MAE pooled over all series-days; bottom-level WRMSSE-style = sum_i w_i * RMSSE_i / sum_i w_i with RMSSE_i = sqrt(mean_h (y-yhat)^2 / s_i),
s_i = mean of squared one-day differences from the series' first sale to d, w_i = revenue (units x shelf price) of the 28 days before d; series with s_i = 0 are dropped.
This is the M5 metric at its lowest of 12 aggregation levels only, not the official WRMSSE. Intermittency = share of zero-sales days over the 364 days before d
(or since launch). Feature importance = permutation importance (RMSE increase when one column is shuffled) on the validation windows.
"""
import csv
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).parent
RES = HERE / "results"
ORIGINS = [1801, 1829, 1857, 1885, 1913]
H = 28
TRAIN_DAYS = 730
METHODS = ["naive", "seasonal_naive_7", "moving_avg_28", "zero", "hgb", "hgb_gap28"]
BINS = [(0.0, 0.1, "0-10%"), (0.1, 0.3, "10-30%"), (0.3, 0.5, "30-50%"), (0.5, 0.7, "50-70%"), (0.7, 1.0001, "70-100%")]
CATS = ["store_id", "state_id", "dept_id", "cat_id", "event_type_1"]
CAL = ["wday", "month", "dom", "has_event", "snap", "price", "price_chg_w", "price_rel91"]
# hgb: origin-anchored features + the horizon h; hgb_gap28: only features anchored 28 days before the target (no h), the first, stale design
FEATS = CATS + ["h"] + CAL + ["price_vs_o", "y_o", "r_m7", "r_m28", "r_m91", "r_std28", "r_share28", "r_exp", "days_since_sale", "lag_w1", "lag_w2", "lag_w3"]
FEATS_G = CATS + CAL + ["g_lag28", "g_lag35", "g_lag42", "g_dow4", "g_m7", "g_m28", "g_m91", "g_std28", "g_share28", "g_exp"]


def hgb():
    return HistGradientBoostingRegressor(loss="poisson", learning_rate=0.1, max_iter=300, max_leaf_nodes=63, min_samples_leaf=200,
                                         early_stopping=False, random_state=0)


def main():
    f = pd.read_parquet(HERE / "data" / "features.parquet").sort_values(["origin", "id", "t"], ignore_index=True)
    bs = pd.read_parquet(HERE / "data" / "base.parquet", columns=["id", "dnum", "y", "price"]).sort_values(["id", "dnum"], ignore_index=True)
    ids = sorted(bs["id"].unique())
    n = len(ids)
    assert n == 3000
    YF = bs["y"].to_numpy(dtype=np.float64).reshape(n, 1941)
    PF = np.nan_to_num(bs["price"].to_numpy(dtype=np.float64)).reshape(n, 1941)      # no price that week = not on sale = 0 in the revenue weight
    assert (bs["dnum"].to_numpy().reshape(n, 1941)[0] == np.arange(1, 1942)).all()
    del bs
    for c in CATS:                                                  # fixed integer codes (<=10 stores, 3 states, 7 depts, 3 cats, 5 event types; -1 = no event)
        f[c] = f[c].astype("category").cat.codes.astype(np.int16)
    for c in set(FEATS + FEATS_G):
        if c not in CATS:
            f[c] = f[c].astype(np.float32)
    f["y"] = f["y"].astype(np.float32)
    trn = f[f["origin"] == 0].reset_index(drop=True)
    preds = {m: {} for m in METHODS}
    rows_pred = []
    info = []
    perm_rows = []
    for d in ORIGINS:
        act = YF[:, d:d + H]                                        # days d+1..d+28
        hist = YF[:, :d]
        pn = np.repeat(hist[:, -1:], H, axis=1)
        ps = np.stack([hist[:, d + h - 1 - 7 * int(np.ceil(h / 7))] for h in range(1, H + 1)], axis=1)
        pm = np.repeat(hist[:, -28:].mean(axis=1, keepdims=True), H, axis=1)
        pz = np.zeros_like(act)
        tr = trn[(trn["t"] >= d - TRAIN_DAYS + 1) & (trn["t"] <= d)]
        va = f[f["origin"] == d]
        assert len(va) == n * H and (va["t"].to_numpy().reshape(n, H)[0] == np.arange(d + 1, d + H + 1)).all() and (va["id"].to_numpy().reshape(n, H)[:, 0] == np.array(ids)).all()
        assert np.allclose(va["y"].to_numpy().reshape(n, H), act)    # features file and raw sales agree on the labels
        models = {}
        out = {}
        for name, cols in (("hgb", FEATS), ("hgb_gap28", FEATS_G)):
            m = hgb().set_params(categorical_features=[cols.index(c) for c in CATS])
            m.fit(tr[cols], tr["y"])
            models[name] = (m, cols)
            out[name] = np.char.mod("%.4f", m.predict(va[cols]).reshape(n, H))      # stored with 4 decimals; the scores use the stored values
        info.append((d, len(tr), models["hgb"][0].n_iter_))
        P_ = {"naive": pn, "seasonal_naive_7": ps, "moving_avg_28": pm, "zero": pz, "hgb": out["hgb"].astype(np.float64), "hgb_gap28": out["hgb_gap28"].astype(np.float64)}
        for k, v in P_.items():
            preds[k][d] = v
        for i, idv in enumerate(ids):
            for h in range(H):
                rows_pred.append((d, idv, h + 1, out["hgb"][i, h], out["hgb_gap28"][i, h]))
        m, cols = models["hgb"]                                     # permutation importance of the main model (shuffle one column of the validation rows)
        Xv = va[cols]
        base_rmse = float(np.sqrt(np.mean((P_["hgb"] - act) ** 2)))
        rng = np.random.default_rng(d)
        for c in cols:
            inc = []
            for r in range(2):
                Xp = Xv.copy()
                Xp[c] = rng.permutation(Xp[c].to_numpy())
                inc.append(float(np.sqrt(np.mean((m.predict(Xp).reshape(n, H) - act) ** 2))) - base_rmse)
            perm_rows.append((d, c, float(np.mean(inc))))
        print("origin", d, "done", info[-1], flush=True)

    # ---- scores
    def scale_weight(d):
        yy = YF[:, :d]
        first = np.where(yy.sum(axis=1) > 0, (yy > 0).argmax(axis=1), d)        # 0-based index of first sale (d if none yet)
        s = np.zeros(n)
        for i in range(n):
            seg = yy[i, first[i]:]
            s[i] = np.mean(np.diff(seg) ** 2) if len(seg) > 1 else 0.0
        w = (YF[:, d - 28:d] * PF[:, d - 28:d]).sum(axis=1)                      # revenue of days d-27..d
        return first, s, w

    out_o, bins_acc, hor_acc = [], [], []
    for d in ORIGINS:
        act = YF[:, d:d + H]
        first, s, w = scale_weight(d)
        valid = s > 0
        zs = np.full(n, np.nan)                                     # zero-day share over the last 364 days (from launch); nan = no history yet
        for i in range(n):
            seg = YF[i, max(first[i], d - 364):d]
            if len(seg):
                zs[i] = (seg == 0).mean()
        for k in METHODS:
            e = preds[k][d] - act
            rmse, mae = float(np.sqrt(np.mean(e ** 2))), float(np.mean(np.abs(e)))
            rmsse = np.sqrt(np.mean(e ** 2, axis=1)[valid] / s[valid])
            wr = float((w[valid] * rmsse).sum() / w[valid].sum())
            out_o.append((d, k, rmse, mae, wr, int(valid.sum())))
            for lo, hi, lab in BINS:
                mk = (zs >= lo) & (zs < hi)
                bins_acc.append((d, lab, k, int(mk.sum()), float((e[mk] ** 2).sum()), float(np.abs(e[mk]).sum()), int(mk.sum()) * H))
            for a, b in ((1, 7), (8, 14), (15, 21), (22, 28)):
                hor_acc.append((d, f"{a}-{b}", k, float((e[:, a - 1:b] ** 2).sum()), n * (b - a + 1)))
    so = pd.DataFrame(out_o, columns=["origin", "method", "rmse", "mae", "wrmsse", "series_scored"])
    so.to_csv(RES / "scores_by_origin.csv", index=False, float_format="%.10g")
    ov = so.groupby("method")[["rmse", "mae", "wrmsse"]].mean().reindex(METHODS).reset_index()
    ov.to_csv(RES / "scores_overall.csv", index=False, float_format="%.10g")
    b = pd.DataFrame(bins_acc, columns=["origin", "bin", "method", "series", "sse", "sae", "points"])
    g = b.groupby(["bin", "method"])[["series", "sse", "sae", "points"]].sum().reset_index()
    g["rmse"] = np.sqrt(g["sse"] / g["points"])
    g["mae"] = g["sae"] / g["points"]
    g["o1"] = g["bin"].map({lab: i for i, (_, _, lab) in enumerate(BINS)})
    g["o2"] = g["method"].map({k: i for i, k in enumerate(METHODS)})
    g.sort_values(["o1", "o2"])[["bin", "method", "series", "points", "rmse", "mae"]].to_csv(RES / "by_intermittency.csv", index=False, float_format="%.10g")
    hh = pd.DataFrame(hor_acc, columns=["origin", "horizon", "method", "sse", "points"]).groupby(["horizon", "method"])[["sse", "points"]].sum().reset_index()
    hh["rmse"] = np.sqrt(hh["sse"] / hh["points"])
    hh["o1"] = hh["horizon"].map({"1-7": 0, "8-14": 1, "15-21": 2, "22-28": 3})
    hh["o2"] = hh["method"].map({k: i for i, k in enumerate(METHODS)})
    hh.sort_values(["o1", "o2"])[["horizon", "method", "points", "rmse"]].to_csv(RES / "by_horizon.csv", index=False, float_format="%.10g")
    pr = pd.DataFrame(perm_rows, columns=["origin", "feature", "rmse_increase"])
    pr.groupby("feature")["rmse_increase"].mean().sort_values(ascending=False).reset_index().to_csv(RES / "importance.csv", index=False, float_format="%.10g")
    pd.DataFrame(info, columns=["origin", "train_rows", "iterations"]).to_csv(RES / "train_info.csv", index=False)
    with gzip.open(RES / "predictions.csv.gz", "wt", newline="", compresslevel=9) as fh:
        w_ = csv.writer(fh)
        w_.writerow(["origin", "id", "h", "hgb", "hgb_gap28"])
        w_.writerows(rows_pred)
    print(ov)


if __name__ == "__main__":
    main()
