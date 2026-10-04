"""Step 5: independent recount of every number in results/ in plain Python (csv, gzip, hashlib, math; no pandas/numpy/sklearn/Spark).

    ./venv/bin/python check.py

Streams the raw M5 CSVs: re-derives the 300-item sample from md5(item_id), the full-vs-sample audit over all 30,490 series, the four benchmark
forecasts and their scores, the scale/weights of the WRMSSE-style metric, the intermittency and horizon tables, and re-scores the stored
gradient-boosting predictions (both models), and recounts the worst-case tables (results/predictions.csv.gz) against the raw sales. What it can NOT recompute: the gradient-boosting fit itself and the
permutation importances (library fits, disclosed); for those it checks the stored predictions are complete and non-negative and that the importance
table lists exactly the 26 model features.
"""
import csv
import gzip
import hashlib
import math
from pathlib import Path

HERE = Path(__file__).parent
RES = HERE / "results"
D = HERE / "data"
ORIGINS = [1801, 1829, 1857, 1885, 1913]
H = 28
METHODS = ["naive", "seasonal_naive_7", "moving_avg_28", "zero", "hgb", "hgb_gap28"]
BINS = [(0.0, 0.1, "0-10%"), (0.1, 0.3, "10-30%"), (0.3, 0.5, "30-50%"), (0.5, 0.7, "50-70%"), (0.7, 1.0001, "70-100%")]
HORS = [(1, 7), (8, 14), (15, 21), (22, 28)]
FEATS = ["store_id", "state_id", "dept_id", "cat_id", "event_type_1", "h", "wday", "month", "dom", "has_event", "snap", "price", "price_chg_w", "price_rel91",
         "price_vs_o", "y_o", "r_m7", "r_m28", "r_m91", "r_std28", "r_share28", "r_exp", "days_since_sale", "lag_w1", "lag_w2", "lag_w3"]
count = 0


def same(a, b, what, exact=False):
    global count
    a, b = float(a), float(b)
    ok = a == b if exact else abs(a - b) <= 1e-8 * max(1.0, abs(b))
    assert ok, f"{what}: results {a} vs recomputed {b}"
    count += 1


def read(name):
    with open(RES / name, newline="") as f:
        return list(csv.DictReader(f))


def main():
    global count
    # ---- sample rule: first 300 items by md5(item_id)
    with open(D / "sales_train_evaluation.csv", newline="") as f:
        rd = csv.reader(f)
        head = next(rd)
        assert head[6] == "d_1" and head[-1] == "d_1941" and len(head) == 6 + 1941
        item_list = [(r[1]) for r in rd]
    items = sorted(set(item_list), key=lambda s: hashlib.md5(s.encode()).hexdigest())
    assert len(items) == 3049 and len(item_list) == 30490
    sample = set(items[:300])
    with open(RES / "sample_items.csv") as f:
        assert sorted(sample) == [x.strip() for x in f.read().split("\n")[1:] if x.strip()]
    count += 1
    # ---- full vs sample (zero share and mean units per day from first sale to d_1913)
    acc = {"all": [], "sample": []}
    series = {}
    f = open(D / "sales_train_evaluation.csv", newline="")
    rd = csv.reader(f)
    next(rd)
    for r in rd:
        sid, item, store, y = r[0], r[1], r[4], list(map(int, r[6:]))
        first = next((i for i, v in enumerate(y) if v > 0), None)
        assert first is not None and first < 1913
        seg = y[first:1913]
        z = sum(1 for v in seg if v == 0)
        rec = (len(seg), z / len(seg), sum(seg), z)
        acc["all"].append(rec)
        if item in sample:
            acc["sample"].append(rec)
            series[sid] = (item, store, y)
    assert len(series) == 3000
    for row in read("full_vs_sample.csv"):
        key = "all" if row["group"].startswith("all") else "sample"
        a = acc[key]
        zs = sorted(r[1] for r in a)
        n = len(a)
        same(row["series"], n, "series", True)
        same(row["mean_zero_share"], sum(r[1] for r in a) / n, "mean zero share")
        same(row["median_zero_share"], zs[math.ceil(0.5 * n) - 1], "median zero share")
        same(row["share_series_zero_gt_50"], sum(1 for v in zs if v > 0.5) / n, "share >50%")
        same(row["mean_units_per_day"], sum(r[2] / r[0] for r in a) / n, "mean units per day")
        same(row["total_units"], sum(r[2] for r in a), "total units", True)
        same(row["series_days"], sum(r[0] for r in a), "series days", True)
    f.close()
    # ---- prices and calendar
    wk = {}
    with open(D / "calendar.csv", newline="") as f:
        for r in csv.DictReader(f):
            wk[int(r["d"][2:])] = r["wm_yr_wk"]
    price = {}
    with open(D / "sell_prices.csv", newline="") as f:
        rd = csv.reader(f)
        next(rd)
        for st, it, w, p in rd:
            if it in sample:
                price[(st, it, w)] = float(p)
    ids = sorted(series)
    # ---- stored predictions
    hgb = {}
    with gzip.open(RES / "predictions.csv.gz", "rt", newline="") as f:
        rd = csv.reader(f)
        assert next(rd) == ["origin", "id", "h", "hgb", "hgb_gap28"]
        for o, sid, h, p, q in rd:
            assert float(p) >= 0 and float(q) >= 0
            hgb.setdefault((int(o), sid), [[None] * H, [None] * H])
            hgb[(int(o), sid)][0][int(h) - 1] = float(p)
            hgb[(int(o), sid)][1][int(h) - 1] = float(q)
    assert len(hgb) == 3000 * len(ORIGINS) and all(None not in v[0] and None not in v[1] for v in hgb.values())
    count += 1
    # ---- scores
    by_origin = {}
    bins_acc = {}
    hor_acc = {}
    sso = {k: [] for k in METHODS}
    ssb = {k: {b: [] for b in HORS} for k in METHODS}
    fmx = {k: [] for k in METHODS}
    fmn = {k: [] for k in METHODS}
    skeys, hmx, mbf, maf = [], [], [], []
    for d in ORIGINS:
        wsum, wr = 0.0, {k: 0.0 for k in METHODS}
        sse = {k: 0.0 for k in METHODS}
        sae = {k: 0.0 for k in METHODS}
        nvalid = 0
        for sid in ids:
            item, store, y = series[sid]
            hist, act = y[:d], y[d:d + H]
            pr = {"naive": [hist[-1]] * H, "moving_avg_28": [sum(hist[-28:]) / 28.0] * H, "zero": [0.0] * H, "hgb": hgb[(d, sid)][0], "hgb_gap28": hgb[(d, sid)][1],
                  "seasonal_naive_7": [hist[d + h - 1 - 7 * ((h + 6) // 7)] for h in range(1, H + 1)]}
            first = next((i for i, v in enumerate(hist) if v > 0), d)
            seg = hist[first:]
            s = sum((seg[i] - seg[i - 1]) ** 2 for i in range(1, len(seg))) / (len(seg) - 1) if len(seg) > 1 else 0.0
            w = sum(hist[t - 1] * price.get((store, item, wk[t]), 0.0) for t in range(d - 27, d + 1))
            seg2 = hist[max(first, d - 364):]
            bi = next((lab for lo, hi, lab in BINS if lo <= sum(1 for v in seg2 if v == 0) / len(seg2) < hi), None) if seg2 else None
            skeys.append((d, sid))
            hmx.append(max(hist))
            mbf.append(sum(hist[-28:]) / 28.0)
            maf.append(sum(act) / H)
            for k in METHODS:
                e = [pr[k][i] - act[i] for i in range(H)]
                fmx[k].append(max(pr[k]))
                fmn[k].append(sum(pr[k]) / H)
                for a, bb in HORS:
                    ssb[k][(a, bb)].append(sum(v * v for v in e[a - 1:bb]))
                ss = sum(v * v for v in e)
                sse[k] += ss
                sso[k].append(ss)
                sae[k] += sum(abs(v) for v in e)
                if s > 0:
                    wr[k] += w * math.sqrt(ss / H / s)
                if bi is not None:
                    b = bins_acc.setdefault((bi, k), [0, 0.0, 0.0, 0])
                    b[0] += 1
                    b[1] += ss
                    b[2] += sum(abs(v) for v in e)
                    b[3] += H
                for a, bb in HORS:
                    hv = hor_acc.setdefault((f"{a}-{bb}", k), [0.0, 0])
                    hv[0] += sum(v * v for v in e[a - 1:bb])
                    hv[1] += bb - a + 1
            if s > 0:
                wsum += w
                nvalid += 1
        for k in METHODS:
            by_origin[(d, k)] = (math.sqrt(sse[k] / (3000 * H)), sae[k] / (3000 * H), wr[k] / wsum, nvalid)
        print("origin", d, "recounted", flush=True)
    for r in read("scores_by_origin.csv"):
        v = by_origin[(int(r["origin"]), r["method"])]
        same(r["rmse"], v[0], f"rmse {r['origin']} {r['method']}")
        same(r["mae"], v[1], f"mae {r['origin']} {r['method']}")
        same(r["wrmsse"], v[2], f"wrmsse {r['origin']} {r['method']}")
        same(r["series_scored"], v[3], "series scored", True)
    for r in read("scores_overall.csv"):
        k = r["method"]
        for j, c in enumerate(("rmse", "mae", "wrmsse")):
            same(r[c], sum(by_origin[(d, k)][j] for d in ORIGINS) / len(ORIGINS), f"overall {c} {k}")
    for r in read("by_intermittency.csv"):
        b = bins_acc[(r["bin"], r["method"])]
        same(r["series"], b[0], "bin series", True)
        same(r["points"], b[3], "bin points", True)
        same(r["rmse"], math.sqrt(b[1] / b[3]), f"bin rmse {r['bin']} {r['method']}")
        same(r["mae"], b[2] / b[3], f"bin mae {r['bin']} {r['method']}")
    for r in read("by_horizon.csv"):
        hv = hor_acc[(r["horizon"], r["method"])]
        same(r["points"], hv[1], "horizon points", True)
        same(r["rmse"], math.sqrt(hv[0] / hv[1]), f"horizon rmse {r['horizon']} {r['method']}")
    for r in read("concentration.csv"):
        v = sorted(sso[r["method"]])
        n = len(v)
        same(r["series_origins"], n, "series-origins", True)
        same(r["share_sse_worst20"], sum(v[-20:]) / sum(v), "share worst 20")
        same(r["rmse_all"], math.sqrt(sum(v) / (n * H)), "pooled rmse")
        same(r["rmse_without_worst20"], math.sqrt(sum(v[:-20]) / ((n - 20) * H)), "rmse without worst 20")
    for r in read("worst_case.csv"):
        m = r["method"]
        v = sso[m]
        j = max(range(len(v)), key=lambda i: v[i])
        assert (skeys[j][1], str(skeys[j][0])) == (r["id"], r["origin"]), (m, skeys[j], r["id"], r["origin"])
        same(r["share_of_method_sse"], v[j] / sum(v), f"worst share {m}")
        same(r["max_forecast"], fmx[m][j], f"max forecast {m}")
        same(r["max_sale_before_origin"], hmx[j], "max sale before origin")
        same(r["mean_sales_28d_before"], mbf[j], "mean sales before")
        same(r["mean_actual"], maf[j], "mean actual")
        same(r["mean_forecast"], fmn[m][j], f"mean forecast {m}")
    kh = max(range(len(sso["hgb"])), key=lambda i: sso["hgb"][i])
    for r in read("without_worst_case.csv"):
        m = r["method"]
        bands = HORS if r["scope"] == "pooled" else [tuple(int(x) for x in r["scope"].split("-"))]
        days = 28 if r["scope"] == "pooled" else bands[0][1] - bands[0][0] + 1
        arrs = [ssb[m][b] for b in bands]
        n = len(arrs[0])
        same(r["rmse_all"], math.sqrt(sum(sum(a) for a in arrs) / (n * days)), f"rmse {r['scope']} {m}")
        same(r["rmse_without_case"], math.sqrt(sum(sum(a) - a[kh] for a in arrs) / ((n - 1) * days)), f"rmse without case {r['scope']} {m}")
    imp = read("importance.csv")
    assert sorted(r["feature"] for r in imp) == sorted(FEATS)
    count += 1
    print(f"check.py: {count} recomputed numbers agree with results/")


if __name__ == "__main__":
    main()
