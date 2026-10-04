"""Steps 2-3 (PySpark): audit the full M5 table at scale, then build the long-form task/feature table for a documented sample of series.

    ./venv/bin/python features.py   -> results/full_vs_sample.csv, results/sample_items.csv,
                                       data/base.parquet, data/features.parquet (neither committed)

Spark 4 in local[4] mode, driver memory 3g (the machine is shared).

Sampling rule (fixed before looking at any score): rank the 3,049 items by md5(item_id) (hex string, ascending), keep the first 300,
and keep all 10 stores for each: 3,000 store-item series x 1,941 days = 5.82 M rows. md5 order is arbitrary with respect to department,
sales volume or intermittency, and anyone can reproduce it. results/full_vs_sample.csv checks the sample against all 30,490 series
(share of zero-sales days and units per day, from each series' first sale to d_1913).

Leak-free design. A forecast is made at an origin o for the target days t = o+1..o+28 (horizon h = t-o). data/features.parquet has one row per
(series, target day t, origin o):
  - training rows (origin = 0): every t >= 1071 with h = t mod 28 + 1, so each horizon appears equally often and o = t-h < t;
  - validation rows (origin = d): for each evaluation origin d in 1801, 1829, 1857, 1885, 1913, t = d+1..d+28, o = d.
Main model ("hgb"): every sales feature is anchored at the origin o (sales on day o, means over 7/28/91 days ending at o, standard deviation and
share of selling days over 28, an expanding mean, days since the last sale, the same-weekday sales 1-3 weeks back that are <= o) plus the horizon h.
Ablation ("hgb_gap28", the first design): features anchored at t-28 (weekly lags 28/35/42, 7/28/91-day means ending at t-28, ...), which are legal
for any horizon up to 28 but 28 days old at the longest. Calendar, events, SNAP and the shelf price of day t (and its changes) are treated as known in
advance (the competition publishes them for the forecast window; a retailer plans them). The item enters only through its own history (expanding mean),
never through an ordinal code. Missing values (history not long enough) stay null.
"""
import hashlib
import json
from pathlib import Path

from pyspark.sql import SparkSession, Window, functions as F, types as T

HERE = Path(__file__).parent
DATA = HERE / "data"
RES = HERE / "results"
NDAYS = 1941            # d_1 .. d_1941 are the labelled days in sales_train_evaluation.csv
NITEMS = 300
ORIGINS = [1801, 1829, 1857, 1885, 1913]
L = 28                  # information gap = forecast horizon


def main():
    spark = (SparkSession.builder.master("local[4]").appName("m5-demand")
             .config("spark.driver.memory", "3g").config("spark.sql.shuffle.partitions", "16")
             .config("spark.ui.enabled", "false").config("spark.sql.session.timeZone", "UTC").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    dcols = [f"d_{i}" for i in range(1, NDAYS + 1)]
    schema = T.StructType([T.StructField(c, T.StringType()) for c in ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]]
                          + [T.StructField(c, T.IntegerType()) for c in dcols])
    wide = spark.read.csv(str(DATA / "sales_train_evaluation.csv"), header=True, schema=schema)
    stack = "stack(%d, %s) as (d, y)" % (NDAYS, ", ".join(f"'{c}', {c}" for c in dcols))
    long_all = wide.select("id", "item_id", F.expr(stack)).withColumn("dnum", F.regexp_replace("d", "d_", "").cast("int")).drop("d")

    # --- the sample: first 300 items by md5(item_id)
    items = sorted([r[0] for r in wide.select("item_id").distinct().collect()], key=lambda s: hashlib.md5(s.encode()).hexdigest())
    sample = set(items[:NITEMS])
    assert len(items) == 3049
    (RES / "sample_items.csv").write_text("item_id\n" + "\n".join(sorted(sample)) + "\n")

    # --- audit at scale: per-series zero share and mean over d_1..d_1913 (from the first sale), all 30,490 series vs the sample
    first = (long_all.where("y > 0").groupBy("id").agg(F.min("dnum").alias("first_d")))
    ser = (long_all.where("dnum <= 1913").join(first, "id").where("dnum >= first_d").groupBy("id", "item_id")
           .agg(F.count("*").alias("n"), F.sum((F.col("y") == 0).cast("int")).alias("zeros"), F.sum("y").alias("units")))
    ser = ser.withColumn("zshare", F.col("zeros") / F.col("n")).withColumn("in_sample", F.col("item_id").isin(list(sample)))
    ser = ser.cache()
    rows = []
    for name, df in (("all 30,490 series", ser), ("sample 3,000 series", ser.where("in_sample"))):
        r = df.agg(F.count("*").alias("series"), F.avg("zshare").alias("mean_zero_share"), F.expr("percentile_approx(zshare, 0.5, 100000)").alias("median_zero_share"),
                   F.avg((F.col("zshare") > 0.5).cast("int")).alias("share_series_zero_gt_50"), F.avg(F.col("units") / F.col("n")).alias("mean_units_per_day"),
                   F.sum("units").alias("total_units"), F.sum("n").alias("series_days")).first()
        rows.append((name,) + tuple(r))
    import csv
    with open(RES / "full_vs_sample.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "series", "mean_zero_share", "median_zero_share", "share_series_zero_gt_50", "mean_units_per_day", "total_units", "series_days"])
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.10f}", f"{r[3]:.10f}", f"{r[4]:.10f}", f"{r[5]:.10f}", int(r[6]), int(r[7])])
    print(rows)
    ser.unpersist()

    # --- long form + joins for the sample
    cal = (spark.read.csv(str(DATA / "calendar.csv"), header=True, inferSchema=True)
           .withColumn("dnum", F.regexp_replace("d", "d_", "").cast("int"))
           .select("dnum", "date", "wm_yr_wk", "wday", "month", "event_type_1", "event_type_2", "snap_CA", "snap_TX", "snap_WI"))
    price = spark.read.csv(str(DATA / "sell_prices.csv"), header=True, inferSchema=True)
    base = (wide.where(F.col("item_id").isin(list(sample))).select("id", "item_id", "dept_id", "cat_id", "store_id", "state_id", F.expr(stack))
            .withColumn("dnum", F.regexp_replace("d", "d_", "").cast("int")).drop("d"))
    df = base.join(F.broadcast(cal), "dnum", "left").join(price, ["store_id", "item_id", "wm_yr_wk"], "left").withColumnRenamed("sell_price", "price")
    snap = (F.when(F.col("state_id") == "CA", F.col("snap_CA")).when(F.col("state_id") == "TX", F.col("snap_TX")).otherwise(F.col("snap_WI")))
    df = df.withColumn("snap", snap).drop("snap_CA", "snap_TX", "snap_WI")

    w = Window.partitionBy("id").orderBy("dnum")
    wr = lambda a, b: w.rowsBetween(a, b)
    yv = F.col("y").cast("double")
    nz = (F.col("y") > 0).cast("double")
    # (a) features anchored at the day itself (r_*): what is known at an origin o if the row is read at day o
    df = (df.withColumn("r_m7", F.avg(yv).over(wr(-6, 0))).withColumn("r_m28", F.avg(yv).over(wr(-27, 0))).withColumn("r_m91", F.avg(yv).over(wr(-90, 0)))
          .withColumn("r_std28", F.stddev(yv).over(wr(-27, 0))).withColumn("r_share28", F.avg(nz).over(wr(-27, 0)))
          .withColumn("r_exp", F.avg(yv).over(w.rowsBetween(Window.unboundedPreceding, 0)))
          .withColumn("r_last_sale", F.max(F.when(F.col("y") > 0, F.col("dnum"))).over(w.rowsBetween(Window.unboundedPreceding, 0)))
          # (b) the 28-day-gap features (g_*): anchored at day-28, so legal for any horizon up to 28 (ablation model)
          .withColumn("g_m7", F.avg(yv).over(wr(-34, -28))).withColumn("g_m28", F.avg(yv).over(wr(-55, -28))).withColumn("g_m91", F.avg(yv).over(wr(-118, -28)))
          .withColumn("g_std28", F.stddev(yv).over(wr(-55, -28))).withColumn("g_share28", F.avg(nz).over(wr(-55, -28)))
          .withColumn("g_exp", F.avg(yv).over(w.rowsBetween(Window.unboundedPreceding, -28))))
    for j in range(1, 10):                                          # y_l{7j}: sales 7j days earlier
        df = df.withColumn(f"y_l{7 * j}", F.lag("y", 7 * j).over(w))
    df = (df.withColumn("price_chg_w", F.col("price") / F.lag("price", 7).over(w) - 1)
          .withColumn("price_rel91", F.col("price") / F.avg("price").over(wr(-90, 0)))
          .withColumn("has_event", ((F.col("event_type_1").isNotNull()) | (F.col("event_type_2").isNotNull())).cast("int"))
          .withColumn("dom", F.dayofmonth("date")))
    df.write.mode("overwrite").parquet(str(DATA / "base.parquet"))
    bs = spark.read.parquet(str(DATA / "base.parquet"))

    # --- task table: one row per (series, target day t, origin o = t-h)
    #  training rows: every t >= 1071, with horizon h = t mod 28 + 1 (each horizon equally often, origin o = t-h < t)
    #  validation rows: for each evaluation origin d, t = d+1..d+28, h = t-d, o = d  (origin column = d; training rows carry origin 0)
    tg = bs.where("dnum >= 1071").drop("r_m7", "r_m28", "r_m91", "r_std28", "r_share28", "r_exp", "r_last_sale").withColumnRenamed("dnum", "t")
    tr = tg.withColumn("h", F.col("t") % 28 + 1).withColumn("origin", F.lit(0))
    va = None
    for d in ORIGINS:
        part = tg.where((F.col("t") > d) & (F.col("t") <= d + 28)).withColumn("h", F.col("t") - d).withColumn("origin", F.lit(d))
        va = part if va is None else va.unionByName(part)
    tasks = tr.unionByName(va).withColumn("o", F.col("t") - F.col("h"))
    o_side = bs.select("id", F.col("dnum").alias("o"), F.col("y").alias("y_o"), F.col("price").alias("price_o"),
                       *[F.col(c) for c in ("r_m7", "r_m28", "r_m91", "r_std28", "r_share28", "r_exp")], (F.col("dnum") - F.col("r_last_sale")).alias("days_since_sale"))
    out = tasks.join(o_side, ["id", "o"], "left")
    k0 = F.ceil(F.col("h") / 7).cast("int")
    arr = F.array(*[F.col(f"y_l{7 * j}") for j in range(1, 10)])
    out = (out.withColumn("lag_w1", F.element_at(arr, k0)).withColumn("lag_w2", F.element_at(arr, k0 + 1)).withColumn("lag_w3", F.element_at(arr, k0 + 2))
           .withColumn("price_vs_o", F.col("price") / F.col("price_o"))
           .withColumn("g_lag28", F.col("y_l28")).withColumn("g_lag35", F.col("y_l35")).withColumn("g_lag42", F.col("y_l42"))
           .withColumn("g_dow4", (F.col("y_l28") + F.col("y_l35") + F.col("y_l42") + F.col("y_l49")) / 4.0)
           .select("id", "t", "origin", "h", "y", "store_id", "state_id", "dept_id", "cat_id", "event_type_1", "wday", "month", "dom", "has_event", "snap",
                   "price", "price_chg_w", "price_rel91", "price_vs_o", "y_o", "r_m7", "r_m28", "r_m91", "r_std28", "r_share28", "r_exp", "days_since_sale",
                   "lag_w1", "lag_w2", "lag_w3", "g_lag28", "g_lag35", "g_lag42", "g_dow4", "g_m7", "g_m28", "g_m91", "g_std28", "g_share28", "g_exp"))
    out.repartition(8).write.mode("overwrite").parquet(str(DATA / "features.parquet"))
    n = spark.read.parquet(str(DATA / "features.parquet"))
    print("feature rows", n.count(), "validation rows", n.where("origin > 0").count())
    spark.stop()


if __name__ == "__main__":
    main()
