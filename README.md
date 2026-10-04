# Can boosted trees beat a 28-day average at forecasting Walmart demand?

Daily unit sales of Walmart store-item series from the M5 forecasting competition, forecast 28 days ahead with PySpark-built features and a gradient-boosted model, against simple benchmarks. Five questions:
- does a boosted model beat the naive, seasonal-naive and moving-average forecasts out of sample, on RMSE, MAE and a scaled, revenue-weighted error (a WRMSSE-style score);
- does the answer hold at five forecast origins (28 days apart, the last being the M5 "validation" window), with no random splits;
- how does the error change with the forecast day (1-7 ... 22-28), and what happens if the model may only use inputs that are 28 days old;
- how does it change with intermittency (the share of zero-sales days), where RMSE and MAE disagree;
- which inputs the model uses, and how much of each method's error sits in a handful of series.

The task follows DataCamp's "Building a Demand Forecasting Model" (PySpark: temporal features, forecasting at scale, e-commerce demand). The page with the results and the charts is the build log:

Build log: https://mrrishit909.github.io/projects/m5-demand-spark/

## Data

The M5 Forecasting Accuracy dataset (Makridakis Open Forecasting Center; Walmart sales in California, Texas and Wisconsin, 2011-01-29 to 2016-06-19): `calendar.csv`, `sell_prices.csv` and `sales_train_evaluation.csv` from the Zenodo record https://zenodo.org/records/12636070 (DOI 10.5281/zenodo.12636070), licence CC BY 4.0 (checked on the record). Cite: Makridakis, Spiliotis, Assimakopoulos (2022), "The M5 competition: Background, organization, and implementation", International Journal of Forecasting 38(4). The three files are 325 MB (the zip is 48 MB), are fetched by `download.py` (checksums in `results/pull.json`) and are not committed; `results/predictions.csv.gz` holds the model forecasts.

```
./venv/bin/pip install pyspark pandas numpy scikit-learn pyarrow matplotlib certifi   # Java 17+ needed for Spark
./venv/bin/python download.py        # data/*
./venv/bin/python features.py        # Spark local[4], 3 GB driver: audit + features (about 3 minutes)
./venv/bin/python analyze.py         # results/*  (about 20 minutes on a busy laptop: 10 boosted fits)
./venv/bin/python concentration.py   # results/concentration.csv
./venv/bin/python check.py           # plain-Python recount (seconds)
./venv/bin/python charts.py
```

## Steps

1. **Download** the three files with their checksums.
2. **Spark audit and sample.** All 30,490 series are stacked to long form in Spark and each series' share of zero-sales days (from its first sale to d_1913) is computed. A sample is taken by a rule fixed beforehand: the first 300 of the 3,049 items by md5(item_id), all 10 stores: 3,000 series x 1,941 days = 5.82 million rows.
3. **Spark features.** Window functions per series: weekly lags, rolling means and standard deviation over 7/28/91 days, the share of selling days, days since the last sale, an expanding mean (the time-aware id encoding: an item enters through its own history), shelf price and its changes, calendar, event and SNAP columns, store/state/department/category codes. Every feature of a target day t is anchored at the forecast origin o = t - h (h = 1..28), so nothing after the origin is used; calendar, events, SNAP and the shelf price of day t are treated as known in advance. A second, deliberately stale feature set anchored at t - 28 (weekly lags 28/35/42, rolling statistics of the 28-day-shifted series) is built for comparison.
4. **Time-aware evaluation.** Five origins d = 1801, 1829, 1857, 1885, 1913 (28 days apart; the last, d_1913, is the M5 validation window d_1914-d_1941). At each origin every method sees only days up to d and forecasts the next 28. Benchmarks: last day repeated, last week repeated, the 28-day mean, and always zero (a reference for how MAE treats intermittent demand). Model: sklearn HistGradientBoosting (Poisson loss, 300 iterations, learning rate 0.1, 63 leaves, 200 rows per leaf; one fixed setting, nothing tuned) trained per origin on the 2,190,000 (series, target day) rows of the previous 730 days, once with the fresh inputs and once with the stale ones.
5. **Scores.** RMSE and MAE pooled over series and days; a bottom-level WRMSSE-style score (each series' RMSSE scaled by its in-sample one-day-difference error, weighted by its revenue in the last 28 days); results by origin, by forecast day, by intermittency bin; permutation importance; concentration of the error.
6. **Check.** `check.py` recomputes the sample, the audit, every benchmark forecast, scale, weight and score, and re-scores the stored model forecasts, from the raw CSVs in plain Python (csv, gzip, hashlib, math; no pandas, numpy, sklearn or Spark): 443 numbers, to a relative tolerance of 1e-8.
7. **Charts.**

## Results

- **The sample is representative of the zeros, a little busier.** In 3,000 sampled series the mean share of zero-sales days is 60.2% against 60.1% over all 30,490; 66.1% against 67.3% of series are zero more than half of the days; the sample sells 1.46 units a day per series against 1.35.
- **Scores, mean of five origins** (RMSE and MAE in units per day, WRMSSE-style unitless, lower is better):

| Method | RMSE | MAE | WRMSSE |
|---|---|---|---|
| Last day | 3.699 | 1.406 | 1.168 |
| Same day last week | 3.124 | 1.246 | 1.090 |
| 28-day average | 2.785 | 1.077 | 0.852 |
| Always zero | 5.020 | 1.482 | 1.346 |
| Boosted, old inputs | 7.717 | 1.204 | 0.860 |
| Boosted trees | 2.861 | 1.052 | 0.821 |

- **On the scaled, revenue-weighted error the boosted model wins, at every origin.** 0.821 against 0.852 for the 28-day average (3.6% lower; 5 of 5 origins), and 1.090 for last week, 1.168 for the last day. On MAE it is 2.4% lower (1.052 against 1.077; 4 of 5 origins). On RMSE it does not win: 2.861 against 2.785, ahead at 1 of 5 origins.
- **Its error by forecast day.** RMSE of the boosted model against the 28-day average: days 1-7 3.56 against 2.80; days 8-14 2.69 against 2.95; days 15-21 2.64 against 2.81; days 22-28 2.75 against 2.84. **The week-1 loss is one forecast:** FOODS_3_090_CA_3 at d_1801 (a best-seller: its largest sale before the origin is 763 units, its mean over the 28 days before it 1.7 a day, its mean in the forecast window 85.0) was forecast at up to 513 units a day, a mean of 171, and holds 17.9% of the boosted model's total squared error. With that one of 15,000 series-origins removed from every method, days 1-7 are 2.59 against 2.67 and the pooled RMSE 2.66 against 2.75: the boosted model is then ahead on RMSE too. I chose that case because it is the boosted model's worst, so those figures flatter it; the point is how little the "not on RMSE" verdict rests on.
- **The first design failed, and not only through one forecast.** With inputs anchored 28 days before the target (the only way to use weekly lags 28/35/42 safely at every horizon) the boosted model scores RMSE 7.72, WRMSSE 0.860 and MAE 1.20: worse than the 28-day average on all three. The same series holds 84.7% of its squared error (a forecast of 2362 units a day against a largest sale of 763); without it the pooled RMSE is 4.57, still above the fresh model's 2.66 and the average's 2.75.
- **Intermittency decides which metric to believe.** 42.5% of the series-origins are zero on 70% or more of the previous 364 days and 25.0% on 50-70%. For those two groups, always forecasting zero has the lowest MAE of all methods (0.329 against 0.420 for the boosted model in the sparsest group; 0.869 against 0.875), while in the sparsest group its RMSE is higher than the average's and the boosted model's (1.02 against 0.81 and 0.78). In the busiest group (under 10% zero days) the boosted RMSE is 5.97 against 5.88 for the average, and MAE 3.50 against 3.62.
- **RMSE is carried by a few series.** The worst series-origin of all six methods is the same series, FOODS_3_090_CA_3 (4 of them at d_1801); the worst 20 of 15,000 series-origins hold 42% of the 28-day average's squared error and 46% of the boosted model's. Pooled over all origins the RMSE is 2.93 against 2.85; dropping each method's own worst 20 it is 2.15 against 2.18, so the ranking on RMSE flips on 0.13% of the series-origins.
- **What the model uses** (permutation importance, RMSE added when a column is shuffled, mean of 5 origins): price 8.08, r_m91 7.29, store_id 4.00, r_m28 1.86, price_rel91 1.78, days_since_sale 1.43. (r_m91 and r_m28: mean sales over the 91 and 28 days before the origin; days_since_sale: days since the last non-zero day; price_rel91: price against its own 91-day mean.) The price level and the store code very probably stand in for the item and its volume, since no item code is given; the importance of correlated inputs is split between them.

## Not done, and caveats

- Only a sample of the data was modelled: 300 of 3,049 items (10% by an md5 rule) in all 10 stores, because of the machine's memory. The audit of zeros covers all 30,490 series; the scores cover the sample.
- The metric is the M5's WRMSSE at its lowest of 12 hierarchy levels only (series level), with weights from the last 28 days of revenue before each origin: not the official WRMSSE and not comparable to the leaderboard. The competition's hidden test period (d_1942-1969) is not in the files; the last origin is the validation window, whose labels are public.
- The feature design was changed once after seeing results (the stale-input model came first and lost to the 28-day average). The five origins therefore informed one design choice, and the reported gap is somewhat optimistic; the hyper-parameters were never tuned.
- Calendar, event, SNAP and shelf-price columns of the target day are used as known in advance; the price is not forecast. That is true in the competition's setting but would have to be planned in practice.
- The model is sklearn's HistGradientBoosting fitted on one machine to the Spark-built feature table (Spark MLlib's GBTRegressor was not run). `check.py` cannot refit it, so it re-scores the stored forecasts (4 decimals) instead, and the permutation importances are not recounted (it checks only the feature list).
- No forecast was capped or corrected: the Poisson (log-link) model over-shot the rebound of one stock-out-affected best-seller (above), and capping it after seeing the result would have been a second design change.
- Intermittency bins use the share of zero days in the 364 days before the origin; one series that launches after an origin has no bin at that origin. No hierarchy reconciliation, no probabilistic forecasts, no stock-out correction (zero sales while out of stock look like zero demand).
- Nothing from the raw data is committed.
