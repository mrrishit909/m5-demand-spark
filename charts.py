"""Step 6: charts.   ./venv/bin/python charts.py -> charts/00_cover.png, 01_scores.png, 02_origins.png, 03_horizon.png, 04_intermittency.png, 05_importance.png"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
R = HERE / "results"
INK, DIM, GRID, BG, BLUE, ORANGE, GRAY, GREEN = "#f2f2f0", "#8a8a87", "#1d1d1d", "#0b0b0b", "#3987e5", "#d95926", "#9a9a96", "#3aa876"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": INK,
                     "axes.edgecolor": "#3a3a3a", "axes.labelcolor": DIM, "xtick.color": DIM, "ytick.color": DIM,
                     "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 1,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 11, "axes.titlesize": 13,
                     "axes.titlelocation": "left", "axes.titlepad": 12})
OV = pd.read_csv(R / "scores_overall.csv").set_index("method")
BO = pd.read_csv(R / "scores_by_origin.csv")
HZ = pd.read_csv(R / "by_horizon.csv")
IB = pd.read_csv(R / "by_intermittency.csv")
IM = pd.read_csv(R / "importance.csv")
NAME = {"naive": "last day", "seasonal_naive_7": "same day last week", "moving_avg_28": "28-day average", "zero": "always zero", "hgb": "boosted trees", "hgb_gap28": "boosted, 28-day-old inputs"}
COL = {"naive": GRAY, "seasonal_naive_7": "#6d6d6a", "moving_avg_28": ORANGE, "zero": "#4a4a48", "hgb": BLUE, "hgb_gap28": GREEN}
ORDER = ["naive", "seasonal_naive_7", "moving_avg_28", "zero", "hgb_gap28", "hgb"]
BINS = ["0-10%", "10-30%", "30-50%", "50-70%", "70-100%"]


def save(fig, name):
    fig.tight_layout(); fig.savefig(HERE / "charts" / name, dpi=150); plt.close(fig)


def bars(ax, col, title, fs=8.5, fmt="{:.2f}"):
    x = np.arange(len(ORDER)); v = [OV.loc[m, col] for m in ORDER]
    ax.bar(x, v, color=[COL[m] for m in ORDER], width=0.72)
    for xi, vi in zip(x, v):
        ax.text(xi, vi, fmt.format(vi), ha="center", va="bottom", fontsize=fs, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(["last\nday", "last\nweek", "28-day\navg", "zero", "boosted\n(old inputs)", "boosted"], fontsize=8.5)
    ax.set_title(title, fontsize=12); ax.grid(axis="x", visible=False); ax.set_ylim(0, max(v) * 1.15)


WC = pd.read_csv(R / "without_worst_case.csv")
WORST = pd.read_csv(R / "worst_case.csv").set_index("method").loc["hgb"]


def horizon(ax, without=False, methods=("seasonal_naive_7", "moving_avg_28", "hgb")):
    hs = ["1-7", "8-14", "15-21", "22-28"]; x = np.arange(4); w = 0.26
    for i, m in enumerate(methods):
        v = [WC[(WC.scope == h) & (WC.method == m)].rmse_without_case.iloc[0] for h in hs] if without else [HZ[(HZ.horizon == h) & (HZ.method == m)].rmse.iloc[0] for h in hs]
        ax.bar(x + (i - 1) * w, v, width=w * 0.92, color=COL[m], label=NAME[m])
        for xi, vi in zip(x + (i - 1) * w, v):
            ax.text(xi, vi, f"{vi:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([f"days {h}" for h in hs]); ax.set_ylabel("RMSE (units per day)"); ax.grid(axis="x", visible=False); ax.set_ylim(0, 4.7)
    ax.legend(frameon=False, labelcolor=INK, fontsize=8.5, loc="upper right", ncol=3)


def intermittency(axes):
    for ax, col, ttl in ((axes[0], "rmse", "RMSE by share of zero-sales days"), (axes[1], "mae", "MAE by share of zero-sales days")):
        x = np.arange(5); w = 0.26
        for i, m in enumerate(("zero", "moving_avg_28", "hgb")):
            v = [IB[(IB.bin == b) & (IB.method == m)][col].iloc[0] for b in BINS]
            ax.bar(x + (i - 1) * w, v, width=w * 0.92, color=COL[m] if m != "zero" else "#6d6d6a", label=NAME[m])
        ax.set_xticks(x); ax.set_xticklabels(BINS, fontsize=9); ax.set_title(ttl, fontsize=12); ax.grid(axis="x", visible=False)
        ax.set_xlabel("zero-sales days in the 364 days before the forecast")
    axes[0].legend(frameon=False, labelcolor=INK, fontsize=8.5, loc="upper right")


def cover():
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    a, h = OV.loc["hgb"], OV.loc["moving_avg_28"]
    fig.suptitle(f"Boosted trees edge out a 28-day average on the scaled, revenue-weighted error ({a.wrmsse:.3f} vs {h.wrmsse:.3f}), not on RMSE ({a.rmse:.2f} vs {h.rmse:.2f})", fontsize=15, x=0.02, ha="left", y=0.98)
    bars(axes[0], "wrmsse", "WRMSSE-style error (lower is better)", fs=10)
    horizon(axes[1]); axes[1].set_title("RMSE by forecast day: the week-1 loss traces to one series", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(HERE / "charts" / "00_cover.png", dpi=100); plt.close(fig)


def main():
    (HERE / "charts").mkdir(exist_ok=True)
    cover()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.9))
    bars(axes[0], "rmse", "RMSE"); bars(axes[1], "mae", "MAE"); bars(axes[2], "wrmsse", "WRMSSE-style")
    for ax in axes:
        ax.set_xticks([])
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COL[m]) for m in ORDER], labels=[NAME[m] for m in ORDER], loc="lower center", ncol=6, frameon=False, labelcolor=INK, fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(HERE / "charts" / "01_scores.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for m in ("naive", "seasonal_naive_7", "moving_avg_28", "hgb"):
        d = BO[BO.method == m]; ax.plot(range(5), d.wrmsse, marker="o", color=COL[m], linewidth=2.4 if m == "hgb" else 1.8, label=NAME[m])
    ax.set_xticks(range(5)); ax.set_xticklabels([f"d_{o}" + ("\n(M5 validation window)" if o == 1913 else "") for o in sorted(BO.origin.unique())], fontsize=9)
    ax.set_ylabel("WRMSSE-style error"); ax.set_ylim(0.7, 1.3); ax.set_xlabel("forecast origin (28 days ahead of each)"); ax.legend(frameon=False, labelcolor=INK, fontsize=9, ncol=2, loc="upper right")
    save(fig, "02_origins.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0)); horizon(axes[0]); horizon(axes[1], without=True)
    axes[0].set_title("All 15,000 series-origins", fontsize=12); axes[1].set_title(f"Without {WORST.id.replace('_evaluation', '')} at d_{WORST.origin}", fontsize=12)
    save(fig, "03_horizon.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1)); intermittency(axes); save(fig, "04_intermittency.png")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    t = IM.head(12).iloc[::-1]
    ax.barh(t.feature, t.rmse_increase, color=BLUE, height=0.7)
    for yi, v in enumerate(t.rmse_increase):
        ax.text(v, yi, f" {v:.2f}", va="center", fontsize=8.5, color=INK)
    ax.set_xlabel("RMSE added when the column is shuffled (units per day, mean of 5 origins)"); ax.grid(axis="y", visible=False); ax.set_xlim(0, t.rmse_increase.max() * 1.12)
    save(fig, "05_importance.png")


if __name__ == "__main__":
    main()
