"""Generate fig5: lead-time (lambda) distribution histogram from ssi_lead_time.csv."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "tables" / "ssi_lead_time.csv"
OUT_DIR = ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
})

df = pd.read_csv(DATA)

# Restrict to IEEE-CIS (where meaningful drift exists) and exclude window-8 (lead=0, first window)
ieee = df[df["dataset"].str.contains("IEEE")].copy()

xgb = ieee[ieee["model"] == "AdaptiveXGBoost"]["lead_time_windows"].values
lgbm = ieee[ieee["model"] == "AdaptiveLightGBM"]["lead_time_windows"].values

bins = np.arange(-0.5, 17.5, 1)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

COLOR_XGB = "#2166ac"
COLOR_LGBM = "#d6604d"

for ax, data, color, label, mean_c in [
    (axes[0], xgb, COLOR_XGB, "AdaptiveXGBoost", "#084594"),
    (axes[1], lgbm, COLOR_LGBM, "AdaptiveLightGBM", "#67001f"),
]:
    ax.hist(data, bins=bins, color=color, edgecolor="white", linewidth=0.5, alpha=0.85)
    mu = data.mean()
    sd = data.std()
    ax.axvline(mu, color=mean_c, linewidth=2, linestyle="--", label=f"Mean = {mu:.1f}")
    ax.set_title(label, pad=6)
    ax.set_xlabel(r"Lead time $\lambda$ (evaluation windows)")
    ax.set_xticks(range(0, 17, 2))
    ax.legend(framealpha=0.9)
    ax.text(
        0.97, 0.95,
        f"$n$ = {len(data)}\n$\\mu$ = {mu:.2f}\n$\\sigma$ = {sd:.2f}",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

axes[0].set_ylabel("Number of drift events")

fig.suptitle(
    r"Distribution of SSI Lead Time $\lambda$ Across ADWIN Drift Events (IEEE-CIS)",
    fontsize=12, y=1.02,
)
plt.tight_layout()

for ext in ("png", "pdf"):
    p = OUT_DIR / f"fig5_lambda_hist.{ext}"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"Saved: {p}")

plt.close(fig)
