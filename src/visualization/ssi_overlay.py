"""Figure 3: SSI as leading indicator (dual-axis: SSI + AUC).

The headline figure of the paper — shows SSI dropping BEFORE AUC drops.
"""

import random
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

np.random.seed(42)
random.seed(42)

FIGURES_DIR = Path(__file__).resolve().parents[3] / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 11})


def plot_ssi_overlay(
    ssi_df: pd.DataFrame,
    drift_windows: list[int],
    dataset_name: str,
    model_name: str = "AdaptiveXGBoost",
    ssi_threshold: float = 0.7,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generate Figure 3: SSI vs AUC dual-axis with drift event annotations.

    Args:
        ssi_df: DataFrame with columns: window_index, ssi_value, auc_this_window.
        drift_windows: Windows where ADWIN fired.
        dataset_name: Used in title and file name.
        model_name: Used in title.
        ssi_threshold: Horizontal dashed line showing SSI alert level.
        output_dir: Save directory.

    Returns:
        Path to saved PNG.
    """
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()

    windows = ssi_df["window_index"].values
    ssi_vals = ssi_df["ssi_value"].values
    auc_vals = ssi_df["auc_this_window"].values

    # SSI area chart (left axis)
    ax1.fill_between(windows, ssi_vals, alpha=0.35, color="#F4A460", label="SSI (area)")
    ax1.plot(windows, ssi_vals, color="#D2691E", linewidth=1.5)
    ax1.axhline(y=ssi_threshold, color="#D2691E", linestyle=":", linewidth=1.0, alpha=0.7)
    ax1.set_ylabel("SSI Value", fontsize=12, color="#D2691E")
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(axis="y", labelcolor="#D2691E")

    # AUC line (right axis)
    ax2.plot(windows, auc_vals, color="#1A3A6B", linewidth=2.2, label="AUC-ROC")
    ax2.set_ylabel("AUC-ROC", fontsize=12, color="#1A3A6B")
    ax2.set_ylim(0.75, 1.0)
    ax2.tick_params(axis="y", labelcolor="#1A3A6B")

    # Drift event markers
    for d in drift_windows:
        ax1.axvline(x=d, color="red", linestyle="--", linewidth=1.2, alpha=0.8)

    # Annotate lead-time arrows: find where SSI drops before AUC at each drift
    for d in drift_windows:
        pre = ssi_df[ssi_df["window_index"] < d]
        if pre.empty:
            continue
        ssi_drop = pre[pre["ssi_value"] < ssi_threshold]["window_index"]
        if ssi_drop.empty:
            continue
        ssi_drop_w = int(ssi_drop.min())
        ax1.annotate(
            "",
            xy=(d, ssi_threshold + 0.05),
            xytext=(ssi_drop_w, ssi_threshold + 0.05),
            arrowprops=dict(arrowstyle="->", color="darkred", lw=1.5),
        )

    ax1.set_xlabel("Window Index", fontsize=12)
    ax1.set_title(
        f"SSI as Leading Indicator of AUC Degradation — {dataset_name} ({model_name})",
        fontsize=13,
    )

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=10)

    plt.tight_layout()
    out_dir = output_dir or FIGURES_DIR
    png_path = out_dir / f"fig3_ssi_overlay_{dataset_name}_{model_name}.png"
    pdf_path = out_dir / f"fig3_ssi_overlay_{dataset_name}_{model_name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 3 saved: {png_path}")
    return png_path
