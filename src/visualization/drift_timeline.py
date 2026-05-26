"""Figure 1: AUC timeline with drift event markers.

4 model lines + vertical red lines at ADWIN drift events
+ shaded red band 2 windows around each event.
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

MODEL_STYLES = {
    "StaticXGBoost":    {"color": "gray",      "linestyle": "--", "label": "Static XGBoost"},
    "StaticLightGBM":   {"color": "gray",      "linestyle": "-",  "label": "Static LightGBM"},
    "AdaptiveXGBoost":  {"color": "steelblue", "linestyle": "--", "label": "Adaptive XGBoost"},
    "AdaptiveLightGBM": {"color": "steelblue", "linestyle": "-",  "label": "Adaptive LightGBM"},
}


def plot_auc_timeline(
    results: dict[str, pd.DataFrame],
    drift_windows: list[int],
    dataset_name: str,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generate Figure 1: AUC timeline with drift events.

    Args:
        results: Dict mapping model_name -> DataFrame with window_index, roc_auc.
        drift_windows: Window indices where ADWIN detected drift.
        dataset_name: Used in title and file name.
        output_dir: Where to save figures (defaults to results/figures/).

    Returns:
        Path to saved PNG file.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    for model_name, df in results.items():
        style = MODEL_STYLES.get(model_name, {"color": "black", "linestyle": "-", "label": model_name})
        ax.plot(
            df["window_index"],
            df["roc_auc"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            label=style["label"],
        )

    # Drift event markers
    for d in drift_windows:
        ax.axvline(x=d, color="red", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.axvspan(d - 2, d + 2, color="red", alpha=0.08)

    ax.set_xlabel("Window Index", fontsize=12)
    ax.set_ylabel("AUC-ROC", fontsize=12)
    ax.set_title(f"AUC Timeline with Drift Events — {dataset_name}", fontsize=13)
    ax.set_ylim(0.75, 1.0)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=10)

    plt.tight_layout()
    out_dir = output_dir or FIGURES_DIR
    png_path = out_dir / f"fig1_auc_timeline_{dataset_name}.png"
    pdf_path = out_dir / f"fig1_auc_timeline_{dataset_name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 1 saved: {png_path}")
    return png_path
