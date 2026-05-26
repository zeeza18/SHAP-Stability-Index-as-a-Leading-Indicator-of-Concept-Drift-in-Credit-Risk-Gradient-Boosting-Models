"""Figure 4: Ablation study grouped bar chart."""

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

ABLATION_CONDITIONS = [
    "Full System",
    "No Drift Detection",
    "No Adaptive Retraining",
    "Static Baseline",
]
ABLATION_COLORS = ["#2C7BB6", "#74C476", "#F4A460", "#D73027"]


def plot_ablation_bars(
    ablation_results: dict[str, dict[str, float]],
    datasets: list[str],
    metric: str = "mean_roc_auc",
    ci: Optional[dict[str, dict[str, float]]] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generate Figure 4: ablation study grouped bar chart.

    Args:
        ablation_results: {condition: {dataset: mean_metric}}.
        datasets: List of dataset names (x-axis groups).
        metric: Metric label for y-axis.
        ci: Optional {condition: {dataset: half_ci_width}} for error bars.
        output_dir: Save directory.

    Returns:
        Path to saved PNG.
    """
    conditions = list(ablation_results.keys())
    n_conditions = len(conditions)
    n_datasets = len(datasets)
    x = np.arange(n_datasets)
    width = 0.8 / n_conditions

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, condition in enumerate(conditions):
        heights = [ablation_results[condition].get(ds, 0) for ds in datasets]
        errors = None
        if ci and condition in ci:
            errors = [ci[condition].get(ds, 0) for ds in datasets]

        color = ABLATION_COLORS[i % len(ABLATION_COLORS)]
        offset = (i - n_conditions / 2 + 0.5) * width
        ax.bar(
            x + offset,
            heights,
            width=width * 0.9,
            label=condition,
            color=color,
            yerr=errors,
            capsize=4,
            error_kw={"linewidth": 1.2},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title("Ablation Study: Contribution of Each Component", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_ylim(0.75, 1.0)

    plt.tight_layout()
    out_dir = output_dir or FIGURES_DIR
    png_path = out_dir / "fig4_ablation_bars.png"
    pdf_path = out_dir / "fig4_ablation_bars.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 4 saved: {png_path}")
    return png_path
