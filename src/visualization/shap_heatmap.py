"""Figure 2: SHAP rank heatmap — top 20 features across all windows."""

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


def plot_shap_heatmap(
    shap_history: dict[int, dict],
    drift_windows: list[int],
    dataset_name: str,
    model_names: tuple[str, str] = ("StaticXGBoost", "AdaptiveXGBoost"),
    top_k: int = 20,
    output_dir: Optional[Path] = None,
) -> Path:
    """Generate Figure 2: SHAP rank heatmap.

    Args:
        shap_history: Dict mapping window_index -> {model_name: mean_abs_shap Series}.
        drift_windows: Window indices for vertical markers.
        dataset_name: Used in title and file name.
        model_names: Tuple of (top subplot model, bottom subplot model).
        top_k: Number of top features to display.
        output_dir: Save directory.

    Returns:
        Path to saved PNG.
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    for ax, model_name in zip(axes, model_names):
        windows = sorted(shap_history.keys())
        # Build matrix: features x windows
        all_features: set = set()
        for w in windows:
            if model_name in shap_history[w]:
                all_features.update(shap_history[w][model_name].index)

        # Use top-k features from the last window
        last_win = windows[-1]
        if model_name in shap_history.get(last_win, {}):
            top_features = (
                shap_history[last_win][model_name]
                .sort_values(ascending=False)
                .head(top_k)
                .index.tolist()
            )
        else:
            top_features = list(all_features)[:top_k]

        matrix = pd.DataFrame(index=top_features, columns=windows, dtype=float)
        for w in windows:
            if model_name in shap_history.get(w, {}):
                vals = shap_history[w][model_name]
                for feat in top_features:
                    matrix.loc[feat, w] = vals.get(feat, 0.0)
        matrix = matrix.fillna(0)

        sns.heatmap(
            matrix,
            ax=ax,
            cmap="RdBu_r",
            cbar_kws={"label": "Mean |SHAP value|"},
            linewidths=0,
        )
        ax.set_title(model_name, fontsize=12)
        ax.set_ylabel("Feature", fontsize=10)

        for d in drift_windows:
            if d in windows:
                col_idx = windows.index(d)
                ax.axvline(x=col_idx, color="red", linewidth=1.2, linestyle="--")

    axes[-1].set_xlabel("Window Index", fontsize=11)
    fig.suptitle(f"SHAP Feature Importance Heatmap — {dataset_name}", fontsize=13, y=1.01)
    plt.tight_layout()

    out_dir = output_dir or FIGURES_DIR
    png_path = out_dir / f"fig2_shap_heatmap_{dataset_name}.png"
    pdf_path = out_dir / f"fig2_shap_heatmap_{dataset_name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 2 saved: {png_path}")
    return png_path
