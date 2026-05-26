"""Statistical significance tests: Wilcoxon signed-rank for adaptive vs static."""

import random
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(42)
random.seed(42)


def wilcoxon_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alternative: str = "two-sided",
) -> dict:
    """Wilcoxon signed-rank test comparing two paired score arrays.

    Args:
        scores_a: Metric values for model A (e.g., adaptive) per window.
        scores_b: Metric values for model B (e.g., static) per window.
        alternative: 'two-sided', 'greater', or 'less'.

    Returns:
        Dict with W_statistic, p_value, effect_size_r, significant.
    """
    result = stats.wilcoxon(scores_a, scores_b, alternative=alternative)
    n = len(scores_a)
    # Effect size r = Z / sqrt(N)
    z_approx = stats.norm.ppf(result.pvalue / 2) if result.pvalue < 1 else 0.0
    effect_size_r = abs(z_approx) / np.sqrt(n)

    return {
        "W_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "effect_size_r": effect_size_r,
        "n_windows": n,
        "significant_at_005": result.pvalue < 0.05,
        "significant_at_001": result.pvalue < 0.01,
        "alternative": alternative,
    }


def run_all_comparisons(
    results: dict[str, pd.DataFrame],
    metric: str = "roc_auc",
) -> pd.DataFrame:
    """Run Wilcoxon tests for all adaptive vs static pairs.

    Args:
        results: Dict mapping model_name -> per-window metrics DataFrame.
        metric: Column name to compare.

    Returns:
        DataFrame with one row per comparison.
    """
    rows = []
    adaptive_models = [k for k in results if k.startswith("Adaptive")]
    static_models = [k for k in results if k.startswith("Static")]

    for adap in adaptive_models:
        for stat in static_models:
            # Align on window_index
            merged = results[adap][["window_index", metric]].merge(
                results[stat][["window_index", metric]],
                on="window_index",
                suffixes=("_adaptive", "_static"),
            ).dropna()

            if len(merged) < 5:
                continue

            test = wilcoxon_test(
                merged[f"{metric}_adaptive"].values,
                merged[f"{metric}_static"].values,
                alternative="greater",  # adaptive > static
            )
            rows.append({
                "adaptive_model": adap,
                "static_model": stat,
                "metric": metric,
                **test,
            })

    return pd.DataFrame(rows)
