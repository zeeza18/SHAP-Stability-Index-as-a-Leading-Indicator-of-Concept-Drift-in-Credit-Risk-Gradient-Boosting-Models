"""Tracks SHAP feature rank shifts across time windows.

Computes Spearman correlation between consecutive rank vectors
as the raw material for the SSI metric.
"""

import random
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(42)
random.seed(42)


class RankShiftTracker:
    """Tracks top-K feature rank vectors across windows and computes rank correlation.

    Usage:
        tracker = RankShiftTracker(top_k=20)
        tracker.update(window_idx=0, shap_values=arr, feature_names=names)
        tracker.update(window_idx=1, shap_values=arr2, feature_names=names)
        rho = tracker.get_rho(1)  # correlation between window 1 and 0
    """

    def __init__(self, top_k: int = 20):
        self.top_k = top_k
        self._rank_history: dict[int, pd.Series] = {}  # window_index -> rank Series
        self._rho_history: dict[int, float] = {}

    def update(
        self,
        window_index: int,
        shap_values: np.ndarray,
        feature_names: list[str],
    ) -> Optional[float]:
        """Record rank vector for this window and compute rho vs previous.

        Args:
            window_index: Current window index.
            shap_values: SHAP values array (n_samples, n_features).
            feature_names: Column names matching shap_values columns.

        Returns:
            Spearman rho vs previous window, or None if first window.
        """
        mean_abs = np.abs(shap_values).mean(axis=0)
        ranking = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)
        top_features = ranking.head(self.top_k)
        # Rank position: 1 = most important
        rank_positions = pd.Series(
            range(1, len(top_features) + 1), index=top_features.index
        )
        self._rank_history[window_index] = rank_positions

        # Compute rho vs previous window
        prev_windows = sorted(w for w in self._rank_history if w < window_index)
        if not prev_windows:
            return None

        prev_idx = prev_windows[-1]
        rho = self._compute_rho(self._rank_history[prev_idx], rank_positions)
        self._rho_history[window_index] = rho
        return rho

    def _compute_rho(self, ranks_a: pd.Series, ranks_b: pd.Series) -> float:
        """Spearman correlation between two rank vectors on shared features."""
        common = ranks_a.index.intersection(ranks_b.index)
        if len(common) < 2:
            return 0.0
        rho, _ = stats.spearmanr(ranks_a[common], ranks_b[common])
        return float(rho) if not np.isnan(rho) else 0.0

    def get_rho(self, window_index: int) -> Optional[float]:
        return self._rho_history.get(window_index)

    def get_rank_vector(self, window_index: int) -> Optional[pd.Series]:
        return self._rank_history.get(window_index)

    def get_top_k_features(self, window_index: int) -> list[str]:
        rv = self.get_rank_vector(window_index)
        if rv is None:
            return []
        return rv.sort_values().index.tolist()

    def all_rhos(self) -> pd.Series:
        return pd.Series(self._rho_history).sort_index()
