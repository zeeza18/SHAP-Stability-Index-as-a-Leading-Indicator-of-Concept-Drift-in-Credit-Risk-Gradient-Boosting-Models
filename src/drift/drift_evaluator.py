"""Drift evaluation: measures model behavior before and after drift events."""

import random

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

np.random.seed(42)
random.seed(42)


class DriftEvaluator:
    """Evaluates AUC recovery after drift events.

    Computes pre-drift AUC, post-drift AUC, and recovery time (windows).
    """

    def __init__(self, drift_window_indices: list[int], recovery_horizon: int = 5):
        self.drift_windows = set(drift_window_indices)
        self.recovery_horizon = recovery_horizon

    def evaluate_recovery(
        self, auc_series: pd.Series, window_indices: list[int]
    ) -> pd.DataFrame:
        """Compute per-drift-event pre/post AUC and recovery stats.

        Args:
            auc_series: AUC values indexed by window index.
            window_indices: All window indices evaluated.

        Returns:
            DataFrame with one row per drift event.
        """
        records = []
        auc_map = dict(zip(window_indices, auc_series))

        for d in sorted(self.drift_windows):
            pre_aucs = [auc_map[i] for i in window_indices if i < d and i in auc_map]
            post_aucs = [
                auc_map[i]
                for i in window_indices
                if d <= i <= d + self.recovery_horizon and i in auc_map
            ]
            pre_auc = np.mean(pre_aucs[-3:]) if len(pre_aucs) >= 1 else np.nan
            post_auc_min = np.min(post_aucs) if post_aucs else np.nan
            post_auc_recover = post_aucs[-1] if post_aucs else np.nan

            records.append({
                "drift_window": d,
                "pre_drift_auc_mean": pre_auc,
                "post_drift_auc_min": post_auc_min,
                "post_drift_auc_recovered": post_auc_recover,
                "auc_drop": pre_auc - post_auc_min if not np.isnan(pre_auc) else np.nan,
            })

        return pd.DataFrame(records)
