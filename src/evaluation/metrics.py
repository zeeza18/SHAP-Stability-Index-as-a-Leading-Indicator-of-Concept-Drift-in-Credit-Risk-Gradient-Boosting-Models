"""Per-window evaluation metrics."""

import random
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    average_precision_score,
    brier_score_loss,
    roc_curve,
)

np.random.seed(42)
random.seed(42)


def compute_ks_statistic(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """KS statistic = max(TPR - FPR) across all thresholds."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return float(np.max(tpr - fpr))


def compute_optimal_threshold_f1(
    y_true: np.ndarray, y_proba: np.ndarray, n_thresholds: int = 100
) -> tuple[float, float]:
    """Find threshold that maximizes F1. Returns (threshold, f1)."""
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_f1, best_thresh = 0.0, 0.5
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        f = f1_score(y_true, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_thresh = f, t
    return best_thresh, best_f1


def evaluate_window(
    y_true: np.ndarray, y_proba: np.ndarray, window_index: int = -1
) -> dict:
    """Compute all metrics for a single window.

    Returns:
        Dict with roc_auc, f1_05, f1_optimal, ks_statistic,
        average_precision, brier_score, n_samples, n_positives.
    """
    y_pred_05 = (y_proba >= 0.5).astype(int)
    opt_thresh, f1_opt = compute_optimal_threshold_f1(y_true, y_proba)

    return {
        "window_index": window_index,
        "roc_auc": roc_auc_score(y_true, y_proba),
        "f1_at_05": f1_score(y_true, y_pred_05, zero_division=0),
        "f1_optimal": f1_opt,
        "optimal_threshold": opt_thresh,
        "ks_statistic": compute_ks_statistic(y_true, y_proba),
        "average_precision": average_precision_score(y_true, y_proba),
        "brier_score": brier_score_loss(y_true, y_proba),
        "n_samples": len(y_true),
        "n_positives": int(y_true.sum()),
    }
