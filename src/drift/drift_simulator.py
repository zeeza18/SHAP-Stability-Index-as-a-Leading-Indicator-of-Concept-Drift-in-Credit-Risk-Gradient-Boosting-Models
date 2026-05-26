"""Synthetic drift injection for ADWIN delta tuning.

Injects known drift at specific windows so we can measure
ADWIN's detection precision and recall (ground truth available).
"""

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

np.random.seed(42)
random.seed(42)


@dataclass
class InjectedDrift:
    """Describes a single synthetic drift event."""
    window_index: int
    drift_type: str         # 'feature_shift', 'label_shift', 'covariate'
    magnitude: float        # how large the shift is (0 to 1)


class DriftSimulator:
    """Inject synthetic drift into windowed data for controlled experiments.

    Usage:
        sim = DriftSimulator(seed=42)
        windows = sim.inject_feature_shift(windows, target_windows=[5, 10, 15])
        ground_truth = sim.ground_truth_drift_windows
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.injected_drifts: list[InjectedDrift] = []

    @property
    def ground_truth_drift_windows(self) -> list[int]:
        return [d.window_index for d in self.injected_drifts]

    def inject_feature_shift(
        self,
        windows: list[pd.DataFrame],
        target_windows: list[int],
        feature_cols: list[str] | None = None,
        magnitude: float = 0.5,
    ) -> list[pd.DataFrame]:
        """Shift the mean of numeric features in target windows.

        Args:
            windows: List of window DataFrames.
            target_windows: Window indices where drift is injected.
            feature_cols: Columns to shift. Defaults to all numeric columns.
            magnitude: Shift as a fraction of the column's std.

        Returns:
            Modified windows list.
        """
        result = [w.copy() for w in windows]
        for idx in target_windows:
            if idx >= len(result):
                continue
            w = result[idx]
            cols = feature_cols or w.select_dtypes(include=np.number).columns.tolist()
            for col in cols:
                shift = magnitude * w[col].std()
                result[idx][col] = w[col] + self._rng.choice([-1, 1]) * shift
            self.injected_drifts.append(
                InjectedDrift(window_index=idx, drift_type="feature_shift", magnitude=magnitude)
            )
        return result

    def inject_label_shift(
        self,
        windows: list[pd.DataFrame],
        target_col: str,
        target_windows: list[int],
        new_positive_rate: float = 0.3,
    ) -> list[pd.DataFrame]:
        """Artificially increase positive class rate in target windows."""
        result = [w.copy() for w in windows]
        for idx in target_windows:
            if idx >= len(result):
                continue
            w = result[idx].copy()
            n_pos_needed = int(new_positive_rate * len(w))
            current_pos = w[target_col].sum()
            neg_idx = w[w[target_col] == 0].index.tolist()
            n_to_flip = max(0, n_pos_needed - current_pos)
            if n_to_flip > 0 and neg_idx:
                flip_idx = self._rng.choice(
                    neg_idx, size=min(n_to_flip, len(neg_idx)), replace=False
                )
                w.loc[flip_idx, target_col] = 1
            result[idx] = w
            self.injected_drifts.append(
                InjectedDrift(window_index=idx, drift_type="label_shift", magnitude=new_positive_rate)
            )
        return result

    def compute_detection_metrics(
        self, detected_windows: list[int], tolerance: int = 1
    ) -> dict:
        """Compute precision, recall, and F1 of drift detection.

        Args:
            detected_windows: Windows where detector fired.
            tolerance: How many windows away counts as a correct detection.

        Returns:
            Dict with precision, recall, f1.
        """
        gt = set(self.ground_truth_drift_windows)
        det = set(detected_windows)

        tp = sum(
            any(abs(d - g) <= tolerance for g in gt) for d in det
        )
        fp = len(det) - tp
        fn = sum(
            not any(abs(g - d) <= tolerance for d in det) for g in gt
        )

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
