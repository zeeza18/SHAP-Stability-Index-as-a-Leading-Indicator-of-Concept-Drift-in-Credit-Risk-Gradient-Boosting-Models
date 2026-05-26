"""SHAP Stability Index (SSI) — the novel contribution of this paper.

Definition:
    At each window t, compute SHAP values and extract top-K feature rankings.
    rho_t = Spearman correlation between rank vectors at t and t-1.
    SSI(t) = mean(rho_{t-L+1}, ..., rho_t)  where L = lookback window.

This measures whether SHAP feature importance rankings are stable over time.
A drop in SSI before a drop in AUC means SSI is a LEADING INDICATOR of
model performance degradation — the key finding of this paper.

Reference equation (paper Section 3.7):
    SSI(t) = (1/L) * Σ_{l=1}^{L} ρ(R_{t-l+1}, R_{t-l})
    where ρ is Spearman correlation, R_t is the rank vector at window t.
"""

import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from .rank_shift_tracker import RankShiftTracker

np.random.seed(42)
random.seed(42)

TABLES_DIR = Path(__file__).resolve().parents[3] / "results" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


class ShapStabilityIndex:
    """Computes SSI across all windows and detects whether SSI leads AUC drops.

    Usage:
        ssi = ShapStabilityIndex(model_name="AdaptiveXGBoost", top_k=20, lookback=5)
        ssi.update(window_index=5, shap_values=arr, feature_names=names,
                   auc=0.92, drift_event=False, model_retrained=False)
        df = ssi.to_dataframe()
        lead_time = ssi.compute_lead_time(drift_windows=[10, 18])
    """

    def __init__(
        self,
        model_name: str,
        top_k: int = 20,
        lookback: int = 5,
        ssi_threshold: float = 0.7,
    ):
        """
        Args:
            model_name: Model identifier for output files.
            top_k: Number of top features tracked in rank vectors.
            lookback: Number of previous windows averaged in SSI (L).
            ssi_threshold: SSI drops below this are considered anomalous.
        """
        self.model_name = model_name
        self.top_k = top_k
        self.lookback = lookback
        self.ssi_threshold = ssi_threshold

        self._tracker = RankShiftTracker(top_k=top_k)
        self._records: list[dict] = []

    def update(
        self,
        window_index: int,
        shap_values: np.ndarray,
        feature_names: list[str],
        auc: float,
        drift_event: bool = False,
        model_retrained: bool = False,
    ) -> float:
        """Process one window: compute rho, SSI, and record all metrics.

        Returns:
            SSI value at this window (NaN if not enough history).
        """
        rho = self._tracker.update(window_index, shap_values, feature_names)
        ssi = self._compute_ssi(window_index)

        top_features = self._tracker.get_top_k_features(window_index)
        top1 = top_features[0] if len(top_features) > 0 else ""
        top2 = top_features[1] if len(top_features) > 1 else ""
        top3 = top_features[2] if len(top_features) > 2 else ""

        self._records.append({
            "window_index": window_index,
            "ssi_value": ssi,
            "rho_current": rho if rho is not None else np.nan,
            "top1_feature": top1,
            "top2_feature": top2,
            "top3_feature": top3,
            "auc_this_window": auc,
            "drift_event": int(drift_event),
            "model_was_retrained": int(model_retrained),
        })

        return ssi

    def _compute_ssi(self, current_window: int) -> float:
        """Compute SSI(t) = mean of last L Spearman correlations."""
        all_rhos = self._tracker.all_rhos()
        relevant = all_rhos[all_rhos.index <= current_window].tail(self.lookback)

        if len(relevant) == 0:
            return np.nan
        return float(relevant.mean())

    def to_dataframe(self) -> pd.DataFrame:
        """Return all window results as a DataFrame."""
        return pd.DataFrame(self._records)

    def save(self, output_dir: Optional[Path] = None) -> Path:
        """Save SSI results CSV to results/tables/."""
        out_dir = output_dir or TABLES_DIR
        path = out_dir / f"ssi_results_{self.model_name}.csv"
        self.to_dataframe().to_csv(path, index=False)
        print(f"SSI results saved to {path}")
        return path

    def compute_lead_time(self, drift_windows: list[int]) -> dict:
        """Compute how many windows SSI drops BEFORE AUC drops at each drift event.

        A positive lead_time means SSI warned before AUC degraded.

        Args:
            drift_windows: Window indices where ADWIN fired.

        Returns:
            Dict with mean_lead_time, std_lead_time, and per-event details.
        """
        df = self.to_dataframe()
        if df.empty:
            return {"mean_lead_time": np.nan, "std_lead_time": np.nan, "events": []}

        events = []
        for d in drift_windows:
            # Find first window where SSI drops below threshold before drift
            pre_drift = df[df["window_index"] < d].copy()
            if pre_drift.empty:
                continue

            ssi_drop_window = pre_drift[
                pre_drift["ssi_value"] < self.ssi_threshold
            ]["window_index"]
            ssi_drop_at = int(ssi_drop_window.min()) if not ssi_drop_window.empty else d

            # Find first window where AUC drops significantly after drift
            post_drift = df[df["window_index"] >= d].copy()
            if post_drift.empty:
                continue
            baseline_auc = pre_drift["auc_this_window"].tail(3).mean()
            auc_drop_windows = post_drift[
                post_drift["auc_this_window"] < baseline_auc - 0.02
            ]["window_index"]
            auc_drop_at = int(auc_drop_windows.min()) if not auc_drop_windows.empty else d

            lead_time = auc_drop_at - ssi_drop_at
            events.append({
                "drift_window": d,
                "ssi_drop_window": ssi_drop_at,
                "auc_drop_window": auc_drop_at,
                "lead_time_windows": lead_time,
            })

        if not events:
            return {"mean_lead_time": np.nan, "std_lead_time": np.nan, "events": []}

        lead_times = [e["lead_time_windows"] for e in events]
        return {
            "mean_lead_time": float(np.mean(lead_times)),
            "std_lead_time": float(np.std(lead_times)),
            "n_events": len(events),
            "events": events,
        }
