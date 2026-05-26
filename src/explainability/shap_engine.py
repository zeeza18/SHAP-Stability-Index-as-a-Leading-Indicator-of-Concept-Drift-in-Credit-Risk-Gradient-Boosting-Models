"""SHAP computation engine for XGBoost and LightGBM models.

Uses TreeExplainer for exact, fast SHAP values on tree models.
Stratified sampling limits computation to 2000 rows per window.
"""

import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import shap

np.random.seed(42)
random.seed(42)

CHECKPOINTS_DIR = Path(__file__).resolve().parents[3] / "results" / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_SHAP_SAMPLES = 2000


class SHAPEngine:
    """Computes and caches SHAP values for tree models.

    Usage:
        engine = SHAPEngine(model, model_name="StaticXGBoost")
        shap_values = engine.compute(X_test, y_test, window_index=5)
        mean_abs = engine.mean_abs_shap(shap_values)
    """

    def __init__(self, model, model_name: str):
        """
        Args:
            model: Fitted XGBoost or LightGBM model (with .predict_proba).
            model_name: Used for checkpoint file naming.
        """
        self.model = model
        self.model_name = model_name
        self._explainer: Optional[shap.TreeExplainer] = None
        self._init_explainer()

    def _init_explainer(self) -> None:
        self._explainer = shap.TreeExplainer(self.model)

    def compute(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        window_index: int = -1,
        save: bool = True,
    ) -> np.ndarray:
        """Compute SHAP values for a window. Stratified sample up to 2000 rows.

        Args:
            X: Feature matrix.
            y: Labels (used for stratified sampling if provided).
            window_index: Window index for checkpoint naming.
            save: Whether to save SHAP values as .npy checkpoint.

        Returns:
            SHAP values array of shape (n_samples, n_features).
        """
        X_sample = self._stratified_sample(X, y)
        shap_values = self._explainer.shap_values(X_sample)

        # For binary classifiers, some SHAP returns list [neg, pos]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        if save and window_index >= 0:
            path = CHECKPOINTS_DIR / f"shap_values_{self.model_name}_window{window_index:03d}.npy"
            np.save(path, shap_values)

        return shap_values

    def _stratified_sample(
        self, X: pd.DataFrame, y: Optional[pd.Series]
    ) -> pd.DataFrame:
        """Sample at most MAX_SHAP_SAMPLES rows, stratified by class if y provided."""
        if len(X) <= MAX_SHAP_SAMPLES:
            return X

        rng = np.random.default_rng(42)
        if y is not None and y.nunique() == 2:
            # Stratified: proportional sample per class
            idx_pos = np.where(y.values == 1)[0]
            idx_neg = np.where(y.values == 0)[0]
            ratio = len(idx_pos) / len(y)
            n_pos = max(1, int(MAX_SHAP_SAMPLES * ratio))
            n_neg = MAX_SHAP_SAMPLES - n_pos

            chosen_pos = rng.choice(idx_pos, size=min(n_pos, len(idx_pos)), replace=False)
            chosen_neg = rng.choice(idx_neg, size=min(n_neg, len(idx_neg)), replace=False)
            chosen = np.concatenate([chosen_pos, chosen_neg])
            rng.shuffle(chosen)
        else:
            chosen = rng.choice(len(X), size=MAX_SHAP_SAMPLES, replace=False)

        return X.iloc[chosen].reset_index(drop=True)

    @staticmethod
    def mean_abs_shap(shap_values: np.ndarray) -> np.ndarray:
        """Compute mean absolute SHAP value per feature."""
        return np.abs(shap_values).mean(axis=0)

    @staticmethod
    def feature_ranking(
        shap_values: np.ndarray, feature_names: list[str]
    ) -> pd.Series:
        """Return features ranked by mean |SHAP| (rank 1 = most important)."""
        mean_abs = SHAPEngine.mean_abs_shap(shap_values)
        ranking = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)
        return ranking

    def load_cached(self, window_index: int) -> Optional[np.ndarray]:
        """Load previously saved SHAP values if they exist."""
        path = CHECKPOINTS_DIR / f"shap_values_{self.model_name}_window{window_index:03d}.npy"
        if path.exists():
            return np.load(path)
        return None
