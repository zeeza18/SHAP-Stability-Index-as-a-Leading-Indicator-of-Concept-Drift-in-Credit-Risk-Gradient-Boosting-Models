"""Imbalance handling strategy per dataset.

CRITICAL: ImbalanceHandler.fit() is called ONLY on training data.
ImbalanceHandler.transform() is called on training data ONLY.
Test data NEVER passes through transform().
"""

import random
from typing import Optional

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight

np.random.seed(42)
random.seed(42)

# Resampling target ratios per dataset
_TARGET_RATIOS = {
    "ieee_cis": 0.1,    # 10% minority after SMOTE
    "ccfraud": 0.05,    # 5% minority — extreme imbalance
    "gmsc": None,       # class weights only, no oversampling
}


class ImbalanceHandler:
    """Handles class imbalance through SMOTE, SMOTETomek, or class weights.

    Usage:
        handler = ImbalanceHandler(dataset="ieee_cis")
        handler.fit(X_train, y_train)
        X_res, y_res = handler.transform(X_train, y_train, is_train=True)
        weights = handler.get_class_weights()
    """

    def __init__(self, dataset: str, k_neighbors: int = 5, random_state: int = 42):
        if dataset not in _TARGET_RATIOS:
            raise ValueError(f"Unknown dataset: {dataset}")
        self.dataset = dataset
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self._class_weights: Optional[dict] = None
        self._sampler = None
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ImbalanceHandler":
        """Fit the sampler on training data and compute class weights."""
        classes = np.array([0, 1])
        weights = compute_class_weight("balanced", classes=classes, y=y)
        self._class_weights = {0: weights[0], 1: weights[1]}

        target_ratio = _TARGET_RATIOS[self.dataset]

        if self.dataset == "ieee_cis":
            n_pos = y.sum()
            n_neg = len(y) - n_pos
            # sampling_strategy = desired n_minority / n_majority
            target_n_pos = int(n_neg * target_ratio)
            if target_n_pos <= n_pos:
                # Already at or above target — no oversampling needed
                self._sampler = None
            else:
                sampling_strategy = target_ratio / (1 - target_ratio)
                self._sampler = SMOTE(
                    sampling_strategy=sampling_strategy,
                    k_neighbors=self.k_neighbors,
                    random_state=self.random_state,
                )
                self._sampler.fit_resample(X, y)  # warm fit

        elif self.dataset == "ccfraud":
            n_pos = y.sum()
            n_neg = len(y) - n_pos
            target_n_pos = int(n_neg * target_ratio)
            if target_n_pos <= n_pos:
                self._sampler = None
            else:
                sampling_strategy = target_ratio / (1 - target_ratio)
                self._sampler = SMOTETomek(
                    sampling_strategy=sampling_strategy,
                    random_state=self.random_state,
                )
                self._sampler.fit_resample(X, y)

        else:  # gmsc — class weights only
            self._sampler = None

        self._fitted = True
        return self

    def transform(
        self, X: pd.DataFrame, y: pd.Series, is_train: bool = True
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Apply oversampling to training data.

        Args:
            X: Feature matrix.
            y: Target series.
            is_train: Must be True. Raises ValueError if False — test data
                      must never be resampled.

        Returns:
            Resampled (X, y) for training, unchanged if no sampler is set.
        """
        if not is_train:
            raise ValueError(
                "ImbalanceHandler.transform() must only be called with is_train=True. "
                "Never resample test or validation data."
            )
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")

        if self._sampler is None:
            return X, y

        X_res, y_res = self._sampler.fit_resample(X, y)
        X_res = pd.DataFrame(X_res, columns=X.columns)
        y_res = pd.Series(y_res, name=y.name)
        print(
            f"Resampling ({self.dataset}): {len(y)} → {len(y_res)} samples "
            f"({y_res.sum()} positives, {(y_res==0).sum()} negatives)"
        )
        return X_res, y_res

    def get_class_weights(self) -> Optional[dict]:
        """Return class weights dict for use in model constructors."""
        return self._class_weights

    def get_scale_pos_weight(self) -> float:
        """Return neg/pos ratio for XGBoost scale_pos_weight parameter."""
        if self._class_weights is None:
            raise RuntimeError("Call fit() first")
        # scale_pos_weight = n_neg / n_pos
        return self._class_weights[1] / self._class_weights[0]
