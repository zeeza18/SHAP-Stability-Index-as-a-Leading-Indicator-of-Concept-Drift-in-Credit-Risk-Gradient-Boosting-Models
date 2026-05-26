"""Adaptive CatBoost model — retrains when ADWIN detects drift."""

import random
from collections import deque
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from .base_model import BaseModel, CHECKPOINTS_DIR
from ..drift.adwin_detector import ADWINDetector
from ..utils.gpu import catboost_device_params

np.random.seed(42)
random.seed(42)


class AdaptiveCatBoost(BaseModel):
    """CatBoost with ADWIN-triggered adaptive retraining."""

    DEFAULT_PARAMS = {
        "iterations": 500,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3.0,
        "random_strength": 1.0,
        "bagging_temperature": 1.0,
        "border_count": 128,
        "verbose": 0,
    }

    def __init__(
        self,
        params: Optional[dict] = None,
        adwin_delta: float = 0.002,
        buffer_size: int = 5,
        random_state: int = 42,
    ):
        super().__init__(name="AdaptiveCatBoost", random_state=random_state)
        gpu = catboost_device_params()
        self.params = {**self.DEFAULT_PARAMS, **gpu, **(params or {})}
        self.params["random_seed"] = random_state
        self.adwin_delta = adwin_delta
        self.buffer_size = buffer_size

        self._model = CatBoostClassifier(**self.params)
        self._detector = ADWINDetector(delta=adwin_delta)
        self._buffer: deque = deque(maxlen=buffer_size)
        self._retraining_log: list[dict] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "AdaptiveCatBoost":
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]

    def update_window(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        window_index: int,
        imbalance_handler=None,
    ) -> dict:
        """Predict, detect drift, optionally retrain."""
        proba = self.predict_proba(X_test)
        preds = (proba >= 0.5).astype(int)
        y_vals = y_test.values
        error_rate = (preds != y_vals).mean()

        drift_detected = False
        for j in range(len(y_vals)):
            if self._detector.update(float(preds[j] != y_vals[j]), window_index=window_index):
                drift_detected = True

        retrained = False
        if drift_detected and len(self._buffer) > 0:
            X_retrain = pd.concat([b[0] for b in self._buffer], ignore_index=True)
            y_retrain = pd.concat([b[1] for b in self._buffer], ignore_index=True)

            if imbalance_handler is not None:
                X_retrain, y_retrain = imbalance_handler.transform(
                    X_retrain, y_retrain, is_train=True
                )

            self._model = CatBoostClassifier(**self.params)
            self._model.fit(X_retrain, y_retrain)
            retrained = True

            self._retraining_log.append({
                "window_index": window_index,
                "n_samples_used": len(y_retrain),
                "drift_delta": self.adwin_delta,
                "error_rate_at_drift": error_rate,
            })
            print(
                f"[AdaptiveCatBoost] Drift @ window {window_index}. "
                f"Retrained on {len(y_retrain)} samples."
            )

        self._buffer.append((X_test, y_test))

        metrics = self.evaluate(X_test, y_test)
        metrics["window_index"] = window_index
        metrics["drift_detected"] = drift_detected
        metrics["model_retrained"] = retrained
        metrics["error_rate"] = error_rate
        return metrics

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or CHECKPOINTS_DIR / f"{self.name}.joblib"
        joblib.dump(
            {
                "model": self._model,
                "detector": self._detector,
                "buffer": list(self._buffer),
                "buffer_size": self.buffer_size,
                "retraining_log": self._retraining_log,
                "params": self.params,
                "adwin_delta": self.adwin_delta,
            },
            path,
        )
        return path

    def load(self, path: Path) -> "AdaptiveCatBoost":
        obj = joblib.load(path)
        self._model = obj["model"]
        self._detector = obj["detector"]
        buf = obj.get("buffer", [])
        self._buffer = deque(buf, maxlen=obj.get("buffer_size", self.buffer_size))
        self._retraining_log = obj.get("retraining_log", [])
        return self

    @property
    def retraining_log(self) -> list[dict]:
        return self._retraining_log

    @classmethod
    def from_tuned_params(
        cls, params_path: Path, adwin_params_path: Optional[Path] = None, **kwargs
    ) -> "AdaptiveCatBoost":
        import json
        params = json.loads(params_path.read_text())
        adwin_delta = 0.002
        if adwin_params_path and adwin_params_path.exists():
            adwin_params = json.loads(adwin_params_path.read_text())
            adwin_delta = adwin_params.get("delta", adwin_delta)
        return cls(params=params, adwin_delta=adwin_delta, **kwargs)
