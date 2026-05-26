"""Static XGBoost model — trained once, never retrained."""

import random
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from .base_model import BaseModel, CHECKPOINTS_DIR
from ..utils.gpu import xgb_device_params

np.random.seed(42)
random.seed(42)


class StaticXGBoost(BaseModel):
    """XGBoost trained once on all training windows. Evaluates on each test window.

    Hyperparameters should be loaded from experiments/tuning/tuning_results/
    after Optuna tuning is complete.
    """

    DEFAULT_PARAMS = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "colsample_bylevel": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "gamma": 0.0,
        "scale_pos_weight": 1,
        "eval_metric": "auc",
        "use_label_encoder": False,
    }

    def __init__(self, params: Optional[dict] = None, random_state: int = 42):
        super().__init__(name="StaticXGBoost", random_state=random_state)
        gpu = xgb_device_params()
        self.params = {**self.DEFAULT_PARAMS, **gpu, **(params or {})}
        self.params["random_state"] = random_state
        self._model = xgb.XGBClassifier(**self.params)

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "StaticXGBoost":
        self._model.fit(X, y, verbose=False)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or CHECKPOINTS_DIR / f"{self.name}.joblib"
        joblib.dump(self._model, path)
        return path

    def load(self, path: Path) -> "StaticXGBoost":
        self._model = joblib.load(path)
        return self

    @classmethod
    def from_tuned_params(cls, params_path: Path, **kwargs) -> "StaticXGBoost":
        """Load best hyperparameters from Optuna tuning results."""
        import json
        params = json.loads(params_path.read_text())
        return cls(params=params, **kwargs)
