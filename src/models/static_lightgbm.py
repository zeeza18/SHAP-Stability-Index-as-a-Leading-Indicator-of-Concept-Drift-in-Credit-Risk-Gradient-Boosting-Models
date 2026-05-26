"""Static LightGBM model — trained once, never retrained."""

import random
from pathlib import Path
from typing import Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from .base_model import BaseModel, CHECKPOINTS_DIR
from ..utils.gpu import lgb_device_params

np.random.seed(42)
random.seed(42)


class StaticLightGBM(BaseModel):
    """LightGBM trained once on all training windows. Evaluates on each test window."""

    DEFAULT_PARAMS = {
        "n_estimators": 500,
        "num_leaves": 63,
        "max_depth": -1,
        "learning_rate": 0.05,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 5,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_split_gain": 0.0,
        "is_unbalance": False,
        "verbose": -1,
        "n_jobs": -1,
    }

    def __init__(self, params: Optional[dict] = None, random_state: int = 42):
        super().__init__(name="StaticLightGBM", random_state=random_state)
        gpu = lgb_device_params()
        self.params = {**self.DEFAULT_PARAMS, **gpu, **(params or {})}
        self.params["random_state"] = random_state
        self._model = lgb.LGBMClassifier(**self.params)

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "StaticLightGBM":
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or CHECKPOINTS_DIR / f"{self.name}.joblib"
        joblib.dump(self._model, path)
        return path

    def load(self, path: Path) -> "StaticLightGBM":
        self._model = joblib.load(path)
        return self

    @classmethod
    def from_tuned_params(cls, params_path: Path, **kwargs) -> "StaticLightGBM":
        import json
        params = json.loads(params_path.read_text())
        return cls(params=params, **kwargs)
