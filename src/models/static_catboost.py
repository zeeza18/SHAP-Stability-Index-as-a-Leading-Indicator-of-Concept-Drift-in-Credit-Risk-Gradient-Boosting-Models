"""Static CatBoost model — trained once, never retrained."""

import random
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from .base_model import BaseModel, CHECKPOINTS_DIR
from ..utils.gpu import catboost_device_params

np.random.seed(42)
random.seed(42)


class StaticCatBoost(BaseModel):
    """CatBoost trained once on all training windows. Evaluates on each test window."""

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

    def __init__(self, params: Optional[dict] = None, random_state: int = 42):
        super().__init__(name="StaticCatBoost", random_state=random_state)
        gpu = catboost_device_params()
        self.params = {**self.DEFAULT_PARAMS, **gpu, **(params or {})}
        self.params["random_seed"] = random_state
        self._model = CatBoostClassifier(**self.params)

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "StaticCatBoost":
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

    def load(self, path: Path) -> "StaticCatBoost":
        self._model = joblib.load(path)
        return self

    @classmethod
    def from_tuned_params(cls, params_path: Path, **kwargs) -> "StaticCatBoost":
        import json
        params = json.loads(params_path.read_text())
        return cls(params=params, **kwargs)
