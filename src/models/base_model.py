"""Abstract base class that all models inherit from.

All models must implement: fit, predict, predict_proba, evaluate, save, load.
"""

import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

np.random.seed(42)
random.seed(42)

CHECKPOINTS_DIR = Path(__file__).resolve().parents[3] / "results" / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


class BaseModel(ABC):
    """Abstract base for all credit risk models."""

    def __init__(self, name: str, random_state: int = 42):
        self.name = name
        self.random_state = random_state
        self._model = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "BaseModel":
        """Fit the model on training data."""
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return binary predictions."""
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability estimates for the positive class."""
        ...

    def evaluate(
        self, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5
    ) -> dict:
        """Compute standard evaluation metrics.

        Returns:
            Dict with roc_auc, f1, average_precision, and n_samples.
        """
        proba = self.predict_proba(X)
        preds = (proba >= threshold).astype(int)
        return {
            "roc_auc": roc_auc_score(y, proba),
            "f1": f1_score(y, preds, zero_division=0),
            "average_precision": average_precision_score(y, proba),
            "n_samples": len(y),
            "n_positives": int(y.sum()),
        }

    @abstractmethod
    def save(self, path: Optional[Path] = None) -> Path:
        """Persist model to disk. Returns the save path."""
        ...

    @abstractmethod
    def load(self, path: Path) -> "BaseModel":
        """Load model from disk."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
