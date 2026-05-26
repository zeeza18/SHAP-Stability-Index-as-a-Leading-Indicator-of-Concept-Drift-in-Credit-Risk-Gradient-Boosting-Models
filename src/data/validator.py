"""Data validation utilities for processed datasets and window splits.

Checks for leakage, schema integrity, class balance, and temporal ordering.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(42)
random.seed(42)


class DataValidator:
    """Validates datasets for leakage, ordering, and schema integrity."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._errors: list[str] = []
        self._warnings: list[str] = []

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def check_no_future_leakage(
        self, train: pd.DataFrame, test: pd.DataFrame, time_col: str
    ) -> bool:
        """Verify that max(train[time_col]) < min(test[time_col])."""
        train_max = train[time_col].max()
        test_min = test[time_col].min()
        if train_max >= test_min:
            msg = (
                f"LEAKAGE: train max time ({train_max}) >= test min time ({test_min})"
            )
            self._errors.append(msg)
            self._log(f"[FAIL] {msg}")
            return False
        self._log(
            f"[OK] No leakage: train ends at {train_max}, test starts at {test_min}"
        )
        return True

    def check_class_balance(
        self, df: pd.DataFrame, target_col: str, name: str = ""
    ) -> dict:
        """Report class distribution."""
        counts = df[target_col].value_counts()
        ratio = counts.min() / counts.max()
        self._log(f"[INFO] {name} class balance: {dict(counts)} | minority ratio: {ratio:.4f}")
        return {"counts": dict(counts), "minority_ratio": ratio}

    def check_no_nulls_in_features(
        self, df: pd.DataFrame, exclude_cols: list | None = None
    ) -> bool:
        """Check that no feature columns have unexpected nulls after preprocessing."""
        exclude_cols = exclude_cols or []
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        null_counts = df[feature_cols].isnull().sum()
        null_cols = null_counts[null_counts > 0]
        if not null_cols.empty:
            msg = f"NULLs found in: {null_cols.to_dict()}"
            self._warnings.append(msg)
            self._log(f"[WARN] {msg}")
            return False
        self._log("[OK] No nulls in feature columns")
        return True

    def check_temporal_ordering(
        self, df: pd.DataFrame, time_col: str
    ) -> bool:
        """Verify that the DataFrame is sorted by time_col ascending."""
        is_sorted = df[time_col].is_monotonic_increasing
        if not is_sorted:
            msg = f"DataFrame is NOT sorted by {time_col}"
            self._errors.append(msg)
            self._log(f"[FAIL] {msg}")
            return False
        self._log(f"[OK] Temporal ordering verified on {time_col}")
        return True

    def check_target_col(
        self, df: pd.DataFrame, target_col: str, valid_values: list | None = None
    ) -> bool:
        """Check target column exists and contains only expected values."""
        if target_col not in df.columns:
            self._errors.append(f"Target column '{target_col}' not found")
            self._log(f"[FAIL] Target column '{target_col}' not found")
            return False
        if valid_values is not None:
            unexpected = set(df[target_col].unique()) - set(valid_values)
            if unexpected:
                msg = f"Unexpected values in {target_col}: {unexpected}"
                self._errors.append(msg)
                self._log(f"[FAIL] {msg}")
                return False
        self._log(f"[OK] Target column '{target_col}' is valid")
        return True

    def summary(self) -> dict:
        """Return a summary of all errors and warnings."""
        return {
            "errors": self._errors,
            "warnings": self._warnings,
            "passed": len(self._errors) == 0,
        }

    def assert_clean(self) -> None:
        """Raise ValueError if any errors were recorded."""
        s = self.summary()
        if not s["passed"]:
            raise ValueError(
                f"Validation failed with {len(s['errors'])} error(s): {s['errors']}"
            )
