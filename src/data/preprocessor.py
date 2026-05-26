"""High-level preprocessing pipeline: load → validate → feature-engineer → split.

Orchestrates loader, validator, feature_engineer, and temporal_splitter
into a single pipeline call per dataset.
"""

import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .loader import load_ieee_cis, load_gmsc, load_ccfraud
from .validator import DataValidator
from .feature_engineer import FeatureEngineer
from .temporal_splitter import TemporalSplitter

np.random.seed(42)
random.seed(42)

TARGET_COLS = {
    "ieee_cis": "isFraud",
    "gmsc": "SeriousDlqin2yrs",
    "ccfraud": "Class",
}

TIME_COLS = {
    "ieee_cis": "TransactionDT",
    "gmsc": "pseudo_time_index",
    "ccfraud": "Time",
}


class DataPipeline:
    """End-to-end data pipeline for one dataset.

    Usage:
        pipeline = DataPipeline(dataset="ieee_cis")
        windows = pipeline.run()
        for win in windows:
            X_train, y_train = win["X_train"], win["y_train"]
            X_test, y_test   = win["X_test"],  win["y_test"]
    """

    def __init__(self, dataset: str, raw_dir: Optional[Path] = None):
        if dataset not in TARGET_COLS:
            raise ValueError(f"Unknown dataset: {dataset}")
        self.dataset = dataset
        self.target_col = TARGET_COLS[dataset]
        self.time_col = TIME_COLS[dataset]
        self.raw_dir = raw_dir

        self.feature_engineer = FeatureEngineer(dataset=dataset)
        self.splitter = TemporalSplitter(dataset=dataset)
        self.validator = DataValidator()

    def run(self) -> list[dict]:
        """Load, validate, split, and feature-engineer all windows.

        Returns:
            List of window dicts with X_train, y_train, X_test, y_test, window_index.
        """
        # 1. Load
        df = self._load()

        # 2. Validate raw data
        self.validator.check_target_col(df, self.target_col, valid_values=[0, 1])
        if self.time_col in df.columns:
            self.validator.check_temporal_ordering(
                df.sort_values(self.time_col), self.time_col
            )

        # 3. Split by time
        windows = self.splitter.split(df)

        # 4. Feature engineer each window (expanding window fit)
        result_windows = []
        for win in windows:
            train_df = win["train"]
            test_df = win["test"]

            y_train = train_df[self.target_col]
            y_test = test_df[self.target_col]

            X_train_raw = train_df.drop(columns=[self.target_col])
            X_test_raw = test_df.drop(columns=[self.target_col])

            # Fit on train, transform both
            X_train = self.feature_engineer.fit_transform(X_train_raw, y_train)
            X_test = self.feature_engineer.transform(X_test_raw)

            # Validate no leakage
            if self.time_col in train_df.columns and self.time_col in test_df.columns:
                self.validator.check_no_future_leakage(train_df, test_df, self.time_col)

            result_windows.append({
                "window_index": win["window_index"],
                "X_train": X_train,
                "y_train": y_train.reset_index(drop=True),
                "X_test": X_test,
                "y_test": y_test.reset_index(drop=True),
                "time_range": win["time_range"],
            })

        # Align all windows to window 0's X_train feature set so per-window
        # feature selection doesn't produce mismatched vocabularies.
        if result_windows:
            ref_cols = result_windows[0]["X_train"].columns.tolist()
            for win in result_windows:
                for key in ("X_train", "X_test"):
                    df = win[key]
                    for col in ref_cols:
                        if col not in df.columns:
                            df[col] = np.float32(-999)
                    win[key] = df[ref_cols]

        print(f"DataPipeline: {len(result_windows)} windows ready for {self.dataset}")
        return result_windows

    def _load(self) -> pd.DataFrame:
        kwargs = {} if self.raw_dir is None else {"raw_dir": self.raw_dir}
        if self.dataset == "ieee_cis":
            return load_ieee_cis(**kwargs)
        elif self.dataset == "gmsc":
            return load_gmsc(**kwargs)
        else:
            return load_ccfraud(**kwargs)
