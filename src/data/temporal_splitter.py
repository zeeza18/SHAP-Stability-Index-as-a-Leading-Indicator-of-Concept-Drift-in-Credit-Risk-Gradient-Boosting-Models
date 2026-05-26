"""Temporal windowing/splitting strategy for all three datasets.

Implements sliding-window splits that strictly respect time order.
No data from future windows ever bleeds into earlier windows.
"""

import random
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

np.random.seed(42)
random.seed(42)

SPLITS_DIR = Path(__file__).resolve().parents[3] / "data" / "splits"


class TemporalSplitter:
    """Create time-ordered window splits for streaming evaluation.

    Usage:
        splitter = TemporalSplitter(dataset="ieee_cis")
        windows = splitter.split(df)
        for win in windows:
            train, test = win["train"], win["test"]
    """

    DATASET_CONFIGS = {
        "ieee_cis": {
            "time_col": "TransactionDT",
            "window_size_days": 14,
            "stride_days": 7,
            "min_train_windows": 6,
        },
        "gmsc": {
            "time_col": "pseudo_time_index",
            "n_windows": 20,
            "n_train_windows": 8,
        },
        "ccfraud": {
            "time_col": "Time",
            "window_size_seconds": 4 * 3600,
            "stride_seconds": 2 * 3600,
            "min_train_windows": 4,
        },
    }

    def __init__(self, dataset: str):
        if dataset not in self.DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset}")
        self.dataset = dataset
        self.config = self.DATASET_CONFIGS[dataset]
        self._windows: list[dict] = []

    def split(self, df: pd.DataFrame) -> list[dict]:
        """Split df into time-ordered windows.

        Returns:
            List of dicts with keys: window_index, train, test, time_range.
            Each dict's train = all data before this window,
            test = data in this window.
        """
        if self.dataset == "ieee_cis":
            self._windows = self._split_ieee_cis(df)
        elif self.dataset == "gmsc":
            self._windows = self._split_gmsc(df)
        else:
            self._windows = self._split_ccfraud(df)
        return self._windows

    def save_windows(self, base_dir: Path | None = None) -> None:
        """Save each window's test split to parquet."""
        if not self._windows:
            raise RuntimeError("Call split() first")
        base_dir = base_dir or (SPLITS_DIR / f"{self.dataset}_windows")
        base_dir.mkdir(parents=True, exist_ok=True)
        for win in self._windows:
            i = win["window_index"]
            path = base_dir / f"window_{i:03d}.parquet"
            win["test"].to_parquet(path, index=False)
        print(f"Saved {len(self._windows)} window splits to {base_dir}")

    def iter_expanding(self) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
        """Yield (train, test) pairs with expanding training window."""
        for win in self._windows:
            yield win["train"], win["test"]

    # ------------------------------------------------------------------ #
    #  IEEE-CIS: sliding window by calendar days                          #
    # ------------------------------------------------------------------ #

    def _split_ieee_cis(self, df: pd.DataFrame) -> list[dict]:
        cfg = self.config
        time_col = cfg["time_col"]
        window_sec = cfg["window_size_days"] * 86400
        stride_sec = cfg["stride_days"] * 86400
        min_train = cfg["min_train_windows"]

        df = df.sort_values(time_col).reset_index(drop=True)
        t_min, t_max = df[time_col].min(), df[time_col].max()

        windows_raw = []
        start = t_min
        while start + window_sec <= t_max:
            end = start + window_sec
            mask = (df[time_col] >= start) & (df[time_col] < end)
            windows_raw.append((start, end, df[mask]))
            start += stride_sec

        print(f"IEEE-CIS: {len(windows_raw)} raw windows of {cfg['window_size_days']} days")

        results = []
        for i, (start, end, test_df) in enumerate(windows_raw):
            if i < min_train:
                continue
            train_df = df[df[time_col] < start]
            results.append({
                "window_index": i,
                "train": train_df.reset_index(drop=True),
                "test": test_df.reset_index(drop=True),
                "time_range": (start, end),
            })

        print(f"IEEE-CIS: {len(results)} usable test windows (min_train={min_train})")
        return results

    # ------------------------------------------------------------------ #
    #  GMSC: equal-size chunks by pseudo_time_index                       #
    # ------------------------------------------------------------------ #

    def _split_gmsc(self, df: pd.DataFrame) -> list[dict]:
        cfg = self.config
        time_col = cfg["time_col"]
        n_windows = cfg["n_windows"]
        n_train = cfg["n_train_windows"]

        # GMSC has no real time column — pseudo_time_index is the row order.
        # If FeatureEngineer hasn't run yet, just use the integer index.
        if time_col not in df.columns:
            df = df.reset_index(drop=True)
            df = df.copy()
            df[time_col] = df.index
        df = df.sort_values(time_col).reset_index(drop=True)
        chunks = np.array_split(df, n_windows)

        results = []
        for i, chunk in enumerate(chunks):
            if i < n_train:
                continue
            train_df = pd.concat(chunks[:i], ignore_index=True)
            results.append({
                "window_index": i,
                "train": train_df,
                "test": chunk.reset_index(drop=True),
                "time_range": (chunk[time_col].min(), chunk[time_col].max()),
            })

        print(
            f"GMSC: {len(results)} test windows out of {n_windows} total "
            f"({n_train} used for training)"
        )
        return results

    # ------------------------------------------------------------------ #
    #  Credit Card Fraud: sliding window by seconds                       #
    # ------------------------------------------------------------------ #

    def _split_ccfraud(self, df: pd.DataFrame) -> list[dict]:
        cfg = self.config
        time_col = cfg["time_col"]
        window_sec = cfg["window_size_seconds"]
        stride_sec = cfg["stride_seconds"]
        min_train = cfg["min_train_windows"]

        df = df.sort_values(time_col).reset_index(drop=True)
        t_min, t_max = df[time_col].min(), df[time_col].max()

        windows_raw = []
        start = t_min
        while start + window_sec <= t_max:
            end = start + window_sec
            mask = (df[time_col] >= start) & (df[time_col] < end)
            windows_raw.append((start, end, df[mask]))
            start += stride_sec

        print(f"CC Fraud: {len(windows_raw)} raw windows of {window_sec/3600:.0f}h")

        results = []
        for i, (start, end, test_df) in enumerate(windows_raw):
            if i < min_train:
                continue
            train_df = df[df[time_col] < start]
            results.append({
                "window_index": i,
                "train": train_df.reset_index(drop=True),
                "test": test_df.reset_index(drop=True),
                "time_range": (start, end),
            })

        print(f"CC Fraud: {len(results)} usable test windows")
        return results
