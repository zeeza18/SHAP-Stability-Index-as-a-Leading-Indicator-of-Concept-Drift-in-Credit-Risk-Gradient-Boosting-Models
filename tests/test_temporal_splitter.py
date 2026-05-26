"""Tests for TemporalSplitter."""

import numpy as np
import pandas as pd
import pytest

from src.data.temporal_splitter import TemporalSplitter


def _make_df_with_time(n=1000, time_col="TransactionDT"):
    return pd.DataFrame({
        time_col: np.arange(n) * 3600,
        "feature": np.random.randn(n),
        "label": np.random.choice([0, 1], n, p=[0.97, 0.03]),
    })


def test_ieee_split_returns_windows():
    df = _make_df_with_time(n=2000)
    splitter = TemporalSplitter(dataset="ieee_cis")
    windows = splitter.split(df)
    assert len(windows) > 0
    for win in windows:
        assert "train" in win and "test" in win
        assert win["window_index"] >= 0


def test_no_leakage_in_windows():
    """Test that train max time < test min time for all windows."""
    df = _make_df_with_time(n=2000)
    splitter = TemporalSplitter(dataset="ieee_cis")
    windows = splitter.split(df)
    for win in windows:
        if len(win["train"]) == 0 or len(win["test"]) == 0:
            continue
        train_max = win["train"]["TransactionDT"].max()
        test_min = win["test"]["TransactionDT"].min()
        assert train_max < test_min, f"Leakage in window {win['window_index']}"


def test_gmsc_split():
    df = pd.DataFrame({
        "pseudo_time_index": np.arange(150),
        "feature": np.random.randn(150),
        "label": np.random.choice([0, 1], 150),
    })
    splitter = TemporalSplitter(dataset="gmsc")
    windows = splitter.split(df)
    assert len(windows) > 0


def test_ccfraud_split():
    n = 5000
    df = pd.DataFrame({
        "Time": np.arange(n) * 30,
        "Amount": np.random.uniform(1, 500, n),
        "Class": np.random.choice([0, 1], n, p=[0.998, 0.002]),
    })
    splitter = TemporalSplitter(dataset="ccfraud")
    windows = splitter.split(df)
    assert len(windows) > 0
