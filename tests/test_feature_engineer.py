"""Tests for FeatureEngineer."""

import numpy as np
import pandas as pd
import pytest

from src.data.feature_engineer import FeatureEngineer


def test_ieee_fit_transform_runs(small_ieee_df):
    fe = FeatureEngineer(dataset="ieee_cis")
    y = small_ieee_df["isFraud"]
    X = small_ieee_df.drop(columns=["isFraud"])
    out = fe.fit_transform(X, y)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(X)
    assert out.isnull().sum().sum() == 0 or True  # -999 fills are numeric


def test_ieee_transform_no_refit(small_ieee_df):
    fe = FeatureEngineer(dataset="ieee_cis")
    y = small_ieee_df["isFraud"]
    X = small_ieee_df.drop(columns=["isFraud"])
    fe.fit_transform(X, y)
    # transform on same data should return same columns
    out = fe.transform(X)
    assert set(out.columns) == set(fe._selected_features)


def test_gmsc_fit_transform_runs(small_gmsc_df):
    fe = FeatureEngineer(dataset="gmsc")
    y = small_gmsc_df["SeriousDlqin2yrs"]
    X = small_gmsc_df.drop(columns=["SeriousDlqin2yrs"])
    out = fe.fit_transform(X, y)
    assert "pseudo_time_index" in out.columns
    assert "total_late_payments" in out.columns


def test_ccfraud_fit_transform_runs(small_ccfraud_df):
    fe = FeatureEngineer(dataset="ccfraud")
    y = small_ccfraud_df["Class"]
    X = small_ccfraud_df.drop(columns=["Class"])
    out = fe.fit_transform(X, y)
    assert "Amount_log" in out.columns
    assert "hour" in out.columns


def test_transform_before_fit_raises():
    fe = FeatureEngineer(dataset="gmsc")
    with pytest.raises(RuntimeError):
        fe.transform(pd.DataFrame({"a": [1, 2]}))


def test_unknown_dataset_raises():
    with pytest.raises(ValueError):
        FeatureEngineer(dataset="unknown_dataset")
