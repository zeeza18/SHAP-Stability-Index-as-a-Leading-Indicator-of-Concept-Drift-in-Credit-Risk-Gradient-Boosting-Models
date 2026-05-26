"""Tests for data loader (schema validation, not full file loading)."""

import pandas as pd
import pytest

from src.data.validator import DataValidator


def test_validator_leakage_check():
    validator = DataValidator(verbose=False)
    train = pd.DataFrame({"time": [1, 2, 3], "x": [0.1, 0.2, 0.3]})
    test = pd.DataFrame({"time": [4, 5, 6], "x": [0.4, 0.5, 0.6]})
    assert validator.check_no_future_leakage(train, test, "time") is True


def test_validator_leakage_fails_when_overlap():
    validator = DataValidator(verbose=False)
    train = pd.DataFrame({"time": [1, 2, 5], "x": [0.1, 0.2, 0.3]})
    test = pd.DataFrame({"time": [4, 5, 6], "x": [0.4, 0.5, 0.6]})
    assert validator.check_no_future_leakage(train, test, "time") is False


def test_validator_target_col():
    validator = DataValidator(verbose=False)
    df = pd.DataFrame({"isFraud": [0, 1, 0], "x": [1, 2, 3]})
    assert validator.check_target_col(df, "isFraud", valid_values=[0, 1]) is True
    assert validator.check_target_col(df, "missing_col") is False


def test_validator_assert_clean_raises():
    validator = DataValidator(verbose=False)
    df = pd.DataFrame({"time": [5, 1, 3]})
    validator.check_temporal_ordering(df, "time")  # not sorted → error
    with pytest.raises(ValueError):
        validator.assert_clean()
