"""Shared fixtures for all tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def small_ieee_df():
    """Minimal IEEE-CIS-like DataFrame for fast unit tests."""
    rng = np.random.default_rng(42)
    n = 500
    return pd.DataFrame({
        "TransactionID": range(n),
        "TransactionDT": np.arange(n) * 3600,
        "TransactionAmt": rng.uniform(10, 500, n),
        "ProductCD": rng.choice(["W", "H", "C", "S", "R"], n),
        "card1": rng.integers(1000, 9999, n).astype(str),
        "card4": rng.choice(["visa", "mastercard", "discover"], n),
        "card6": rng.choice(["debit", "credit"], n),
        "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", "company.com"], n),
        "R_emaildomain": rng.choice(["gmail.com", "yahoo.com", "other.com"], n),
        "D1": rng.uniform(0, 300, n),
        "D2": rng.uniform(0, 100, n),
        "V1": rng.standard_normal(n),
        "V2": rng.standard_normal(n),
        "isFraud": rng.choice([0, 1], n, p=[0.965, 0.035]),
    })


@pytest.fixture
def small_gmsc_df():
    """Minimal GMSC-like DataFrame."""
    rng = np.random.default_rng(42)
    n = 300
    return pd.DataFrame({
        "SeriousDlqin2yrs": rng.choice([0, 1], n, p=[0.933, 0.067]),
        "RevolvingUtilizationOfUnsecuredLines": rng.uniform(0, 1, n),
        "age": rng.integers(20, 80, n),
        "NumberOfTime30-59DaysPastDueNotWorse": rng.integers(0, 5, n),
        "DebtRatio": rng.uniform(0, 1, n),
        "MonthlyIncome": rng.uniform(1000, 10000, n),
        "NumberOfOpenCreditLinesAndLoans": rng.integers(0, 20, n),
        "NumberOfTimes90DaysLate": rng.integers(0, 3, n),
        "NumberRealEstateLoansOrLines": rng.integers(0, 5, n),
        "NumberOfTime60-89DaysPastDueNotWorse": rng.integers(0, 3, n),
        "NumberOfDependents": rng.integers(0, 5, n),
    })


@pytest.fixture
def small_ccfraud_df():
    """Minimal Credit Card Fraud-like DataFrame."""
    rng = np.random.default_rng(42)
    n = 400
    df = pd.DataFrame(
        {f"V{i}": rng.standard_normal(n) for i in range(1, 29)}
    )
    df["Time"] = np.arange(n) * 60
    df["Amount"] = rng.uniform(1, 500, n)
    df["Class"] = rng.choice([0, 1], n, p=[0.998, 0.002])
    return df
