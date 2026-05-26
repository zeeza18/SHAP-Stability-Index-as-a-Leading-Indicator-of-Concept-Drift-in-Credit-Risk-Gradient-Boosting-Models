"""Tests for SHAPEngine — uses a real small LightGBM model."""

import numpy as np
import pandas as pd
import pytest


def _make_model_and_data():
    from lightgbm import LGBMClassifier
    rng = np.random.default_rng(42)
    n, f = 300, 15
    X = pd.DataFrame(rng.standard_normal((n, f)), columns=[f"f{i}" for i in range(f)])
    y = pd.Series(rng.choice([0, 1], n, p=[0.9, 0.1]))
    model = LGBMClassifier(n_estimators=50, verbose=-1, random_state=42)
    model.fit(X, y)
    return model, X, y


def test_shap_engine_compute_shape(tmp_path):
    from src.explainability.shap_engine import SHAPEngine
    model, X, y = _make_model_and_data()
    engine = SHAPEngine(model, model_name="test_lgbm")
    shap_vals = engine.compute(X, y, window_index=-1, save=False)
    assert shap_vals.shape[1] == X.shape[1]


def test_mean_abs_shap_shape():
    from src.explainability.shap_engine import SHAPEngine
    model, X, y = _make_model_and_data()
    engine = SHAPEngine(model, model_name="test_lgbm")
    shap_vals = engine.compute(X, y, window_index=-1, save=False)
    mean_abs = engine.mean_abs_shap(shap_vals)
    assert len(mean_abs) == X.shape[1]


def test_feature_ranking_sorted():
    from src.explainability.shap_engine import SHAPEngine
    model, X, y = _make_model_and_data()
    engine = SHAPEngine(model, model_name="test_lgbm")
    shap_vals = engine.compute(X, y, window_index=-1, save=False)
    ranking = engine.feature_ranking(shap_vals, X.columns.tolist())
    assert ranking.is_monotonic_decreasing
