"""Tests for ShapStabilityIndex and RankShiftTracker."""

import numpy as np
import pytest

from src.explainability.rank_shift_tracker import RankShiftTracker
from src.explainability.shap_stability_index import ShapStabilityIndex


def _make_shap_values(n_samples=100, n_features=20, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_samples, n_features))


def _feature_names(n=20):
    return [f"feature_{i}" for i in range(n)]


class TestRankShiftTracker:
    def test_first_window_returns_none(self):
        tracker = RankShiftTracker(top_k=10)
        shap = _make_shap_values()
        rho = tracker.update(0, shap, _feature_names())
        assert rho is None

    def test_second_window_returns_rho(self):
        tracker = RankShiftTracker(top_k=10)
        shap1 = _make_shap_values(seed=1)
        shap2 = _make_shap_values(seed=2)
        tracker.update(0, shap1, _feature_names())
        rho = tracker.update(1, shap2, _feature_names())
        assert rho is not None
        assert -1 <= rho <= 1

    def test_identical_shap_gives_high_rho(self):
        tracker = RankShiftTracker(top_k=10)
        shap = _make_shap_values(seed=42)
        tracker.update(0, shap, _feature_names())
        rho = tracker.update(1, shap, _feature_names())
        assert rho is not None
        assert rho > 0.95  # same data → nearly perfect correlation


class TestShapStabilityIndex:
    def test_ssi_nan_on_first_window(self):
        ssi = ShapStabilityIndex("TestModel", top_k=10, lookback=3)
        shap = _make_shap_values()
        val = ssi.update(0, shap, _feature_names(), auc=0.9)
        assert np.isnan(val)

    def test_ssi_computed_after_lookback(self):
        ssi = ShapStabilityIndex("TestModel", top_k=10, lookback=3)
        for i in range(5):
            shap = _make_shap_values(seed=i)
            ssi.update(i, shap, _feature_names(), auc=0.85 + i * 0.01)
        df = ssi.to_dataframe()
        assert len(df) == 5
        non_nan = df["ssi_value"].dropna()
        assert len(non_nan) >= 1

    def test_lead_time_computation(self):
        ssi = ShapStabilityIndex(
            "TestModel", top_k=10, lookback=3, ssi_threshold=0.8
        )
        rng = np.random.default_rng(42)
        # Simulate stable then degraded SSI/AUC
        for i in range(15):
            shap = _make_shap_values(seed=i if i < 10 else i * 10)
            auc = 0.92 if i < 12 else 0.80
            ssi.update(i, shap, _feature_names(), auc=auc)
        result = ssi.compute_lead_time(drift_windows=[12])
        assert "mean_lead_time" in result
        assert "events" in result

    def test_save_creates_file(self, tmp_path):
        ssi = ShapStabilityIndex("TestModel", top_k=5, lookback=2)
        for i in range(3):
            ssi.update(i, _make_shap_values(seed=i), _feature_names(), auc=0.9)
        path = ssi.save(output_dir=tmp_path)
        assert path.exists()
