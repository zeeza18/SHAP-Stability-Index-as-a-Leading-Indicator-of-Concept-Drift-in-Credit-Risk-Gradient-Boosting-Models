"""Orchestrator: run Optuna tuning for XGBoost, LightGBM, and ADWIN.

Uses window cache (builds once, reuses every time).
Optuna studies persist in SQLite — interrupting and restarting continues
from where it left off automatically.
"""

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

np.random.seed(42)
random.seed(42)

TUNING_RESULTS = ROOT / "experiments" / "tuning" / "tuning_results"


def run_all_tuning(
    dataset: str = "ieee_cis",
    n_trials: int = 150,
    fresh_cache: bool = False,
) -> None:
    from src.data.window_cache import load_or_build
    from src.models.static_xgboost import StaticXGBoost
    from src.drift.drift_simulator import DriftSimulator
    from src.utils.gpu import print_gpu_status
    from experiments.tuning.tune_xgboost import run_tuning as tune_xgb
    from experiments.tuning.tune_lightgbm import run_tuning as tune_lgb
    from experiments.tuning.tune_catboost import run_tuning as tune_cb
    from experiments.tuning.tune_adwin import run_tuning as tune_adwin

    print_gpu_status()

    print(f"Loading '{dataset}' windows (cached if available)...")
    windows = load_or_build(dataset, force=fresh_cache)

    print(f"\n[1/4] Tuning XGBoost ({n_trials} trials, resumes if interrupted)...")
    tune_xgb(windows, n_trials=n_trials)

    print(f"\n[2/4] Tuning LightGBM ({n_trials} trials, resumes if interrupted)...")
    tune_lgb(windows, n_trials=n_trials)

    print(f"\n[3/4] Tuning CatBoost ({n_trials} trials, resumes if interrupted)...")
    tune_cb(windows, n_trials=n_trials)

    print("\n[4/4] Tuning ADWIN delta...")
    import pandas as pd
    from src.drift.drift_simulator import InjectedDrift

    X_train_all = pd.concat([w["X_train"] for w in windows[:6]], ignore_index=True)
    y_train_all: pd.Series = pd.concat([w["y_train"] for w in windows[:6]], ignore_index=True)  # type: ignore[assignment]
    static_model = StaticXGBoost()
    static_model.fit(X_train_all, y_train_all)

    # Align test features to training columns once — prevents per-window mismatch
    train_cols = static_model._model.get_booster().feature_names
    drift_positions = {i for i in [2, 6, 10] if i < len(windows[6:])}

    sim = DriftSimulator(seed=42)
    test_windows = []
    for i, w in enumerate(windows[6:]):
        X_aligned = w["X_test"].reindex(columns=train_cols, fill_value=0)
        if i in drift_positions:
            # Multiply all features by 5x — pushes samples past tree thresholds,
            # guaranteeing a detectable spike in error rate for ADWIN to catch.
            test_windows.append({**w, "X_test": X_aligned * 5.0})
            sim.injected_drifts.append(
                InjectedDrift(
                    window_index=w["window_index"],
                    drift_type="feature_shift",
                    magnitude=5.0,
                )
            )
        else:
            test_windows.append({**w, "X_test": X_aligned})

    tune_adwin(test_windows, drift_simulator=sim, model=static_model, n_trials=100)

    print("\nAll tuning complete. Best params saved to:", TUNING_RESULTS)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ieee_cis",
                        help="Dataset to tune on (ieee_cis recommended)")
    parser.add_argument("--n-trials", type=int, default=150)
    parser.add_argument("--fresh-cache", action="store_true",
                        help="Rebuild the window cache from scratch")
    args = parser.parse_args()
    run_all_tuning(
        dataset=args.dataset,
        n_trials=args.n_trials,
        fresh_cache=args.fresh_cache,
    )
