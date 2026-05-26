"""Optuna hyperparameter tuning for LightGBM.

Temporal validation: train on window N, validate on window N+1.
Uses SQLite storage so the study resumes from where it left off on restart.
Saves best params to experiments/tuning/tuning_results/lightgbm_best_params.json.
"""

import json
import random
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
np.random.seed(42)
random.seed(42)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "experiments" / "tuning" / "tuning_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR = ROOT / "results" / "figures" / "tuning"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

STORAGE_URL = f"sqlite:///{RESULTS_DIR}/optuna_studies.db"


_RNG = np.random.default_rng(42)

def _sample_train(X: pd.DataFrame, y: pd.Series, max_rows: int) -> tuple:
    """Stratified subsample for tuning speed."""
    if len(y) <= max_rows:
        return X, y
    pos_idx = np.where(y.values == 1)[0]
    neg_idx = np.where(y.values == 0)[0]
    ratio = len(pos_idx) / len(y)
    n_pos = max(1, int(max_rows * ratio))
    n_neg = max_rows - n_pos
    chosen = np.concatenate([
        _RNG.choice(pos_idx, min(n_pos, len(pos_idx)), replace=False),
        _RNG.choice(neg_idx, min(n_neg, len(neg_idx)), replace=False),
    ])
    _RNG.shuffle(chosen)
    return X.iloc[chosen].reset_index(drop=True), y.iloc[chosen].reset_index(drop=True)


def objective(trial: optuna.Trial, windows: list[dict], gpu_params: dict) -> float:
    """Temporal cross-validation objective for LightGBM.

    1 fold, 20k rows, no callbacks — keeps each trial under 20s.
    """
    MAX_TRAIN_ROWS = 20_000

    using_gpu = gpu_params.get("device") == "gpu"

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500, step=50),
        "num_leaves": trial.suggest_int("num_leaves", 20, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        "is_unbalance": trial.suggest_categorical("is_unbalance", [True, False]),
        "verbose": -1,
        "n_jobs": 1 if using_gpu else -1,
        "random_state": 42,
        **gpu_params,
    }

    i = len(windows) - 1
    X_train, y_train = _sample_train(
        windows[i - 1]["X_train"], windows[i - 1]["y_train"], MAX_TRAIN_ROWS
    )
    X_val, y_val = windows[i]["X_test"], windows[i]["y_test"]

    if y_val.sum() == 0 or (y_val == 0).all():
        return 0.0

    try:
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
    except Exception:
        cpu_params = {k: v for k, v in params.items() if k != "device"}
        cpu_params["n_jobs"] = -1
        model = lgb.LGBMClassifier(**cpu_params)
        model.fit(X_train, y_train)

    proba = model.predict_proba(X_val)[:, 1]
    return float(roc_auc_score(y_val, proba))


def run_tuning(windows: list[dict], n_trials: int = 150, timeout: int = 7200) -> dict:
    """Run (or resume) Optuna study for LightGBM.

    Persists to SQLite — restart picks up where it left off.
    """
    from src.utils.gpu import lgb_device_params
    gpu_params = lgb_device_params()

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=5)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="lightgbm_tuning",
        storage=STORAGE_URL,
        load_if_exists=True,
    )

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_trials - completed)

    if remaining == 0:
        print(f"[LGB Tuning] Already completed {completed} trials — skipping.")
    else:
        print(f"[LGB Tuning] {completed} done, {remaining} remaining. GPU: {gpu_params}")
        study.optimize(
            lambda trial: objective(trial, windows, gpu_params),
            n_trials=remaining,
            timeout=timeout,
            show_progress_bar=True,
        )

    best_params = study.best_params
    print(f"\nBest AUC: {study.best_value:.4f}")
    print(f"Best params: {best_params}")

    params_path = RESULTS_DIR / "lightgbm_best_params.json"
    params_path.write_text(json.dumps(best_params, indent=2))

    import joblib
    joblib.dump(study, RESULTS_DIR / "lightgbm_study.pkl")

    try:
        import optuna.visualization as ov
        import plotly.io as pio
        pio.write_image(
            ov.plot_param_importances(study),
            str(FIGURES_DIR / "lightgbm_param_importance.png"),
        )
        pio.write_image(
            ov.plot_optimization_history(study),
            str(FIGURES_DIR / "lightgbm_opt_history.png"),
        )
        print("Optuna figures saved.")
    except Exception as e:
        print(f"Could not save Optuna figures: {e}")

    return best_params
