"""Optuna hyperparameter tuning for CatBoost.

Temporal validation: train on window N, validate on window N+1.
Uses SQLite storage so the study resumes from where it left off on restart.
Saves best params to experiments/tuning/tuning_results/catboost_best_params.json.
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
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


def objective(trial: optuna.Trial, windows: list, gpu_params: dict) -> float:
    """Temporal cross-validation objective for CatBoost.

    1 fold, 20k rows — keeps each trial under 30s.
    """
    MAX_TRAIN_ROWS = 20_000

    params = {
        "iterations": trial.suggest_int("iterations", 200, 1500, step=50),
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.1, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "border_count": trial.suggest_categorical("border_count", [32, 64, 128, 254]),
        "verbose": 0,
        "random_seed": 42,
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
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train)
    except Exception:
        cpu_params = {k: v for k, v in params.items() if k != "task_type"}
        cpu_params["task_type"] = "CPU"
        model = CatBoostClassifier(**cpu_params)
        model.fit(X_train, y_train)

    proba = model.predict_proba(X_val)[:, 1]
    return float(roc_auc_score(y_val, proba))


def run_tuning(windows: list, n_trials: int = 150, timeout: int = 7200) -> dict:
    """Run (or resume) Optuna study for CatBoost.

    Persists to SQLite — restart picks up where it left off.
    """
    from src.utils.gpu import catboost_device_params
    gpu_params = catboost_device_params()

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=5)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="catboost_tuning",
        storage=STORAGE_URL,
        load_if_exists=True,
    )

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_trials - completed)

    if remaining == 0:
        print(f"[CB Tuning] Already completed {completed} trials — skipping.")
    else:
        print(f"[CB Tuning] {completed} done, {remaining} remaining. GPU: {gpu_params}")
        study.optimize(
            lambda trial: objective(trial, windows, gpu_params),
            n_trials=remaining,
            timeout=timeout,
            show_progress_bar=True,
        )

    best_params = study.best_params
    print(f"\nBest AUC: {study.best_value:.4f}")
    print(f"Best params: {best_params}")

    params_path = RESULTS_DIR / "catboost_best_params.json"
    params_path.write_text(json.dumps(best_params, indent=2))

    import joblib
    joblib.dump(study, RESULTS_DIR / "catboost_study.pkl")

    try:
        import optuna.visualization as ov
        import plotly.io as pio
        pio.write_image(
            ov.plot_param_importances(study),
            str(FIGURES_DIR / "catboost_param_importance.png"),
        )
        pio.write_image(
            ov.plot_optimization_history(study),
            str(FIGURES_DIR / "catboost_opt_history.png"),
        )
        print("Optuna figures saved.")
    except Exception as e:
        print(f"Could not save Optuna figures: {e}")

    return best_params
