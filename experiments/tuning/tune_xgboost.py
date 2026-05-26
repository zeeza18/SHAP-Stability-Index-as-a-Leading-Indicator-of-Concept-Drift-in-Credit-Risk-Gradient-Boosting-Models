"""Optuna hyperparameter tuning for XGBoost.

Temporal validation: train on window N, validate on window N+1.
Uses SQLite storage so the study resumes from where it left off on restart.
Saves best params to experiments/tuning/tuning_results/xgboost_best_params.json.
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
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

# SQLite DB shared by all studies — enables resume after kill/crash
STORAGE_URL = f"sqlite:///{RESULTS_DIR}/optuna_studies.db"


_RNG = np.random.default_rng(42)

def _sample_train(X: pd.DataFrame, y: pd.Series, max_rows: int) -> tuple:
    """Stratified subsample for tuning speed. Hyperparams generalise from 50k rows."""
    if len(y) <= max_rows:
        return X, y
    # Keep class ratio
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
    """Temporal cross-validation objective for XGBoost.

    Design choices for speed on large datasets:
    - 1 fold (last window pair): avoids repeated large-window fits.
    - 20k row stratified sample: hyperparameter rankings are stable at this size.
    - No early_stopping_rounds: each round evaluation triggers a CPU->GPU
      data transfer that compounds to minutes per trial; avoid entirely.
    - n_estimators 50-300: smaller range, GPU trains these in seconds.
    150 trials at ~10-20s each = 30-50 minutes total.
    """
    MAX_TRAIN_ROWS = 20_000

    using_gpu = gpu_params.get("device") == "cuda"

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=25),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "scale_pos_weight": trial.suggest_int("scale_pos_weight", 1, 100),
        "random_state": 42,
        "eval_metric": "auc",
        "n_jobs": 1 if using_gpu else -1,
        "use_label_encoder": False,
        **gpu_params,
    }

    # Use only the last window pair (largest, most representative train set)
    i = len(windows) - 1
    train_win = windows[i - 1]
    val_win   = windows[i]

    X_train, y_train = _sample_train(
        train_win["X_train"], train_win["y_train"], MAX_TRAIN_ROWS
    )
    X_val, y_val = val_win["X_test"], val_win["y_test"]

    if y_val.sum() == 0 or (y_val == 0).all():
        return 0.0

    all_cols = sorted(set(X_train.columns) | set(X_val.columns))
    X_train = X_train.reindex(columns=all_cols, fill_value=-999)
    X_val   = X_val.reindex(columns=all_cols, fill_value=-999)

    try:
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
    except Exception:
        cpu_params = {**params, "device": "cpu", "tree_method": "hist", "n_jobs": -1}
        model = xgb.XGBClassifier(**cpu_params)
        model.fit(X_train, y_train, verbose=False)

    proba = model.predict_proba(X_val)[:, 1]
    return float(roc_auc_score(y_val, proba))


def run_tuning(windows: list[dict], n_trials: int = 150, timeout: int = 7200) -> dict:
    """Run (or resume) Optuna study and save results.

    The study persists in SQLite — interrupting and restarting will continue
    from where it left off. Completed studies are skipped immediately.

    Args:
        windows: List of window dicts from DataPipeline / window_cache.
        n_trials: Target number of Optuna trials.
        timeout: Max wall-clock seconds per run.

    Returns:
        Best hyperparameters dict.
    """
    from src.utils.gpu import xgb_device_params
    gpu_params = xgb_device_params()

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=5)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="xgboost_tuning",
        storage=STORAGE_URL,
        load_if_exists=True,   # resume if interrupted
    )

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_trials - completed)

    if remaining == 0:
        print(f"[XGB Tuning] Already completed {completed} trials — skipping.")
    else:
        print(f"[XGB Tuning] {completed} done, {remaining} remaining. GPU: {gpu_params}")
        study.optimize(
            lambda trial: objective(trial, windows, gpu_params),
            n_trials=remaining,
            timeout=timeout,
            show_progress_bar=True,
        )

    best_params = study.best_params
    print(f"\nBest AUC: {study.best_value:.4f}")
    print(f"Best params: {best_params}")

    params_path = RESULTS_DIR / "xgboost_best_params.json"
    params_path.write_text(json.dumps(best_params, indent=2))

    import joblib
    joblib.dump(study, RESULTS_DIR / "xgboost_study.pkl")

    try:
        import optuna.visualization as ov
        import plotly.io as pio
        pio.write_image(
            ov.plot_param_importances(study),
            str(FIGURES_DIR / "xgboost_param_importance.png"),
        )
        pio.write_image(
            ov.plot_optimization_history(study),
            str(FIGURES_DIR / "xgboost_opt_history.png"),
        )
        print("Optuna figures saved.")
    except Exception as e:
        print(f"Could not save Optuna figures: {e}")

    return best_params
