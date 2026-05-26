"""Optuna tuning for ADWIN delta parameter.

Objective: maximize F1 of drift detection on synthetic drift
(ground truth known via DriftSimulator).
Uses SQLite storage so the study resumes from where it left off on restart.
Saves best params to experiments/tuning/tuning_results/adwin_best_params.json.
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import optuna

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


def objective(
    trial: optuna.Trial,
    windows: list[dict],
    drift_simulator,
    model,
) -> float:
    """Evaluate ADWIN F1 on synthetic drift."""
    from src.drift.adwin_detector import ADWINDetector

    delta = trial.suggest_float("delta", 0.0001, 0.5, log=True)
    metric = trial.suggest_categorical(
        "metric_to_monitor", ["error_rate", "auc_window", "prediction_confidence"]
    )

    detector = ADWINDetector(delta=delta)
    detected_windows = []

    # ADWIN is a stream algorithm — it needs one value per sample, not per window.
    # Feeding it a single aggregate per window (~17 total) gives it too little data
    # to detect a change. Feed one error/confidence value per prediction instead.
    train_cols = model._model.get_booster().feature_names

    for win in windows:
        X_test = win["X_test"].reindex(columns=train_cols, fill_value=0)
        y_test = win["y_test"]
        proba = model.predict_proba(X_test)
        preds = (proba >= 0.5).astype(int)
        y_vals = y_test.values

        fired_this_window = False
        n_samples = len(y_vals)
        for j in range(n_samples):
            if metric == "error_rate":
                val = float(preds[j] != y_vals[j])
            elif metric == "auc_window":
                val = float(proba[j]) if y_vals[j] == 0 else float(1.0 - proba[j])
            else:  # prediction_confidence
                val = float(1.0 - proba[j])

            fired = detector.update(val, window_index=win["window_index"])
            if fired and not fired_this_window:
                detected_windows.append(win["window_index"])
                fired_this_window = True

    metrics = drift_simulator.compute_detection_metrics(
        detected_windows, tolerance=1
    )
    return metrics["f1"]


def run_tuning(
    windows: list[dict],
    drift_simulator,
    model,
    n_trials: int = 100,
    timeout: int = 3600,
) -> dict:
    """Run (or resume) Optuna study and save results.

    The study persists in SQLite — interrupting and restarting will continue
    from where it left off. Completed studies are skipped immediately.

    Args:
        windows: List of window dicts from DataPipeline / window_cache.
        drift_simulator: DriftSimulator instance with injected drift ground truth.
        model: Fitted model with predict_proba() and XGBoost booster (for feature names).
        n_trials: Target number of Optuna trials.
        timeout: Max wall-clock seconds per run.

    Returns:
        Best hyperparameters dict.
    """
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=5)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="adwin_tuning",
        storage=STORAGE_URL,
        load_if_exists=True,
    )

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_trials - completed)

    if remaining == 0:
        print(f"[ADWIN Tuning] Already completed {completed} trials — skipping.")
    else:
        print(f"[ADWIN Tuning] {completed} done, {remaining} remaining.")
        study.optimize(
            lambda trial: objective(trial, windows, drift_simulator, model),
            n_trials=remaining,
            timeout=timeout,
            show_progress_bar=True,
        )

    best_params = study.best_params
    print(f"\nBest drift detection F1: {study.best_value:.4f}")
    print(f"Best ADWIN params: {best_params}")

    params_path = RESULTS_DIR / "adwin_best_params.json"
    params_path.write_text(json.dumps(best_params, indent=2))

    import joblib
    joblib.dump(study, RESULTS_DIR / "adwin_study.pkl")

    try:
        import optuna.visualization as ov
        import plotly.io as pio
        pio.write_image(
            ov.plot_param_importances(study),
            str(FIGURES_DIR / "adwin_param_importance.png"),
        )
        pio.write_image(
            ov.plot_optimization_history(study),
            str(FIGURES_DIR / "adwin_opt_history.png"),
        )
        print("Optuna figures saved.")
    except Exception as e:
        print(f"Could not save Optuna figures: {e}")

    return best_params


if __name__ == "__main__":
    print("ADWIN tuning: requires fitted model + DriftSimulator with injected drift.")
    print("Load windows, inject synthetic drift, fit a static model, then call run_tuning().")
