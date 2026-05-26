"""Ablation study: 2x2 design to isolate component contributions.

Condition 1: Full system (drift detection + adaptive retraining)
Condition 2: No drift detection (fixed interval retrain every 3 windows)
Condition 3: No adaptive retraining (detect drift but don't retrain)
Condition 4: Static baseline (no drift detection, no retraining)

Checkpointing: each (condition, dataset) combo is checkpointed so the
study can be restarted without repeating completed conditions.
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

np.random.seed(42)
random.seed(42)

TABLES_DIR = ROOT / "results" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

FIXED_RETRAIN_INTERVAL = 3


def run_ablation(dataset: str, fresh: bool = False) -> pd.DataFrame:
    from src.data.window_cache import load_or_build
    from src.data.imbalance_handler import ImbalanceHandler
    from src.evaluation.metrics import evaluate_window
    from src.utils.checkpoint import CheckpointManager
    from src.utils.gpu import xgb_device_params
    import xgboost as xgb

    print(f"\nAblation study — {dataset.upper()}")

    ckpt_key = f"ablation_{dataset}"
    ckpt = CheckpointManager(prefix=ckpt_key, subdir="ablation")

    if fresh:
        ckpt.clear()

    out_path = TABLES_DIR / f"ablation_{dataset}.csv"
    if ckpt.is_done() and out_path.exists():
        print(f"  Already complete — skipping. (use --fresh to re-run)")
        return pd.read_csv(out_path)

    windows = load_or_build(dataset, force=fresh)
    if not windows:
        return pd.DataFrame()

    first_win = windows[0]
    handler = ImbalanceHandler(dataset=dataset)
    handler.fit(first_win["X_train"], first_win["y_train"])
    X0_res, y0_res = handler.transform(
        first_win["X_train"], first_win["y_train"], is_train=True
    )

    gpu = xgb_device_params()
    base_params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "random_state": 42,
        "use_label_encoder": False,
        "eval_metric": "auc",
        **gpu,
    }

    conditions = [
        ("Full System",            True,  True),
        ("No Drift Detection",     False, True),
        ("No Adaptive Retraining", True,  False),
        ("Static Baseline",        False, False),
    ]

    completed_conditions: list[str] = ckpt.get("completed_conditions", [])
    all_records: list[dict] = ckpt.load_metrics()

    for condition_name, use_drift, use_retrain in conditions:
        if condition_name in completed_conditions:
            print(f"  Condition '{condition_name}' already done — skipping.")
            continue

        print(f"\n  Condition: {condition_name}")
        from collections import deque
        from src.drift.adwin_detector import ADWINDetector

        model = xgb.XGBClassifier(**base_params)
        model.fit(X0_res, y0_res, verbose=False)
        buffer: deque = deque(maxlen=5)
        detector = ADWINDetector(delta=0.002)

        for i, win in enumerate(windows[1:], start=1):
            X_test, y_test = win["X_test"], win["y_test"]
            win_idx = win["window_index"]

            proba = model.predict_proba(X_test)[:, 1]
            preds = (proba >= 0.5).astype(int)
            error_rate = (preds != y_test.values).mean()

            retrain = False
            if use_drift:
                drift = detector.update(error_rate, window_index=win_idx)
                if drift and use_retrain and buffer:
                    X_r = pd.concat([b[0] for b in buffer], ignore_index=True)
                    y_r = pd.concat([b[1] for b in buffer], ignore_index=True)
                    model = xgb.XGBClassifier(**base_params)
                    model.fit(X_r, y_r, verbose=False)
                    retrain = True
            elif use_retrain:
                # Fixed interval retraining
                if i % FIXED_RETRAIN_INTERVAL == 0 and buffer:
                    X_r = pd.concat([b[0] for b in buffer], ignore_index=True)
                    y_r = pd.concat([b[1] for b in buffer], ignore_index=True)
                    model = xgb.XGBClassifier(**base_params)
                    model.fit(X_r, y_r, verbose=False)
                    retrain = True

            buffer.append((X_test, y_test))

            proba_final = model.predict_proba(X_test)[:, 1]
            metrics = evaluate_window(y_test.values, proba_final, window_index=win_idx)
            metrics["condition"] = condition_name
            metrics["model_retrained"] = retrain
            all_records.append(metrics)
            ckpt.append_metrics(metrics)

        completed_conditions.append(condition_name)
        ckpt.save_state(completed_conditions=completed_conditions)
        print(f"  Condition '{condition_name}' complete.")

    result_df = pd.DataFrame(all_records)
    result_df.to_csv(out_path, index=False)
    print(f"\nAblation results saved: {out_path}")

    ckpt.mark_done()
    return result_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--fresh", action="store_true", help="Ignore checkpoints and re-run")
    args = parser.parse_args()

    datasets = ["ieee_cis", "gmsc", "ccfraud"] if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        run_ablation(ds, fresh=args.fresh)
