"""Run static baseline models (Static XGBoost + Static LightGBM) on all datasets.

Trains once on all training windows, evaluates on each test window.
Saves per-window metrics to results/tables/.

Checkpointing: if a model checkpoint exists for (model, dataset), the fit step is
skipped and evaluation resumes. Delete results/checkpoints/baseline/ to re-run.
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

DATASETS = ["ieee_cis", "gmsc", "ccfraud"]


def run_baseline(
    dataset: str,
    tuned_params_dir: Path | None = None,
    fresh: bool = False,
) -> None:
    from src.data.window_cache import load_or_build
    from src.data.imbalance_handler import ImbalanceHandler
    from src.models.static_xgboost import StaticXGBoost
    from src.models.static_lightgbm import StaticLightGBM
    from src.models.static_catboost import StaticCatBoost
    from src.evaluation.metrics import evaluate_window
    from src.utils.checkpoint import CheckpointManager

    print(f"\n{'='*60}")
    print(f"BASELINE — {dataset.upper()}")
    print(f"{'='*60}")

    windows = load_or_build(dataset, force=fresh)
    if not windows:
        print(f"No windows available for {dataset}. Skipping.")
        return

    # Build full training set
    X_train_all = pd.concat([w["X_train"] for w in windows], ignore_index=True)
    y_train_all: pd.Series = pd.concat([w["y_train"] for w in windows], ignore_index=True)  # type: ignore[assignment]
    # Windows may have different selected features; fill gaps with -999 (project convention)
    X_train_all = X_train_all.fillna(-999)

    handler = ImbalanceHandler(dataset=dataset)
    handler.fit(X_train_all, y_train_all)
    X_train_res, y_train_res = handler.transform(X_train_all, y_train_all, is_train=True)

    # Load tuned params if available
    xgb_params, lgb_params, cb_params = None, None, None
    if tuned_params_dir:
        xgb_path = tuned_params_dir / "xgboost_best_params.json"
        lgb_path = tuned_params_dir / "lightgbm_best_params.json"
        cb_path = tuned_params_dir / "catboost_best_params.json"
        if xgb_path.exists():
            xgb_params = json.loads(xgb_path.read_text())
        if lgb_path.exists():
            lgb_params = json.loads(lgb_path.read_text())
        if cb_path.exists():
            cb_params = json.loads(cb_path.read_text())

    models = {
        "StaticXGBoost": StaticXGBoost(params=xgb_params),
        "StaticLightGBM": StaticLightGBM(params=lgb_params),
        "StaticCatBoost": StaticCatBoost(params=cb_params),
    }

    for model_name, model in models.items():
        ckpt_key = f"baseline_{model_name}_{dataset}"
        ckpt = CheckpointManager(prefix=ckpt_key, subdir="baseline")

        if fresh:
            ckpt.clear()

        out_path = TABLES_DIR / f"baseline_{model_name}_{dataset}.csv"

        if ckpt.is_done() and out_path.exists():
            print(f"  [{model_name}] Already complete — skipping. (delete checkpoints/baseline/ to re-run)")
            continue

        # Fit (or load from checkpoint)
        model_ckpt_path = ckpt.dir / f"{ckpt_key}_fitted.joblib"
        if ckpt.has_model() and not fresh:
            print(f"  [{model_name}] Loading fitted model from checkpoint...")
            model.load(model_ckpt_path)
        else:
            print(f"\nFitting {model_name} on {len(y_train_res):,} samples...")
            model.fit(X_train_res, y_train_res)
            model.save(model_ckpt_path)
            ckpt.save_state(fitted=True)

        # Evaluate window by window
        records = []
        for win in windows:
            metrics = evaluate_window(
                win["y_test"].values,
                model.predict_proba(win["X_test"]),
                window_index=win["window_index"],
            )
            records.append(metrics)
            print(f"  Window {win['window_index']:3d} | AUC: {metrics['roc_auc']:.4f}")

        result_df = pd.DataFrame(records)
        result_df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

        ckpt.mark_done()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS + ["all"], default="all")
    parser.add_argument("--tuned-params", type=Path, default=None)
    parser.add_argument("--fresh", action="store_true", help="Ignore checkpoints and re-run")
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        run_baseline(ds, tuned_params_dir=args.tuned_params, fresh=args.fresh)
