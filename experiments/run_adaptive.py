"""Run adaptive models (Adaptive XGBoost + Adaptive LightGBM) on all datasets.

Window-level checkpointing: after each window the model, detector, buffer, SSI
tracker, and metrics row are written to disk. On restart, the script loads the
last saved state and continues from the next unprocessed window — no work lost.

Delete results/checkpoints/adaptive/ to force a full re-run.
"""

import json
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

np.random.seed(42)
random.seed(42)

TABLES_DIR = ROOT / "results" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["ieee_cis", "gmsc", "ccfraud"]


def run_adaptive(
    dataset: str,
    tuned_params_dir: Path | None = None,
    fresh: bool = False,
) -> None:
    from src.data.window_cache import load_or_build
    from src.data.imbalance_handler import ImbalanceHandler
    from src.models.adaptive_xgboost import AdaptiveXGBoost
    from src.models.adaptive_lightgbm import AdaptiveLightGBM
    from src.models.adaptive_catboost import AdaptiveCatBoost
    from src.explainability.shap_engine import SHAPEngine
    from src.explainability.shap_stability_index import ShapStabilityIndex
    from src.utils.checkpoint import CheckpointManager

    print(f"\n{'='*60}")
    print(f"ADAPTIVE — {dataset.upper()}")
    print(f"{'='*60}")

    windows = load_or_build(dataset, force=fresh)
    if not windows:
        print(f"No windows available for {dataset}. Skipping.")
        return

    first_win = windows[0]
    X_train_init, y_train_init = first_win["X_train"], first_win["y_train"]

    handler = ImbalanceHandler(dataset=dataset)
    handler.fit(X_train_init, y_train_init)
    X_train_res, y_train_res = handler.transform(X_train_init, y_train_init, is_train=True)

    # Load tuned params
    xgb_params, lgb_params, cb_params, adwin_params = None, None, None, {}
    if tuned_params_dir:
        for fname, key in [
            ("xgboost_best_params.json", "xgb"),
            ("lightgbm_best_params.json", "lgb"),
            ("catboost_best_params.json", "cb"),
            ("adwin_best_params.json", "adwin"),
        ]:
            p = tuned_params_dir / fname
            if p.exists():
                data = json.loads(p.read_text())
                if key == "xgb":
                    xgb_params = data
                elif key == "lgb":
                    lgb_params = data
                elif key == "cb":
                    cb_params = data
                else:
                    adwin_params = data

    adwin_delta = adwin_params.get("delta", 0.002)
    feature_names = first_win["X_train"].columns.tolist()

    model_configs = {
        "AdaptiveXGBoost": AdaptiveXGBoost(params=xgb_params, adwin_delta=adwin_delta),
        "AdaptiveLightGBM": AdaptiveLightGBM(params=lgb_params, adwin_delta=adwin_delta),
        "AdaptiveCatBoost": AdaptiveCatBoost(params=cb_params, adwin_delta=adwin_delta),
    }

    for model_name, model in model_configs.items():
        ckpt_key = f"adaptive_{model_name}_{dataset}"
        ckpt = CheckpointManager(prefix=ckpt_key, subdir="adaptive")

        if fresh:
            ckpt.clear()

        out_path = TABLES_DIR / f"adaptive_{model_name}_{dataset}.csv"

        if ckpt.is_done() and out_path.exists():
            print(f"  [{model_name}] Already complete — skipping.")
            continue

        # ── Resume or initial fit ────────────────────────────────────────────
        last_window_done = ckpt.get("last_window_done", -1)

        model_ckpt_path = ckpt.dir / f"{ckpt_key}_model.joblib"
        ssi_ckpt_path = ckpt.dir / f"{ckpt_key}_ssi.joblib"

        if ckpt.has_model() and last_window_done >= 0 and not fresh:
            print(f"  [{model_name}] Resuming from window {last_window_done}...")
            model.load(model_ckpt_path)
            ssi_state = joblib.load(ssi_ckpt_path) if ssi_ckpt_path.exists() else {}
            ssi_tracker = ssi_state.get(
                "ssi_tracker",
                ShapStabilityIndex(model_name=f"{model_name}_{dataset}"),
            )
        else:
            print(f"\n  [{model_name}] Initial fit on {len(y_train_res):,} samples...")
            model.fit(X_train_res, y_train_res)
            ssi_tracker = ShapStabilityIndex(
                model_name=f"{model_name}_{dataset}"
            )
            last_window_done = -1

        print(f"  [{model_name}] Streaming windows...")

        for win in windows[1:]:
            win_idx = win["window_index"]

            if win_idx <= last_window_done:
                continue  # already processed before the crash/interrupt

            X_test, y_test = win["X_test"], win["y_test"]

            metrics = model.update_window(
                X_test, y_test, window_index=win_idx, imbalance_handler=handler
            )

            # SHAP + SSI
            try:
                shap_engine = SHAPEngine(model._model, model_name=model_name)
                shap_vals = shap_engine.compute(X_test, y_test, window_index=win_idx)
                ssi = ssi_tracker.update(
                    window_index=win_idx,
                    shap_values=shap_vals,
                    feature_names=feature_names,
                    auc=metrics["roc_auc"],
                    drift_event=metrics["drift_detected"],
                    model_retrained=metrics["model_retrained"],
                )
                metrics["ssi"] = ssi
            except Exception as e:
                print(f"  SHAP error at window {win_idx}: {e}")
                metrics["ssi"] = np.nan

            # Append metrics row immediately (survives interrupt)
            ckpt.append_metrics(metrics)

            print(
                f"  Window {win_idx:3d} | AUC: {metrics['roc_auc']:.4f} | "
                f"SSI: {metrics.get('ssi', float('nan')):.3f} | "
                f"Drift: {metrics['drift_detected']} | Retrain: {metrics['model_retrained']}"
            )

            # Save full model state + SSI tracker after every window
            model.save(model_ckpt_path)
            joblib.dump({"ssi_tracker": ssi_tracker}, ssi_ckpt_path)
            ckpt.save_state(last_window_done=win_idx)

        # ── Finalise ─────────────────────────────────────────────────────────
        # Write final CSV from accumulated checkpoint metrics
        records = ckpt.load_metrics()
        result_df = pd.DataFrame(records)
        result_df.to_csv(out_path, index=False)
        print(f"Metrics saved: {out_path}")

        ssi_path = ssi_tracker.save()
        print(f"SSI saved: {ssi_path}")

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
        run_adaptive(ds, tuned_params_dir=args.tuned_params, fresh=args.fresh)
