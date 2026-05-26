"""GPU detection utility for XGBoost, LightGBM, and CatBoost.

Detects once at import time (cached). Call xgb_device_params(), lgb_device_params(),
or catboost_device_params() to get the right params dict to merge into any model.

RTX 4060 / CUDA setup:
  XGBoost 2.x:  device='cuda', tree_method='hist'
  LightGBM 4.x: device='gpu'  (requires LightGBM built with GPU support)
  CatBoost:     task_type='GPU'
"""

import os
import numpy as np

_XGB_PARAMS: dict | None = None
_LGB_PARAMS: dict | None = None
_CB_PARAMS: dict | None = None


def _probe_xgb() -> dict:
    if os.environ.get("FORCE_CPU", "0") == "1":
        print("[GPU] XGBoost: forced CPU (FORCE_CPU=1)")
        return {"device": "cpu", "tree_method": "hist"}
    try:
        import xgboost as xgb
        rng = np.random.default_rng(0)
        X = rng.random((20, 4)).astype(np.float32)
        y = (rng.random(20) > 0.5).astype(int)
        m = xgb.XGBClassifier(
            device="cuda", tree_method="hist", n_estimators=1, verbosity=0
        )
        m.fit(X, y, verbose=False)
        print("[GPU] XGBoost: CUDA available - RTX 4060 enabled")
        return {"device": "cuda", "tree_method": "hist"}
    except Exception as e:
        print(f"[GPU] XGBoost: no CUDA ({e!s:.60}), using CPU hist")
        return {"device": "cpu", "tree_method": "hist"}


def _probe_lgb() -> dict:
    if os.environ.get("FORCE_CPU", "0") == "1":
        print("[GPU] LightGBM: forced CPU (FORCE_CPU=1)")
        return {}
    try:
        import lightgbm as lgb
        rng = np.random.default_rng(0)
        X = rng.random((20, 4))
        y = (rng.random(20) > 0.5).astype(int)
        m = lgb.LGBMClassifier(device="gpu", n_estimators=1, verbose=-1)
        m.fit(X, y)
        print("[GPU] LightGBM: GPU available - RTX 4060 enabled")
        return {"device": "gpu"}
    except Exception as e:
        print(f"[GPU] LightGBM: no GPU ({e!s:.60}), using CPU")
        return {}


def xgb_device_params() -> dict:
    """Return XGBoost device params (cached after first call)."""
    global _XGB_PARAMS
    if _XGB_PARAMS is None:
        _XGB_PARAMS = _probe_xgb()
    return _XGB_PARAMS


def lgb_device_params() -> dict:
    """Return LightGBM device params (cached after first call)."""
    global _LGB_PARAMS
    if _LGB_PARAMS is None:
        _LGB_PARAMS = _probe_lgb()
    return _LGB_PARAMS


def _probe_catboost() -> dict:
    if os.environ.get("FORCE_CPU", "0") == "1":
        print("[GPU] CatBoost: forced CPU (FORCE_CPU=1)")
        return {"task_type": "CPU"}
    try:
        from catboost import CatBoostClassifier
        rng = np.random.default_rng(0)
        X = rng.random((20, 4))
        y = (rng.random(20) > 0.5).astype(int)
        m = CatBoostClassifier(iterations=1, task_type="GPU", verbose=0)
        m.fit(X, y)
        print("[GPU] CatBoost: GPU available - RTX 4060 enabled")
        return {"task_type": "GPU"}
    except Exception as e:
        print(f"[GPU] CatBoost: no GPU ({e!s:.60}), using CPU")
        return {"task_type": "CPU"}


def catboost_device_params() -> dict:
    """Return CatBoost device params (cached after first call)."""
    global _CB_PARAMS
    if _CB_PARAMS is None:
        _CB_PARAMS = _probe_catboost()
    return _CB_PARAMS


def print_gpu_status() -> None:
    """Print GPU availability for all three libraries. Useful at pipeline start."""
    print("\n" + "=" * 50)
    print("GPU STATUS CHECK")
    print("=" * 50)
    xgb_device_params()
    lgb_device_params()
    catboost_device_params()
    print("=" * 50 + "\n")
