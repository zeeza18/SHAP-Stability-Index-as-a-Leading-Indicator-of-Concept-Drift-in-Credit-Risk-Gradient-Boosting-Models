"""Cache preprocessed windows to parquet so DataPipeline only runs once per dataset.

Cache lives at: data/processed/<dataset>_cache/
  meta.json         — window count + column list
  window_NNNN/
    X_train.parquet
    X_test.parquet
    y_train.parquet
    y_test.parquet
    meta.json         — window_index + any extra scalar metadata

Usage:
    from src.data.window_cache import load_or_build

    windows = load_or_build("ieee_cis", force=False)
"""

import json
from pathlib import Path

import pandas as pd

CACHE_ROOT = Path(__file__).resolve().parents[3] / "data" / "processed"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)


def _cache_dir(dataset: str) -> Path:
    return CACHE_ROOT / f"{dataset}_cache"


def is_cached(dataset: str) -> bool:
    return (_cache_dir(dataset) / "meta.json").exists()


def save_windows(dataset: str, windows: list[dict]) -> None:
    """Persist windows list to parquet. Overwrites any existing cache."""
    base = _cache_dir(dataset)
    if base.exists():
        import shutil
        shutil.rmtree(base)
    base.mkdir(parents=True)

    for i, win in enumerate(windows):
        wd = base / f"window_{i:04d}"
        wd.mkdir()

        win["X_train"].to_parquet(wd / "X_train.parquet", index=False)
        win["X_test"].to_parquet(wd / "X_test.parquet", index=False)
        win["y_train"].rename("label").to_frame().to_parquet(
            wd / "y_train.parquet", index=False
        )
        win["y_test"].rename("label").to_frame().to_parquet(
            wd / "y_test.parquet", index=False
        )

        # Scalar metadata only (no DataFrames)
        scalar_meta = {
            k: str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v
            for k, v in win.items()
            if k not in ("X_train", "X_test", "y_train", "y_test")
        }
        (wd / "meta.json").write_text(json.dumps(scalar_meta, default=str))

    (base / "meta.json").write_text(json.dumps({"n_windows": len(windows)}))
    print(f"[Cache] Saved {len(windows)} windows for '{dataset}' -> {base}")


def _align_to_window0(windows: list[dict]) -> list[dict]:
    """Align all windows to window 0's X_train feature set.

    Per-window LightGBM feature selection in FeatureEngineer produces a
    different top-200 vocabulary for each window. Models trained on window 0
    cannot predict on later windows without this alignment. Missing columns
    are filled with -999 (the project-wide sentinel for unknown/missing).
    """
    if not windows:
        return windows
    ref_cols = windows[0]["X_train"].columns.tolist()
    for win in windows:
        for key in ("X_train", "X_test"):
            df = win[key]
            for col in ref_cols:
                if col not in df.columns:
                    df[col] = -999.0
            win[key] = df[ref_cols]
    return windows


def load_windows(dataset: str) -> list[dict]:
    """Load cached windows from parquet."""
    base = _cache_dir(dataset)
    meta = json.loads((base / "meta.json").read_text())
    n = meta["n_windows"]

    windows = []
    for i in range(n):
        wd = base / f"window_{i:04d}"
        win_meta = json.loads((wd / "meta.json").read_text())

        # Restore numeric types for window_index
        if "window_index" in win_meta:
            win_meta["window_index"] = int(win_meta["window_index"])

        win = {
            "X_train": pd.read_parquet(wd / "X_train.parquet"),
            "X_test": pd.read_parquet(wd / "X_test.parquet"),
            "y_train": pd.read_parquet(wd / "y_train.parquet")["label"],
            "y_test": pd.read_parquet(wd / "y_test.parquet")["label"],
            **win_meta,
        }
        windows.append(win)

    print(f"[Cache] Loaded {n} windows for '{dataset}' from cache")
    return _align_to_window0(windows)


def load_or_build(dataset: str, force: bool = False) -> list[dict]:
    """Return windows, building from DataPipeline and caching if not already cached.

    Args:
        dataset: One of 'ieee_cis', 'gmsc', 'ccfraud'.
        force: If True, ignore existing cache and rebuild.
    """
    if not force and is_cached(dataset):
        return load_windows(dataset)

    print(f"[Cache] Building windows for '{dataset}' (this runs DataPipeline once)...")
    from src.data.preprocessor import DataPipeline
    pipeline = DataPipeline(dataset=dataset)
    windows = pipeline.run()

    if windows:
        save_windows(dataset, windows)
    return windows
