"""Checkpoint manager — save/resume experiment state at any granularity.

Each experiment (model × dataset) gets its own CheckpointManager.
Checkpoints are stored under results/checkpoints/<subdir>/.

State JSON tracks: what's done, last completed window, etc.
Model joblib stores: serialized model, detector, buffer, SSI tracker, etc.
"""

import json
from collections import deque
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

CHECKPOINTS_DIR = Path(__file__).resolve().parents[3] / "results" / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


class CheckpointManager:
    """Manages checkpoint state for a single experiment run.

    Files written:
      {dir}/{prefix}_state.json   — lightweight JSON metadata
      {dir}/{prefix}_model.joblib — heavy model/detector/buffer state
      {dir}/{prefix}_metrics.csv  — accumulated per-window metrics
    """

    def __init__(self, prefix: str, subdir: str = ""):
        self.prefix = prefix
        self.dir = (CHECKPOINTS_DIR / subdir) if subdir else CHECKPOINTS_DIR
        self.dir.mkdir(parents=True, exist_ok=True)

        self._state_path = self.dir / f"{prefix}_state.json"
        self._model_path = self.dir / f"{prefix}_model.joblib"
        self._metrics_path = self.dir / f"{prefix}_metrics.csv"

        self._state: dict = self._load_state()

    # ------------------------------------------------------------------ state
    def _load_state(self) -> dict:
        if self._state_path.exists():
            return json.loads(self._state_path.read_text())
        return {}

    def save_state(self, **kwargs) -> None:
        self._state.update(kwargs)
        self._state_path.write_text(json.dumps(self._state, indent=2, default=str))

    def get(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    def is_done(self) -> bool:
        return bool(self._state.get("done", False))

    def mark_done(self) -> None:
        self.save_state(done=True)

    # ------------------------------------------------------------------ model
    def save_model(self, obj: Any) -> None:
        joblib.dump(obj, self._model_path)

    def load_model(self) -> Any:
        return joblib.load(self._model_path)

    def has_model(self) -> bool:
        return self._model_path.exists()

    # ---------------------------------------------------------------- metrics
    def append_metrics(self, row: dict) -> None:
        """Append one row to the incremental metrics CSV."""
        df_row = pd.DataFrame([row])
        if self._metrics_path.exists():
            df_row.to_csv(self._metrics_path, mode="a", header=False, index=False)
        else:
            df_row.to_csv(self._metrics_path, index=False)

    def load_metrics(self) -> list[dict]:
        if self._metrics_path.exists():
            return pd.read_csv(self._metrics_path).to_dict("records")
        return []

    def has_metrics(self) -> bool:
        return self._metrics_path.exists()

    # ------------------------------------------------------------------ util
    def clear(self) -> None:
        """Delete all checkpoint files for this prefix (force re-run)."""
        for p in [self._state_path, self._model_path, self._metrics_path]:
            if p.exists():
                p.unlink()
        self._state = {}
        print(f"[Checkpoint] Cleared {self.prefix}")

    def summary(self) -> str:
        last = self._state.get("last_window_done", "none")
        done = self._state.get("done", False)
        n_rows = len(self.load_metrics()) if self.has_metrics() else 0
        return (
            f"[Checkpoint:{self.prefix}] "
            f"done={done}  last_window={last}  metrics_rows={n_rows}"
        )


def clear_all_checkpoints(subdir: str = "") -> None:
    """Wipe the checkpoints directory (or a sub-directory). Use --fresh flag."""
    target = (CHECKPOINTS_DIR / subdir) if subdir else CHECKPOINTS_DIR
    import shutil
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    print(f"[Checkpoint] Cleared all checkpoints in {target}")
