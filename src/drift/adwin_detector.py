"""ADWIN-based concept drift detector.

Wraps River's ADWIN implementation with logging and event tracking.
ADWIN (ADaptive WINdowing) detects drift by monitoring a sliding window
of model error rates and flagging statistically significant changes.

Reference: Bifet & Gavalda (2007) — Learning from time-changing data
with adaptive windowing.
"""

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from river.drift import ADWIN

np.random.seed(42)
random.seed(42)


@dataclass
class DriftEvent:
    """Record of a single drift detection event."""
    window_index: int
    value_at_detection: float
    adwin_width: int
    adwin_estimation: float


class ADWINDetector:
    """Wraps River ADWIN with event logging for the adaptive retraining loop.

    Usage:
        detector = ADWINDetector(delta=0.002)
        for window_idx, error_rate in stream:
            detector.update(error_rate, window_index=window_idx)
            if detector.drift_detected:
                # trigger retraining
    """

    def __init__(self, delta: float = 0.002):
        """
        Args:
            delta: Confidence parameter. Lower = more sensitive.
                   Tuned via experiments/tuning/tune_adwin.py.
        """
        self.delta = delta
        self._adwin = ADWIN(delta=delta)
        self.drift_detected: bool = False
        self.events: list[DriftEvent] = []
        self._current_window: int = -1

    def update(self, value: float, window_index: int = -1) -> bool:
        """Feed a new value to ADWIN and check for drift.

        Args:
            value: Scalar metric (error rate, 1-AUC, prediction confidence, etc.)
            window_index: Current window index for logging.

        Returns:
            True if drift was detected on this update.
        """
        self._current_window = window_index
        self._adwin.update(value)
        self.drift_detected = self._adwin.drift_detected

        if self.drift_detected:
            event = DriftEvent(
                window_index=window_index,
                value_at_detection=value,
                adwin_width=self._adwin.width,
                adwin_estimation=self._adwin.estimation,
            )
            self.events.append(event)

        return self.drift_detected

    def reset(self) -> None:
        """Reset the detector (but preserve event history)."""
        self._adwin = ADWIN(delta=self.delta)
        self.drift_detected = False

    @property
    def n_drift_events(self) -> int:
        return len(self.events)

    @property
    def drift_window_indices(self) -> list[int]:
        return [e.window_index for e in self.events]

    def summary(self) -> dict:
        return {
            "delta": self.delta,
            "n_drift_events": self.n_drift_events,
            "drift_window_indices": self.drift_window_indices,
        }
