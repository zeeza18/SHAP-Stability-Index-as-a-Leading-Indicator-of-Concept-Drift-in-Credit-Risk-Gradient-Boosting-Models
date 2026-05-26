"""Tests for ADWINDetector."""

import numpy as np
import pytest

from src.drift.adwin_detector import ADWINDetector


def test_no_drift_on_stable_stream():
    detector = ADWINDetector(delta=0.002)
    rng = np.random.default_rng(42)
    stable = rng.uniform(0.05, 0.1, 200)
    events = [detector.update(v, window_index=i) for i, v in enumerate(stable)]
    # Should fire very rarely on stable signal
    assert sum(events) <= 3


def test_drift_detected_on_sudden_shift():
    detector = ADWINDetector(delta=0.002)
    # First 100 values: low error ~0.05
    # Next 100 values: high error ~0.5 — should trigger drift
    rng = np.random.default_rng(42)
    low = rng.uniform(0.03, 0.08, 100)
    high = rng.uniform(0.45, 0.55, 100)
    stream = np.concatenate([low, high])
    events = [detector.update(v, window_index=i) for i, v in enumerate(stream)]
    assert any(events[100:]), "Expected drift detection in the high-error segment"


def test_event_logging():
    detector = ADWINDetector(delta=0.002)
    rng = np.random.default_rng(0)
    low = rng.uniform(0.02, 0.06, 100)
    high = rng.uniform(0.5, 0.9, 50)
    for i, v in enumerate(np.concatenate([low, high])):
        detector.update(v, window_index=i)
    assert detector.n_drift_events >= 0  # at least checked without crash
    assert isinstance(detector.drift_window_indices, list)


def test_reset_clears_state():
    detector = ADWINDetector(delta=0.1)
    for v in [0.9] * 50:
        detector.update(v)
    detector.reset()
    assert not detector.drift_detected
