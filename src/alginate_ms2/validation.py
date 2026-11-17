"""Seven-fold LOO by calibration *composition*, not individual spectrum.

Only the splitting algorithm is released at this stage. Model fitting,
preprocessing and score calculation must occur *inside* each training fold.
"""
from __future__ import annotations
import numpy as np

EXPECTED_GROUPS = (0, 20, 40, 50, 60, 80, 100)

def grouped_loo_splits(compositions, *, require_seven=True):
    """Yield (held-out nominal %, train indices, test indices).

    All scans with the held-out nominal composition belong to the test fold.
    """
    groups = np.asarray(compositions)
    if groups.ndim != 1 or len(groups) == 0:
        raise ValueError("Expected a nonempty 1-D composition label array")
    if not np.all(np.isfinite(groups.astype(float))):
        raise ValueError("Composition labels must be finite")
    unique = sorted(set(groups.tolist()))
    if require_seven and unique != list(EXPECTED_GROUPS):
        raise ValueError(f"Expected groups {EXPECTED_GROUPS}, observed {unique}")
    for held in unique:
        test = np.flatnonzero(groups == held)
        train = np.flatnonzero(groups != held)
        if not len(test) or not len(train):
            raise ValueError("A fold must have nonempty training and test sets")
        yield held, train, test

def mae_percentage_points(y_true, y_pred):
    a, b = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if a.shape != b.shape or a.size == 0 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Predictions and reference values must be aligned, finite, and nonempty")
    return float(np.mean(np.abs(a - b)))
