"""Construct score transformations that preserve within-learner ordering.

The construction illustrates non-identification: pooled AUC can move while every
learner's within-learner rank/tie pattern is preserved exactly.
"""
from __future__ import annotations

import numpy as np


def _within_rank01(values: np.ndarray) -> np.ndarray:
    """Map values to [0,1] while preserving ties and strict order."""
    values = np.asarray(values, dtype=float)
    unique = np.unique(values)
    if len(unique) == 1:
        return np.zeros(len(values), dtype=float)
    mapping = {v: i / (len(unique) - 1) for i, v in enumerate(unique)}
    return np.asarray([mapping[v] for v in values], dtype=float)


def restratify(scores, learner, learner_levels, epsilon: float = 1e-3):
    """Add learner-specific levels while preserving within-learner ordering."""
    scores = np.asarray(scores, dtype=float)
    learner = np.asarray(learner, dtype=object)
    if len(scores) != len(learner):
        raise ValueError("scores and learner must have equal length")

    ids = np.unique(learner)
    if isinstance(learner_levels, dict):
        level = {k: float(learner_levels[k]) for k in ids}
    else:
        vals = np.asarray(learner_levels, dtype=float)
        if len(vals) != len(ids):
            raise ValueError("one learner level is required for each unique learner")
        level = dict(zip(ids.tolist(), vals.tolist()))

    out = np.empty_like(scores, dtype=float)
    for sid in ids:
        idx = np.flatnonzero(learner == sid)
        local = _within_rank01(scores[idx]) - 0.5
        out[idx] = level[sid] + epsilon * local
    return out


def levels_from_binary_rate(y, learner):
    """Learner levels equal to each learner's empirical positive rate."""
    y = np.asarray(y, dtype=float)
    learner = np.asarray(learner, dtype=object)
    return {sid: float(y[learner == sid].mean()) for sid in np.unique(learner)}


def reversed_levels(levels: dict):
    """Reverse a set of learner levels around their observed range."""
    vals = np.asarray(list(levels.values()), dtype=float)
    lo, hi = float(vals.min()), float(vals.max())
    return {k: hi + lo - float(v) for k, v in levels.items()}
