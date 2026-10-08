"""Core AUC decomposition utilities for the pooled-AUC knowledge-tracing study.

The implementation uses the standard Mann-Whitney interpretation of AUC with
0.5 credit for ties. It exposes exact positive-negative pair weights for
within-learner and nested within-learner/within-group evaluations.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Hashable, Iterable

import numpy as np
from scipy.stats import rankdata


@dataclass(frozen=True)
class AUCDecomposition:
    auc_pooled: float
    auc_within: float
    auc_between: float
    lambda_within: float
    total_pairs: int
    within_pairs: int
    between_pairs: int

    def as_dict(self) -> dict:
        return asdict(self)


def _arrays(score: Iterable[float], y: Iterable[int]):
    s = np.asarray(score, dtype=float)
    y = np.asarray(y, dtype=int)
    if s.ndim != 1 or y.ndim != 1 or len(s) != len(y):
        raise ValueError("score and y must be one-dimensional arrays of equal length")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y must be binary (0/1)")
    return s, y


def auc_and_numerator(score: Iterable[float], y: Iterable[int]):
    """Return (AUC, Mann-Whitney numerator, positive-negative pair count)."""
    score, y = _arrays(score, y)
    p = int(y.sum())
    n = int(len(y) - p)
    pairs = p * n
    if pairs == 0:
        return float("nan"), 0.0, 0
    ranks = rankdata(score, method="average")
    numerator = float(ranks[y == 1].sum() - p * (p + 1) / 2.0)
    return numerator / pairs, numerator, pairs


def _group_indices(keys: Iterable[Hashable]):
    keys = list(keys)
    groups: dict[Hashable, list[int]] = {}
    for i, k in enumerate(keys):
        if isinstance(k, list):
            k = tuple(k)
        elif isinstance(k, np.ndarray):
            k = tuple(k.tolist())
        groups.setdefault(k, []).append(i)
    return [np.asarray(v, dtype=int) for v in groups.values()]


def weighted_within_auc(score, y, groups):
    """Pair-weighted AUC within groups and its exact pair numerator."""
    score, y = _arrays(score, y)
    if len(groups) != len(y):
        raise ValueError("groups must have one entry per observation")
    numerator = 0.0
    pairs = 0
    for idx in _group_indices(groups):
        _, num_g, pairs_g = auc_and_numerator(score[idx], y[idx])
        numerator += num_g
        pairs += pairs_g
    auc = numerator / pairs if pairs else float("nan")
    return auc, numerator, pairs


def decompose_pooled_auc(score, y, learner) -> AUCDecomposition:
    """Exact pooled-AUC decomposition into within- and between-learner pairs."""
    score, y = _arrays(score, y)
    learner = np.asarray(learner, dtype=object)
    if len(learner) != len(y):
        raise ValueError("learner must have one entry per observation")

    auc_p, num_p, pairs_p = auc_and_numerator(score, y)
    auc_w, num_w, pairs_w = weighted_within_auc(score, y, learner)
    pairs_b = pairs_p - pairs_w
    num_b = num_p - num_w
    auc_b = num_b / pairs_b if pairs_b else float("nan")
    lam = pairs_w / pairs_p if pairs_p else float("nan")
    return AUCDecomposition(
        auc_pooled=auc_p,
        auc_within=auc_w,
        auc_between=auc_b,
        lambda_within=lam,
        total_pairs=pairs_p,
        within_pairs=pairs_w,
        between_pairs=pairs_b,
    )


def nested_within_auc(score, y, learner, subgroup):
    """AUC restricted to pairs sharing both learner and subgroup (KC/item)."""
    learner = np.asarray(learner, dtype=object)
    subgroup = np.asarray(subgroup, dtype=object)
    if len(learner) != len(subgroup):
        raise ValueError("learner and subgroup must have equal length")
    keys = list(zip(learner.tolist(), subgroup.tolist()))
    auc_n, _, pairs_n = weighted_within_auc(score, y, keys)
    _, _, total_pairs = auc_and_numerator(score, y)
    weight = pairs_n / total_pairs if total_pairs else float("nan")
    return {"auc": auc_n, "pair_weight": weight, "pairs": pairs_n, "total_pairs": total_pairs}


def verify_identity(d: AUCDecomposition, atol: float = 1e-12) -> bool:
    """Check pooled = lambda*within + (1-lambda)*between."""
    if d.total_pairs == 0 or d.between_pairs == 0:
        return True
    rhs = d.lambda_within * d.auc_within + (1.0 - d.lambda_within) * d.auc_between
    return bool(np.isclose(d.auc_pooled, rhs, atol=atol, rtol=0.0))
