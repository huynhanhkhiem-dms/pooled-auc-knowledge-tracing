"""
Exact within/between-cluster decomposition of the pooled ROC-AUC.

Implements the estimators and diagnostics defined in the paper

    "Knowledge Tracing Is Not What AUC Measures:
     A Within-Student Decomposition of the Area Under the ROC Curve"

The pooled AUC of a binary-outcome model evaluated on clustered data
(here: interactions nested within students) is an exactly weighted average
of a *within-cluster* concordance and a *between-cluster* concordance.
This module computes those components, the mixing weights, and
cluster-bootstrap confidence intervals for them.

All AUCs use the Mann-Whitney definition with 1/2 credit for ties,
which is the standard (and the one `sklearn.metrics.roc_auc_score` returns).

Author: (anonymised for review)
License: MIT
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "auc_mw",
    "grouped_auc",
    "decompose_auc",
    "pair_mass",
    "cluster_bootstrap",
    "within_precision_at_k",
]


# --------------------------------------------------------------------------
# Core AUC
# --------------------------------------------------------------------------
def _midranks(x: np.ndarray) -> np.ndarray:
    """Ranks 1..n with average (mid-) ranks for ties. Vectorised."""
    n = x.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    new_block = np.empty(n, dtype=bool)
    new_block[0] = True
    np.not_equal(xs[1:], xs[:-1], out=new_block[1:])
    block_id = np.cumsum(new_block) - 1
    pos = np.arange(1, n + 1, dtype=np.float64)
    bsz = np.bincount(block_id)
    bsum = np.bincount(block_id, weights=pos)
    ranks = (bsum / bsz)[block_id]
    out = np.empty(n, dtype=np.float64)
    out[order] = ranks
    return out


def auc_mw(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Mann-Whitney AUC with 1/2 credit for ties.

    Returns
    -------
    (auc, n_pairs) : float, float
        ``n_pairs`` is P * N, the number of (positive, negative) pairs.
        ``auc`` is ``nan`` when the label vector is constant.
    """
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=np.float64)
    P = float(y.sum())
    N = float(y.shape[0] - P)
    if P == 0.0 or N == 0.0:
        return float("nan"), P * N
    r = _midranks(p)
    auc = (r[y == 1].sum() - P * (P + 1.0) / 2.0) / (P * N)
    return float(auc), P * N


# --------------------------------------------------------------------------
# Grouped (within-cluster) AUC
# --------------------------------------------------------------------------
def _group_slices(g: np.ndarray):
    """Yield (order, start, stop) slices of a sorted-by-group index array."""
    order = np.argsort(g, kind="mergesort")
    gs = g[order]
    # boundaries
    cut = np.flatnonzero(np.diff(gs)) + 1
    starts = np.concatenate(([0], cut))
    stops = np.concatenate((cut, [gs.shape[0]]))
    return order, starts, stops


def grouped_auc(
    y: np.ndarray, p: np.ndarray, g: np.ndarray
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Pair-weighted average of per-group AUCs (the within-cluster AUC).

    Groups whose labels are constant contribute zero pairs and are skipped
    (they are *uninformative*, not *wrong* -- they carry no concordance
    information at all).

    Returns
    -------
    auc_w   : float   pair-weighted mean of per-group AUCs
    n_pairs : float   sum over groups of P_g * N_g
    per_auc : ndarray per-group AUCs (nan for constant-label groups)
    per_w   : ndarray per-group pair counts P_g * N_g
    """
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=np.float64)
    g = np.asarray(g)
    order, starts, stops = _group_slices(g)
    ys, ps = y[order], p[order]

    n_groups = starts.shape[0]
    per_auc = np.full(n_groups, np.nan)
    per_w = np.zeros(n_groups)
    for k in range(n_groups):
        a, b = starts[k], stops[k]
        yy = ys[a:b]
        s = yy.sum()
        if s == 0 or s == (b - a):
            continue
        auc, w = auc_mw(yy, ps[a:b])
        per_auc[k] = auc
        per_w[k] = w
    tot = per_w.sum()
    if tot == 0:
        return float("nan"), 0.0, per_auc, per_w
    auc_w = float(np.nansum(per_auc * per_w) / tot)
    return auc_w, float(tot), per_auc, per_w


# --------------------------------------------------------------------------
# The decomposition
# --------------------------------------------------------------------------
def pair_mass(y: np.ndarray, g: np.ndarray) -> float:
    """Number of (positive, negative) pairs falling inside the same group."""
    y = np.asarray(y, dtype=np.int8)
    g = np.asarray(g)
    order, starts, stops = _group_slices(g)
    ys = y[order]
    tot = 0.0
    for a, b in zip(starts, stops):
        s = float(ys[a:b].sum())
        tot += s * ((b - a) - s)
    return tot


def decompose_auc(
    y: np.ndarray,
    p: np.ndarray,
    student: np.ndarray,
    skill: np.ndarray | None = None,
    item: np.ndarray | None = None,
) -> dict:
    """Full three-level decomposition of the pooled AUC.

    Level B : pairs from *different* students        (ability ranking)
    Level W : pairs from the same student            (within-student ranking)
        W-K : same student, same knowledge component (knowledge tracing signal)
        W-I : same student, same item                (pure learning signal)

    Returns a dict with the AUCs, the mixing weights lambda_*, and the raw
    pair counts.  The identity

        AUC_pooled = lambda_W * AUC_W + (1 - lambda_W) * AUC_B

    holds exactly (up to floating point) by construction.
    """
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=np.float64)
    student = np.asarray(student)

    pooled, n_all = auc_mw(y, p)
    auc_w, n_w, per_auc, per_w = grouped_auc(y, p, student)

    lam_w = n_w / n_all if n_all > 0 else np.nan
    # between-student component recovered from the exact identity
    auc_b = (pooled - lam_w * auc_w) / (1.0 - lam_w) if lam_w < 1.0 else np.nan

    out = {
        "auc_pooled": pooled,
        "auc_within": auc_w,
        "auc_between": auc_b,
        "lambda_within": lam_w,
        "n_pairs_total": n_all,
        "n_pairs_within": n_w,
        "S_eff": (n_all / n_w) if n_w > 0 else np.inf,
        "n_students_informative": int(np.isfinite(per_auc).sum()),
    }

    if skill is not None:
        skill = np.asarray(skill)
        key = _pair_key(student, skill)
        a_wk, n_wk, pa_k, pw_k = grouped_auc(y, p, key)
        out.update(
            auc_within_skill=a_wk,
            n_pairs_within_skill=n_wk,
            lambda_within_skill=n_wk / n_all if n_all > 0 else np.nan,
            # share of the *within-student* mass that is also same-skill
            share_skill_of_within=(n_wk / n_w) if n_w > 0 else np.nan,
            # independent units, not pairs: cells and students that contribute
            n_cells_within_skill=int(np.isfinite(pa_k).sum()),
            n_students_within_skill=_n_units(student, key, pa_k),
        )
    if item is not None:
        item = np.asarray(item)
        key = _pair_key(student, item)
        a_wi, n_wi, pa_i, pw_i = grouped_auc(y, p, key)
        out.update(
            auc_within_item=a_wi,
            n_pairs_within_item=n_wi,
            lambda_within_item=n_wi / n_all if n_all > 0 else np.nan,
            n_cells_within_item=int(np.isfinite(pa_i).sum()),
            n_students_within_item=_n_units(student, key, pa_i),
        )
    return out


def _n_units(student: np.ndarray, key: np.ndarray, per_auc: np.ndarray) -> int:
    """How many distinct *students* contribute at least one informative cell.

    Pair counts are not sample sizes: a component supported by millions of pairs
    may rest on a handful of exchangeable units.  This is the number that governs
    the width of a cluster bootstrap interval.
    """
    order = np.argsort(key, kind="mergesort")
    ks = key[order]
    cut = np.flatnonzero(np.diff(ks)) + 1
    starts = np.concatenate(([0], cut))
    good = np.isfinite(per_auc)
    if not good.any():
        return 0
    return int(np.unique(student[order[starts[good]]]).size)


def _pair_key(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Combine two integer-codeable arrays into one group key."""
    ac = np.unique(a, return_inverse=True)[1].astype(np.int64)
    bc = np.unique(b, return_inverse=True)[1].astype(np.int64)
    return ac * (bc.max() + 1) + bc


# --------------------------------------------------------------------------
# Uncertainty: bootstrap over *students* (the independent sampling unit)
# --------------------------------------------------------------------------
def cluster_bootstrap(
    y: np.ndarray,
    p: np.ndarray,
    student: np.ndarray,
    skill: np.ndarray | None = None,
    stat_keys: tuple[str, ...] = ("auc_pooled", "auc_within", "auc_between"),
    n_boot: int = 200,
    seed: int = 0,
) -> dict:
    """Percentile CIs from resampling *students* with replacement.

    Resampling students (rather than interactions) is the correct scheme here:
    interactions within a student are dependent, students are exchangeable.
    """
    rng = np.random.default_rng(seed)
    student = np.asarray(student)
    uniq, inv = np.unique(student, return_inverse=True)
    idx_by_student = [np.flatnonzero(inv == k) for k in range(uniq.shape[0])]
    S = len(idx_by_student)

    draws = {k: [] for k in stat_keys}
    for _ in range(n_boot):
        pick = rng.integers(0, S, size=S)
        idx = np.concatenate([idx_by_student[j] for j in pick])
        # relabel students so that a student drawn twice counts as two clusters
        lens = np.array([idx_by_student[j].shape[0] for j in pick])
        pseudo = np.repeat(np.arange(S), lens)
        try:
            d = decompose_auc(
                y[idx], p[idx], pseudo,
                skill=None if skill is None else np.asarray(skill)[idx],
            )
        except Exception:
            continue
        for k in stat_keys:
            draws[k].append(d.get(k, np.nan))

    out = {}
    for k, v in draws.items():
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            out[k] = (np.nan, np.nan)
        else:
            out[k] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return out


# --------------------------------------------------------------------------
# Downstream decision task: within-student triage
# --------------------------------------------------------------------------
def within_precision_at_k(
    y: np.ndarray, p: np.ndarray, student: np.ndarray, k: int = 5
) -> float:
    """Teacher-facing triage: for each student, flag the k interactions the
    model deems most likely to be answered incorrectly; report the fraction
    that really were incorrect, averaged over students (macro).

    This decision is made *conditional on a student*, so only the within-student
    ordering of predictions can influence it.  It is deliberately invariant to
    any monotone per-student recentering of the scores.
    """
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=np.float64)
    student = np.asarray(student)
    order, starts, stops = _group_slices(student)
    ys, ps = y[order], p[order]

    vals = []
    for a, b in zip(starts, stops):
        yy, pp = ys[a:b], ps[a:b]
        n_err = int((yy == 0).sum())
        if n_err == 0 or (b - a) <= k:
            continue  # no signal / nothing to choose
        sel = np.argsort(pp, kind="mergesort")[:k]  # k lowest predicted P(correct)
        vals.append(float((yy[sel] == 0).mean()))
    return float(np.mean(vals)) if vals else float("nan")


# --------------------------------------------------------------------------
# Vectorised within-cluster AUC (same result as grouped_auc, no Python loop)
# --------------------------------------------------------------------------
def grouped_auc_fast(y: np.ndarray, p: np.ndarray, g: np.ndarray) -> tuple[float, float]:
    """Pair-weighted within-group AUC, fully vectorised.

    Identical output to :func:`grouped_auc` (mid-ranks for ties, groups with a
    constant label contribute no pairs), but computed with sorts and
    ``bincount`` instead of a per-group Python loop -- which is what makes the
    cluster bootstrap tractable on the larger benchmarks.
    """
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    gi = np.unique(np.asarray(g), return_inverse=True)[1].astype(np.int64)
    G = int(gi.max()) + 1 if gi.size else 0
    if G == 0:
        return float("nan"), 0.0

    order = np.lexsort((p, gi))            # by group, then score
    gs, ps, ys = gi[order], p[order], y[order]

    n_g = np.bincount(gs, minlength=G)
    starts = np.concatenate(([0], np.cumsum(n_g)[:-1]))
    pos_in_g = np.arange(gs.size) - starts[gs]          # 0-based rank slot

    # mid-ranks: average the slot index over each (group, score) tie block
    new_block = np.empty(gs.size, dtype=bool)
    new_block[0] = True
    new_block[1:] = (gs[1:] != gs[:-1]) | (ps[1:] != ps[:-1])
    block_id = np.cumsum(new_block) - 1
    bsz = np.bincount(block_id)
    bsum = np.bincount(block_id, weights=pos_in_g + 1.0)
    rank = (bsum / bsz)[block_id]                       # mid-rank within group

    P_g = np.bincount(gs, weights=ys, minlength=G)
    N_g = n_g - P_g
    Rsum = np.bincount(gs, weights=rank * ys, minlength=G)

    w = P_g * N_g
    ok = w > 0
    if not ok.any():
        return float("nan"), 0.0
    auc_g = (Rsum[ok] - P_g[ok] * (P_g[ok] + 1.0) / 2.0) / w[ok]
    tot = float(w[ok].sum())
    return float((auc_g * w[ok]).sum() / tot), tot


def decompose_fast(y, p, student, skill=None) -> dict:
    """decompose_auc restricted to the quantities the bootstrap needs."""
    pooled, n_all = auc_mw(y, p)
    aw, n_w = grouped_auc_fast(y, p, student)
    lam = n_w / n_all if n_all > 0 else np.nan
    out = {"auc_pooled": pooled, "auc_within": aw,
           "auc_between": (pooled - lam * aw) / (1 - lam) if lam < 1 else np.nan,
           "lambda_within": lam}
    if skill is not None:
        awk, n_wk = grouped_auc_fast(y, p, _pair_key(student, np.asarray(skill)))
        out["auc_within_skill"] = awk
        out["lambda_within_skill"] = n_wk / n_all if n_all > 0 else np.nan
    return out