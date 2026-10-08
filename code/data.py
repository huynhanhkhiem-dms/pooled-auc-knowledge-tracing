"""Loading and causal feature construction for the knowledge-tracing datasets.

Datasets are the preprocessed benchmarks released with
Gervet, Koedinger, Schneider & Mitchell (2020), "When is Deep Learning the
Best Approach to Knowledge Tracing?", Journal of Educational Data Mining 12(3).

Each file is a TSV with columns
    user_id  item_id  timestamp  correct  skill_id
already sorted by student and time, and already split into disjoint sets of
students (a *student-level* split, i.e. every test student is unseen).
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

DATASETS = [
    "assistments09",
    "assistments12",
    "assistments15",
    "assistments17",
    "algebra05",
    "bridge_algebra06",
    "spanish",
    "statics",
]

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "benchmarks")


DATA_NPZ = os.path.join(os.path.dirname(DATA_ROOT), "data_npz")
_COLS = ["user_id", "item_id", "timestamp", "correct", "skill_id"]


def _read_split(name: str, split: str, root: str) -> pd.DataFrame:
    """Read one split from the TSV release, or from the compact npz mirror.

    The public repository retrieves the original preprocessed TSV files through
    ``fetch_data.sh``.  An optional local ``data_npz/`` mirror is also
    supported for compact offline use; when both are absent, the loader explains
    how to retrieve the pinned benchmark release.
    """
    csv = os.path.join(root, name, f"preprocessed_data_{split}.csv")
    if os.path.exists(csv):
        return pd.read_csv(csv, sep="\t")
    npz = os.path.join(DATA_NPZ, name, f"{split}.npz")
    if os.path.exists(npz):
        with np.load(npz) as z:
            return pd.DataFrame({c: z[c] for c in _COLS})
    raise FileNotFoundError(
        f"no data for {name}/{split}: expected {csv} or {npz}. "
        f"Run ./fetch_data.sh to download the benchmarks."
    )


def load(name: str, root: str = DATA_ROOT):
    """Return (train_df, test_df) with contiguous ids shared across splits."""
    tr = _read_split(name, "train", root)
    te = _read_split(name, "test", root)
    tr["split"], te["split"] = 0, 1
    df = pd.concat([tr, te], ignore_index=True)

    for c in ("user_id", "item_id", "skill_id"):
        df[c] = np.unique(df[c].to_numpy(), return_inverse=True)[1]
    df["correct"] = df["correct"].astype(np.int8)

    # stable chronological order inside each student
    df["_o"] = np.arange(len(df))
    df = df.sort_values(["user_id", "timestamp", "_o"], kind="mergesort").drop(columns="_o")
    df = df.reset_index(drop=True)
    return df[df.split == 0].reset_index(drop=True), df[df.split == 1].reset_index(drop=True)


def meta(tr: pd.DataFrame, te: pd.DataFrame) -> dict:
    both = pd.concat([tr, te])
    return {
        "n_items": int(both.item_id.max()) + 1,
        "n_skills": int(both.skill_id.max()) + 1,
        "n_users": int(both.user_id.max()) + 1,
    }


# --------------------------------------------------------------------------
# Causal count features (everything is strictly "past only")
# --------------------------------------------------------------------------
_COUNT_CACHE: dict = {}


def _fingerprint(df: pd.DataFrame, n_skills: int) -> bytes:
    """Content hash of the columns the causal counts depend on.

    Keying the cache on ``id(df)`` was unsafe: CPython reuses the address of a
    freed frame, so a later frame with different contents could collide with a
    stale entry and silently receive the wrong features.  ``tests/test_leakage.py``
    exposed this.  Hashing the contents costs one pass and removes the hazard.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(str(n_skills).encode())
    for c in ("user_id", "item_id", "skill_id", "correct"):
        a = np.ascontiguousarray(df[c].to_numpy())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.digest()


def causal_counts(df: pd.DataFrame, n_skills: int) -> dict[str, np.ndarray]:
    """Cached wrapper around :func:`_causal_counts` (the scan is O(n) but in Python)."""
    key = _fingerprint(df, n_skills)
    if key not in _COUNT_CACHE:
        if len(_COUNT_CACHE) > 8:            # keep the cache small
            _COUNT_CACHE.clear()
        _COUNT_CACHE[key] = _causal_counts(df, n_skills)
    return _COUNT_CACHE[key]


def _causal_counts(df: pd.DataFrame, n_skills: int) -> dict[str, np.ndarray]:
    """Counts of the student's prior attempts / successes, overall and per skill.

    Feature at row t uses only rows < t of the same student, so the features are
    legitimate at prediction time.  This reproduces the feature set of the
    Best-LR model of Gervet et al. (2020), itself built on PFA (Pavlik, Cen &
    Koedinger, 2009) and DAS3H (Choffin et al., 2019).
    """
    u = df.user_id.to_numpy()
    s = df.skill_id.to_numpy()
    y = df.correct.to_numpy().astype(np.int64)
    n = len(df)

    tot_att = np.zeros(n, dtype=np.float64)
    tot_suc = np.zeros(n, dtype=np.float64)
    sk_att = np.zeros(n, dtype=np.float64)
    sk_suc = np.zeros(n, dtype=np.float64)
    it_att = np.zeros(n, dtype=np.float64)
    it_suc = np.zeros(n, dtype=np.float64)
    it = df.item_id.to_numpy()
    ia: dict[int, int] = {}
    isu: dict[int, int] = {}

    cur_u = -1
    ta = ts_ = 0
    sa = np.zeros(n_skills, dtype=np.int64)
    ss = np.zeros(n_skills, dtype=np.int64)
    touched: list[int] = []

    for i in range(n):
        if u[i] != cur_u:
            cur_u = u[i]
            ta = ts_ = 0
            for k in touched:
                sa[k] = 0
                ss[k] = 0
            touched = []
            ia = {}
            isu = {}
        k = s[i]
        tot_att[i] = ta
        tot_suc[i] = ts_
        sk_att[i] = sa[k]
        sk_suc[i] = ss[k]
        j = it[i]
        it_att[i] = ia.get(j, 0)
        it_suc[i] = isu.get(j, 0)
        ia[j] = it_att[i] + 1
        isu[j] = it_suc[i] + y[i]
        if sa[k] == 0 and ss[k] == 0:
            touched.append(k)
        ta += 1
        ts_ += y[i]
        sa[k] += 1
        ss[k] += y[i]
    return {
        "tot_att": tot_att,
        "tot_suc": tot_suc,
        "tot_fail": tot_att - tot_suc,
        "sk_att": sk_att,
        "sk_suc": sk_suc,
        "sk_fail": sk_att - sk_suc,
        "it_att": it_att,
        "it_suc": it_suc,
        "it_fail": it_att - it_suc,
    }


def seqs(df: pd.DataFrame, key_item: str = "skill_id"):
    """Group a dataframe into per-student arrays (kc, label, row index, item)."""
    u = df.user_id.to_numpy()
    k = df[key_item].to_numpy()
    itm = df["item_id"].to_numpy()
    y = df.correct.to_numpy().astype(np.int8)
    cuts = np.flatnonzero(np.diff(u)) + 1
    starts = np.concatenate(([0], cuts))
    stops = np.concatenate((cuts, [len(df)]))
    out = []
    for a, b in zip(starts, stops):
        out.append((k[a:b], y[a:b], np.arange(a, b), itm[a:b]))
    return out