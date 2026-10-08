"""Causality regression test for outcome-dependent predictors.

The test perturbs one outcome at a time and asserts that every strictly earlier
prediction is unchanged.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import models_classic as MC  # noqa: E402


def toy(n_users=3, n_per=14, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        for t in range(n_per):
            rows.append((u, int(rng.integers(0, 6)), t,
                         int(rng.integers(0, 2)), int(rng.integers(0, 3))))
    return pd.DataFrame(rows, columns=["user_id", "item_id", "timestamp",
                                       "correct", "skill_id"])


def check(model_factory, name, meta=None):
    te = toy()
    meta = meta or {"n_items": 6, "n_skills": 3, "n_users": 3}
    tr = toy(seed=1)
    m = model_factory().fit(tr, meta)
    base = m.predict(te, meta)
    worst, where = 0.0, ""
    for row in range(len(te)):
        te2 = te.copy()
        te2.loc[row, "correct"] = 1 - te2.loc[row, "correct"]
        p2 = model_factory().fit(tr, meta).predict(te2, meta)
        # every prediction at or before `row` must be untouched
        d = np.abs(p2[: row + 1] - base[: row + 1]).max() if row >= 0 else 0.0
        if d > worst:
            worst, where = d, f"flipping outcome {row}"
    ok = worst < 1e-12
    print(f"[{'OK  ' if ok else 'FAIL'}] {name:22s} max change to a past "
          f"prediction = {worst:.2e} ({where})")
    return ok


if __name__ == "__main__":
    M = {"n_items": 6, "n_skills": 3, "n_users": 3}
    ok = True
    ok &= check(lambda: MC.FrozenAbility(k=10), "FrozenAbility", M)
    ok &= check(lambda: MC.FrozenIRT(k=10), "FrozenIRT", M)
    ok &= check(lambda: MC.RunningAbility(), "RunningAbility", M)
    ok &= check(lambda: MC.ItemMean(), "ItemMean", M)
    ok &= check(lambda: MC.PFA(), "PFA-shared", M)
    ok &= check(lambda: MC.BestLR(), "Best-LR", M)
    ok &= check(lambda: MC.BestLR_I(), "Best-LR+I", M)
    print("\nNO TARGET LEAKAGE" if ok else "\nLEAKAGE DETECTED")
    sys.exit(0 if ok else 1)