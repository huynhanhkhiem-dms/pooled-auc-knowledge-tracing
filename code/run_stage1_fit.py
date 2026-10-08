"""Stage 1: fit every model on every dataset and cache the test-set predictions.

Predictions are written to results/preds/<dataset>__<model>.npy so that the
metric stage can be re-run without re-fitting anything.

Usage:  python3 run_stage1_fit.py [dataset ...]
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import load, meta, seqs                                # noqa: E402
from dkt import DKT                                              # noqa: E402
from sakt import SAKT                                            # noqa: E402
import models_classic as MC                                      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED = os.path.join(ROOT, "results", "preds")
os.makedirs(PRED, exist_ok=True)

# smallest first so that partial results are useful early
ORDER = [
    "statics", "assistments09", "spanish", "assistments15",
    "algebra05", "bridge_algebra06", "assistments17", "assistments12",
]

DKT_KW = dict(hidden=64, emb=64, lr=8e-3, epochs=10, batch=32, maxlen=200, l2=1e-5)
SAKT_KW = dict(d_model=64, d_ff=128, lr=3e-3, epochs=10, batch=32, maxlen=100, l2=1e-5)


def classic_specs():
    return [
        ("GlobalMean", lambda M: MC.GlobalMean()),
        ("ItemMean", lambda M: MC.ItemMean()),
        ("SkillMean", lambda M: MC.SkillMean()),
        ("FrozenAbility", lambda M: MC.FrozenAbility(k=10)),
        ("FrozenIRT", lambda M: MC.FrozenIRT(k=10)),
        ("RunningAbility", lambda M: MC.RunningAbility()),
        ("PFA-shared", lambda M: MC.PFA()),
        ("BKT", lambda M: MC.BKT()),
        ("Best-LR", lambda M: MC.BestLR()),
        ("Best-LR+I", lambda M: MC.BestLR_I()),
    ]


def dkt_specs(M, seed=0):
    K, NI = M["n_skills"], M["n_items"]
    return [
        ("DKT", lambda: DKT(K, seed=seed, **DKT_KW)),
        ("DKT+I", lambda: DKT(K, n_items=NI, seed=seed, **DKT_KW)),
        ("SAKT", lambda: SAKT(K, seed=seed, **SAKT_KW)),
        ("SAKT+I", lambda: SAKT(K, n_items=NI, seed=seed, **SAKT_KW)),
    ]


def main(which):
    log = os.path.join(ROOT, "results", "stage1_log.json")
    done = json.load(open(log)) if os.path.exists(log) else {}
    for ds in which:
        t_ds = time.time()
        tr, te = load(ds)
        M = meta(tr, te)
        prior = float(tr.correct.mean())
        print(f"\n=== {ds}: train={len(tr)} test={len(te)} "
              f"items={M['n_items']} skills={M['n_skills']}", flush=True)

        for name, mk in classic_specs():
            key = f"{ds}__{name}"
            f = os.path.join(PRED, key + ".npy")
            if os.path.exists(f):
                continue
            t0 = time.time()
            p = mk(M).fit(tr, M).predict(te, M)
            np.save(f, p.astype(np.float32))
            done[key] = round(time.time() - t0, 1)
            print(f"  {name:16s} {done[key]:7.1f}s", flush=True)
            json.dump(done, open(log, "w"), indent=1)

        str_, ste = seqs(tr), seqs(te)
        for name, mk in dkt_specs(M):
            key = f"{ds}__{name}"
            f = os.path.join(PRED, key + ".npy")
            if os.path.exists(f):
                continue
            t0 = time.time()
            m = mk()
            m.fit_seqs(str_)
            p = m.predict_seqs(ste, len(te), prior)
            np.save(f, p.astype(np.float32))
            done[key] = round(time.time() - t0, 1)
            print(f"  {name:16s} {done[key]:7.1f}s", flush=True)
            json.dump(done, open(log, "w"), indent=1)

        # ground truth + grouping keys, saved once
        gt = os.path.join(PRED, f"{ds}__GT.npz")
        if not os.path.exists(gt):
            np.savez_compressed(
                gt,
                y=te.correct.to_numpy().astype(np.int8),
                user=te.user_id.to_numpy().astype(np.int32),
                skill=te.skill_id.to_numpy().astype(np.int32),
                item=te.item_id.to_numpy().astype(np.int32),
            )
        print(f"=== {ds} finished in {time.time() - t_ds:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ORDER)