"""Stage 2: turn cached predictions into the full metric table.

Reads results/preds/*.npy written by run_stage1_fit.py and writes
results/metrics.csv (one row per dataset x model).

Usage:  python3 run_stage2_metrics.py [--boot N]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decomp import decompose_auc, cluster_bootstrap, within_precision_at_k  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED = os.path.join(ROOT, "results", "preds")

MODEL_ORDER = [
    "GlobalMean", "ItemMean", "SkillMean", "FrozenAbility", "FrozenIRT",
    "RunningAbility", "PFA-shared", "BKT", "Best-LR", "Best-LR+I",
    "DKT", "DKT+I", "SAKT", "SAKT+I",
]


def brier_nll_acc(y, p):
    p = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(np.sqrt(np.mean((p - y) ** 2))), float(nll), float(np.mean((p >= 0.5) == y))


def main(n_boot: int):
    rows = []
    for gt_file in sorted(glob.glob(os.path.join(PRED, "*__GT.npz"))):
        ds = os.path.basename(gt_file).split("__")[0]
        g = np.load(gt_file)
        y, u, sk, it = g["y"], g["user"], g["skill"], g["item"]
        for m in MODEL_ORDER:
            f = os.path.join(PRED, f"{ds}__{m}.npy")
            if not os.path.exists(f):
                continue
            p = np.load(f).astype(np.float64)
            d = decompose_auc(y, p, u, skill=sk, item=it)
            rmse, nll, acc = brier_nll_acc(y, p)
            r = dict(dataset=ds, model=m, n=len(y), **d,
                     rmse=rmse, nll=nll, acc=acc,
                     wP5=within_precision_at_k(y, p, u, k=5),
                     wP10=within_precision_at_k(y, p, u, k=10))
            if n_boot > 0:
                ci = cluster_bootstrap(y, p, u, skill=sk, n_boot=n_boot,
                                       stat_keys=("auc_pooled", "auc_within"))
                r["ci_pooled_lo"], r["ci_pooled_hi"] = ci["auc_pooled"]
                r["ci_within_lo"], r["ci_within_hi"] = ci["auc_within"]
            rows.append(r)
            print(f"{ds:18s} {m:14s} pooled={d['auc_pooled']:.4f} W={d['auc_within']:.4f} "
                  f"B={d['auc_between']:.4f} W|K={d['auc_within_skill']:.4f} "
                  f"W|I={d['auc_within_item']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    out = os.path.join(ROOT, "results", "metrics.csv")
    df.to_csv(out, index=False)
    print("\nwrote", out, df.shape)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=0)
    main(ap.parse_args().boot)