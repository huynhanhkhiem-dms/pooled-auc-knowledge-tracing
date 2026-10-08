"""Finite-difference verification of every analytic gradient in dkt.py,
and algebraic verification of the AUC decomposition identity in decomp.py.

Run:  python3 tests/test_gradients.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/code")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dkt import DKT                      # noqa: E402
from decomp import decompose_auc, auc_mw  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402


def make_toy(rng, n_students=6, K=5):
    seqs = []
    r = 0
    for _ in range(n_students):
        n = rng.integers(4, 12)
        k = rng.integers(0, K, size=n)
        y = rng.integers(0, 2, size=n).astype(np.int8)
        seqs.append((k, y, np.arange(r, r + n), rng.integers(0, 9, size=n)))
        r += n
    return seqs, r


def loss_of(model, Xi, Kt, Y, M, It):
    Z, cache = model._forward(Xi, Kt, M, It)
    loss, _ = model._backward(Z, cache, Xi, Kt, Y, M, It)
    return loss


def check(alpha, pair_scope, n_items=0, tol=2e-5):
    rng = np.random.default_rng(0)
    seqs, _ = make_toy(rng)
    m = DKT(n_skills=5, hidden=7, emb=6, maxlen=12, alpha=alpha, n_items=n_items,
            pair_scope=pair_scope, pairs_per_seq=8, seed=3)
    ch = m._chunks(seqs)
    Xi, Kt, Y, M, _, It = m._pack(ch)

    # analytic
    m.rng = np.random.default_rng(11)          # freeze pair sampling
    Z, cache = m._forward(Xi, Kt, M, It)
    _, g = m._backward(Z, cache, Xi, Kt, Y, M, It)

    worst = 0.0
    for name in ["E", "Wx", "Wh", "b", "Wo", "bo"] + (["d"] if m.NI > 0 else []):
        A = m.p[name]
        flat = A.reshape(-1)
        idx = np.random.default_rng(5).choice(flat.size, size=min(12, flat.size), replace=False)
        for j in idx:
            if name == "E" and j < A.shape[1]:
                continue                        # padding row is pinned to 0
            old = flat[j]
            eps = 1e-6
            flat[j] = old + eps
            m.rng = np.random.default_rng(11)
            lp = loss_of(m, Xi, Kt, Y, M, It)
            flat[j] = old - eps
            m.rng = np.random.default_rng(11)
            lm = loss_of(m, Xi, Kt, Y, M, It)
            flat[j] = old
            num = (lp - lm) / (2 * eps)
            ana = g[name].reshape(-1)[j]
            denom = max(1.0, abs(num), abs(ana))
            worst = max(worst, abs(num - ana) / denom)
    status = "OK " if worst < tol else "FAIL"
    print(f"[{status}] gradients  alpha={alpha} scope={pair_scope:8s} item_bias={n_items>0} max rel err = {worst:.3e}")
    return worst < tol


def check_decomposition():
    rng = np.random.default_rng(1)
    n = 4000
    u = rng.integers(0, 40, size=n)
    sk = rng.integers(0, 6, size=n)
    it = rng.integers(0, 20, size=n)
    y = (rng.random(n) < 0.5 + 0.3 * np.sin(u)).astype(np.int8)
    p = rng.random(n) + 0.4 * y

    d = decompose_auc(y, p, u, skill=sk, item=it)
    lhs = d["auc_pooled"]
    rhs = d["lambda_within"] * d["auc_within"] + (1 - d["lambda_within"]) * d["auc_between"]
    err = abs(lhs - rhs)
    print(f"[{'OK ' if err < 1e-12 else 'FAIL'}] identity  |AUC - (lam*W + (1-lam)*B)| = {err:.2e}")

    a1, _ = auc_mw(y, p)
    a2 = roc_auc_score(y, p)
    print(f"[{'OK ' if abs(a1 - a2) < 1e-12 else 'FAIL'}] auc_mw vs sklearn        = {abs(a1 - a2):.2e}")

    # a model that is constant within each student must have AUC_W == 1/2
    const = np.array([np.mean(y[u == uu]) for uu in u])
    d2 = decompose_auc(y, const, u)
    print(f"[{'OK ' if abs(d2['auc_within'] - 0.5) < 1e-12 else 'FAIL'}] "
          f"Prop.4 constant-per-student AUC_W = {d2['auc_within']:.12f}")

    # a model that is a function of (student, skill) only must have AUC_{W|K} == 1/2
    key = u * 100 + sk
    cs = np.array([np.mean(y[key == kk]) for kk in key])
    d3 = decompose_auc(y, cs, u, skill=sk)
    print(f"[{'OK ' if abs(d3['auc_within_skill'] - 0.5) < 1e-12 else 'FAIL'}] "
          f"Prop.5 (student,skill)-measurable AUC_W|K = {d3['auc_within_skill']:.12f}")
    return err < 1e-12


if __name__ == "__main__":
    ok = True
    ok &= check(0.0, "student")
    ok &= check(0.7, "student")
    ok &= check(0.7, "skill")
    ok &= check(0.7, "student", n_items=9)
    ok &= check(0.0, "student", n_items=9)
    ok &= check_decomposition()
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok else 1)