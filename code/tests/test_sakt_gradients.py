"""Finite-difference verification of every analytic gradient in sakt.py."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sakt import SAKT  # noqa: E402


def make_toy(rng, n_students=6, K=5, n_items=9):
    seqs = []
    r = 0
    for _ in range(n_students):
        n = int(rng.integers(4, 12))
        k = rng.integers(0, K, size=n)
        y = rng.integers(0, 2, size=n).astype(np.int8)
        seqs.append((k, y, np.arange(r, r + n), rng.integers(0, n_items, size=n)))
        r += n
    return seqs


def loss_of(m, Xi, Kt, Y, M, It):
    Z, cache = m._forward(Xi, Kt, M, It)
    loss, _ = m._backward(Z, cache, Xi, Kt, Y, M, It)
    return loss


def check(alpha=0.0, n_items=0, pair_scope="student", tol=2e-5):
    rng = np.random.default_rng(0)
    seqs = make_toy(rng)
    m = SAKT(n_skills=5, d_model=8, d_ff=12, maxlen=12, alpha=alpha,
             n_items=n_items, pair_scope=pair_scope, pairs_per_seq=8, seed=3)
    # break the symmetry of the zero-initialised output head
    m.p["w_out"] = rng.standard_normal(m.D_MODEL) * 0.3
    m.p["bo"] = rng.standard_normal(m.K) * 0.2
    if n_items:
        m.p["d"] = rng.standard_normal(n_items) * 0.2
    ch = m._chunks(seqs)
    Xi, Kt, Y, M, _, It = m._pack(ch)

    m.rng = np.random.default_rng(11)
    Z, cache = m._forward(Xi, Kt, M, It)
    _, g = m._backward(Z, cache, Xi, Kt, Y, M, It)

    names = ["E_int", "E_exe", "E_pos", "g1", "b1", "W1", "bff1",
             "W2", "bff2", "g2", "b2", "w_out", "bo"] + (["d"] if n_items else [])
    worst, where = 0.0, ""
    for name in names:
        flat = m.p[name].reshape(-1)
        idx = np.random.default_rng(5).choice(flat.size, size=min(10, flat.size),
                                              replace=False)
        for j in idx:
            if name == "E_int" and j < m.p[name].shape[1]:
                continue                       # padding row pinned to zero
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
            rel = abs(num - ana) / max(1.0, abs(num), abs(ana))
            if rel > worst:
                worst, where = rel, name
    ok = worst < tol
    print(f"[{'OK ' if ok else 'FAIL'}] SAKT gradients alpha={alpha} "
          f"item_bias={n_items > 0} scope={pair_scope:8s} "
          f"max rel err = {worst:.3e} ({where})")
    return ok


def check_causality():
    """A prediction at step t must not depend on any interaction at step > t."""
    rng = np.random.default_rng(2)
    seqs = make_toy(rng, n_students=3)
    m = SAKT(n_skills=5, d_model=8, d_ff=12, maxlen=12, n_items=9, seed=1)
    m.p["w_out"] = rng.standard_normal(m.D_MODEL) * 0.3
    ch = m._chunks(seqs)
    Xi, Kt, Y, M, _, It = m._pack(ch)
    Z0, _ = m._forward(Xi, Kt, M, It)
    t = 3
    Xi2 = Xi.copy()
    Xi2[:, t + 1:] = 1 + (Xi2[:, t + 1:] + 3) % (2 * m.K)   # scramble the future
    Z1, _ = m._forward(Xi2, Kt, M, It)
    ok = np.allclose(Z0[:, :t + 1], Z1[:, :t + 1], atol=1e-12)
    print(f"[{'OK ' if ok else 'FAIL'}] SAKT causality: predictions up to step "
          f"{t} unchanged when the future is scrambled "
          f"(max diff {np.abs(Z0[:, :t + 1] - Z1[:, :t + 1]).max():.2e})")
    return ok


if __name__ == "__main__":
    ok = True
    ok &= check(0.0)
    ok &= check(0.0, n_items=9)
    ok &= check(1.5, n_items=9)
    ok &= check(1.5, n_items=9, pair_scope="skill")
    ok &= check_causality()
    print("\nALL SAKT CHECKS PASSED" if ok else "\nSOME SAKT CHECKS FAILED")
    sys.exit(0 if ok else 1)