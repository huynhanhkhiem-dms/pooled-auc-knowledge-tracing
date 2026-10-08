"""Non-neural knowledge-tracing / student models.

Every model exposes ``fit(train_df, meta) -> self`` and
``predict(test_df, meta) -> np.ndarray`` returning P(correct) for each row,
using **only** information available before that row (student-level split
means test students are never seen in training; within a test student, only
the strict prefix of the sequence is used).

Model families
--------------
Static, provably non-tracing
    GlobalMean, ItemMean, SkillMean, FrozenAbility, FrozenIRT
Classical tracing
    RunningAbility, PFA (Pavlik, Cen & Koedinger 2009),
    BestLR (Gervet et al. 2020), BKT (Corbett & Anderson 1995)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

from data import causal_counts

PHI = lambda x: np.log1p(x)          # noqa: E731  (standard DAS3H/Best-LR transform)
EPS = 1e-6


def _clip(p):
    return np.clip(p, EPS, 1 - EPS)


# --------------------------------------------------------------------------
# Static baselines
# --------------------------------------------------------------------------
class GlobalMean:
    name = "GlobalMean"

    def fit(self, tr, meta):
        self.m = float(tr.correct.mean())
        return self

    def predict(self, te, meta):
        return np.full(len(te), self.m)


class _KeyMean:
    """Smoothed historical accuracy of a column, learned on the training students."""

    key = None
    alpha = 5.0

    def fit(self, tr, meta):
        self.m = float(tr.correct.mean())
        g = tr.groupby(self.key)["correct"].agg(["sum", "count"])
        n_key = meta["n_items"] if self.key == "item_id" else meta["n_skills"]
        tab = np.full(n_key, self.m)
        idx = g.index.to_numpy()
        tab[idx] = (g["sum"].to_numpy() + self.alpha * self.m) / (g["count"].to_numpy() + self.alpha)
        self.tab = tab
        return self

    def predict(self, te, meta):
        return self.tab[te[self.key].to_numpy()]


class ItemMean(_KeyMean):
    name, key = "ItemMean", "item_id"


class SkillMean(_KeyMean):
    name, key = "SkillMean", "skill_id"


class FrozenAbility:
    """Estimate a learner's ability from their first ``k`` responses, then freeze.

    **Strictly causal.**  The prediction at position ``t`` uses only outcomes
    strictly before ``t``:

        t <  k :  smoothed running mean of y[0..t-1]   (still warming up)
        t >= k :  frozen at the smoothed mean of y[0..k-1]

    ``tests/test_leakage.py`` asserts that perturbing any outcome leaves every
    strictly earlier prediction bit-identical.

    On the evaluation window ``t >= k`` the prediction is constant within a
    learner, so on that window the model has within-learner AUC exactly 1/2 by
    Proposition~5(i).  It is the reference point for "ability ranking without
    knowledge tracing", and the comparison must therefore be made on that window
    for every model (see ``burn_in_mask``).
    """

    def __init__(self, k=10, alpha=5.0):
        self.k, self.alpha = k, alpha
        self.name = f"FrozenAbility(k={k})"

    def fit(self, tr, meta):
        self.m = float(tr.correct.mean())
        return self

    def _theta(self, te):
        u = te.user_id.to_numpy()
        y = te.correct.to_numpy().astype(np.float64)
        n = len(te)
        out = np.empty(n)
        i = 0
        while i < n:
            j = i
            while j < n and u[j] == u[i]:
                j += 1
            # strict prefix sums: c[t] = sum of y[i .. i+t-1]
            c = np.concatenate(([0.0], np.cumsum(y[i:j])))
            t = np.arange(j - i)
            use = np.minimum(t, self.k)             # how many past outcomes are visible
            out[i:j] = (c[use] + self.alpha * self.m) / (use + self.alpha)
            i = j
        return out

    def predict(self, te, meta):
        return _clip(self._theta(te))


class FrozenIRT(FrozenAbility):
    """Frozen ability + static item easiness: a Rasch-type model with no learning.

    On the evaluation window ``t >= k`` the prediction varies within a learner
    only through item identity, so its concordance on same-learner same-item
    pairs is exactly 1/2 (Proposition~5(iii)).  Strictly causal, as for
    :class:`FrozenAbility`.
    """

    def __init__(self, k=10, alpha=5.0):
        super().__init__(k, alpha)
        self.name = f"FrozenIRT(k={k})"

    def fit(self, tr, meta):
        self.m = float(tr.correct.mean())
        it = ItemMean().fit(tr, meta)
        self.item_logit = np.log(_clip(it.tab) / (1 - _clip(it.tab)))
        self.item_logit -= np.log(_clip(self.m) / (1 - _clip(self.m)))
        return self

    def predict(self, te, meta):
        th = _clip(self._theta(te))
        z = np.log(th / (1 - th)) + self.item_logit[te.item_id.to_numpy()]
        return _clip(1.0 / (1.0 + np.exp(-z)))


class RunningAbility:
    """Smoothed running mean of the student's own past correctness. No item info."""

    name = "RunningAbility"

    def __init__(self, alpha=5.0):
        self.alpha = alpha

    def fit(self, tr, meta):
        self.m = float(tr.correct.mean())
        return self

    def predict(self, te, meta):
        c = causal_counts(te, meta["n_skills"])
        return _clip((c["tot_suc"] + self.alpha * self.m) / (c["tot_att"] + self.alpha))


# --------------------------------------------------------------------------
# Logistic-regression family
# --------------------------------------------------------------------------
class _SparseLR:
    C = 1.0
    name = "LR"

    def _design(self, df, meta):
        raise NotImplementedError

    def fit(self, tr, meta):
        X = self._design(tr, meta)
        self.clf = LogisticRegression(C=self.C, solver="lbfgs", max_iter=1000, tol=1e-5)
        self.clf.fit(X, tr.correct.to_numpy())
        return self

    def predict(self, te, meta):
        return _clip(self.clf.predict_proba(self._design(te, meta))[:, 1])


def _onehot(idx, n, rows):
    return sparse.csr_matrix(
        (np.ones(rows, dtype=np.float32), (np.arange(rows), idx)), shape=(rows, n)
    )


class PFA(_SparseLR):
    """Shared-slope PFA-style logistic baseline.

    This constrained variant uses a skill-specific intercept but shares the
    transformed prior-success and prior-failure slopes across skills.  The
    manuscript labels it ``PFA-shared`` to distinguish it from canonical PFA,
    which estimates skill-specific success/failure slopes.
    """

    name = "PFA-shared"

    def _design(self, df, meta):
        n = len(df)
        c = causal_counts(df, meta["n_skills"])
        S = _onehot(df.skill_id.to_numpy(), meta["n_skills"], n)
        dense = np.column_stack([PHI(c["sk_suc"]), PHI(c["sk_fail"])]).astype(np.float32)
        return sparse.hstack([S, sparse.csr_matrix(dense)], format="csr")


class BestLR(_SparseLR):
    """Best-LR of Gervet et al. (2020): item + skill one-hots and phi-counts."""

    name = "Best-LR"

    def _design(self, df, meta):
        n = len(df)
        c = causal_counts(df, meta["n_skills"])
        I = _onehot(df.item_id.to_numpy(), meta["n_items"], n)
        S = _onehot(df.skill_id.to_numpy(), meta["n_skills"], n)
        dense = np.column_stack(
            [
                PHI(c["tot_att"]), PHI(c["tot_suc"]), PHI(c["tot_fail"]),
                PHI(c["sk_att"]), PHI(c["sk_suc"]), PHI(c["sk_fail"]),
            ]
        ).astype(np.float32)
        return sparse.hstack([I, S, sparse.csr_matrix(dense)], format="csr")


class BestLR_I(_SparseLR):
    """Best-LR augmented with *item-level* repetition counts.

    Motivated directly by the decomposition: AUC_{W|I} isolates repeated
    attempts by the same student on the same item, and no feature in Best-LR
    varies across such a pair except the global/skill counts.  Adding
    phi(prior attempts / successes / failures *on this very item*) is the
    minimal change that makes the model's prediction sensitive to the
    within-item practice signal.
    """

    name = "Best-LR+I"

    def _design(self, df, meta):
        n = len(df)
        c = causal_counts(df, meta["n_skills"])
        I = _onehot(df.item_id.to_numpy(), meta["n_items"], n)
        S = _onehot(df.skill_id.to_numpy(), meta["n_skills"], n)
        dense = np.column_stack(
            [
                PHI(c["tot_att"]), PHI(c["tot_suc"]), PHI(c["tot_fail"]),
                PHI(c["sk_att"]), PHI(c["sk_suc"]), PHI(c["sk_fail"]),
                PHI(c["it_att"]), PHI(c["it_suc"]), PHI(c["it_fail"]),
            ]
        ).astype(np.float32)
        return sparse.hstack([I, S, sparse.csr_matrix(dense)], format="csr")


# --------------------------------------------------------------------------
# Bayesian Knowledge Tracing
# --------------------------------------------------------------------------
class BKT:
    """Corbett & Anderson (1995) BKT, one 4-parameter HMM per knowledge component.

    Parameters (L0, T, G, S) are fitted per skill by direct maximisation of the
    observed-data likelihood (L-BFGS-B on the logit scale) with the customary
    G, S <= 0.3 identifiability constraint (Baker, Corbett & Aleven, 2008).
    """

    name = "BKT"

    def __init__(self, max_sk_seqs=4000, seed=0):
        self.max_sk_seqs = max_sk_seqs
        self.rng = np.random.default_rng(seed)

    # ---- likelihood over a padded batch of sequences ----
    @staticmethod
    def _nll(theta, Y, M):
        L0 = 1 / (1 + np.exp(-theta[0]))
        T = 1 / (1 + np.exp(-theta[1]))
        G = 0.3 / (1 + np.exp(-theta[2]))
        S = 0.3 / (1 + np.exp(-theta[3]))
        n, Tmax = Y.shape
        L = np.full(n, L0)
        ll = np.zeros(n)
        for t in range(Tmax):
            p = L * (1 - S) + (1 - L) * G
            p = np.clip(p, 1e-9, 1 - 1e-9)
            y = Y[:, t]
            m = M[:, t]
            ll += m * (y * np.log(p) + (1 - y) * np.log(1 - p))
            num = np.where(y == 1, L * (1 - S), L * S)
            den = np.where(y == 1, p, 1 - p)
            post = np.where(m > 0, num / den, L)
            L = post + (1 - post) * T
        return -ll.sum()

    def _skill_batches(self, df, n_skills):
        u = df.user_id.to_numpy()
        s = df.skill_id.to_numpy()
        y = df.correct.to_numpy().astype(np.int8)
        order = np.lexsort((np.arange(len(df)), u, s))
        s_o, u_o, y_o = s[order], u[order], y[order]
        out = {}
        i = 0
        n = len(df)
        while i < n:
            j = i
            while j < n and s_o[j] == s_o[i]:
                j += 1
            uu, yy = u_o[i:j], y_o[i:j]
            cuts = np.flatnonzero(np.diff(uu)) + 1
            seqs = np.split(yy, cuts)
            out[int(s_o[i])] = seqs
            i = j
        return out

    def fit(self, tr, meta):
        self.m = float(tr.correct.mean())
        n_skills = meta["n_skills"]
        self.par = np.tile(np.array([0.4, 0.1, 0.2, 0.1]), (n_skills, 1))  # sane defaults
        batches = self._skill_batches(tr, n_skills)
        for k, seqs in batches.items():
            if len(seqs) > self.max_sk_seqs:
                idx = self.rng.choice(len(seqs), self.max_sk_seqs, replace=False)
                seqs = [seqs[i] for i in idx]
            Tmax = min(max(len(s) for s in seqs), 100)
            Y = np.zeros((len(seqs), Tmax), dtype=np.float64)
            M = np.zeros((len(seqs), Tmax), dtype=np.float64)
            for r, s in enumerate(seqs):
                L = min(len(s), Tmax)
                Y[r, :L] = s[:L]
                M[r, :L] = 1.0
            if M.sum() < 20:
                continue
            th0 = np.array([0.0, -2.0, -0.85, -1.5])
            res = minimize(self._nll, th0, args=(Y, M), method="L-BFGS-B",
                           options={"maxiter": 60})
            th = res.x
            self.par[k] = [
                1 / (1 + np.exp(-th[0])), 1 / (1 + np.exp(-th[1])),
                0.3 / (1 + np.exp(-th[2])), 0.3 / (1 + np.exp(-th[3])),
            ]
        return self

    def predict(self, te, meta):
        u = te.user_id.to_numpy()
        s = te.skill_id.to_numpy()
        y = te.correct.to_numpy().astype(np.int8)
        n = len(te)
        out = np.empty(n)
        state: dict[int, float] = {}
        cur_u = -1
        for i in range(n):
            if u[i] != cur_u:
                cur_u = u[i]
                state = {}
            k = int(s[i])
            L0, T, G, S = self.par[k]
            L = state.get(k, L0)
            p = L * (1 - S) + (1 - L) * G
            out[i] = p
            p = min(max(p, 1e-9), 1 - 1e-9)
            post = (L * (1 - S)) / p if y[i] == 1 else (L * S) / (1 - p)
            if not np.isfinite(post):
                post = L
            post = min(max(post, 0.0), 1.0)
            state[k] = post + (1 - post) * T
        return _clip(out)


def burn_in_mask(user: np.ndarray, k: int = 10) -> np.ndarray:
    """Boolean mask selecting the interactions at position ``>= k`` within a learner.

    The frozen baselines spend their first ``k`` observations estimating an
    ability and are only "frozen" afterwards.  Comparing them against tracing
    models on the full test set would score them on a window where they are still
    warming up, so every model must be compared on the same post-burn-in window.
    Rows are assumed to be grouped by learner and ordered in time, which
    :func:`data.load` guarantees.
    """
    user = np.asarray(user)
    n = user.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    new = np.empty(n, dtype=bool)
    new[0] = True
    np.not_equal(user[1:], user[:-1], out=new[1:])
    starts = np.flatnonzero(new)
    pos = np.arange(n) - np.repeat(starts, np.diff(np.append(starts, n)))
    return pos >= k


CLASSIC_MODELS = [
    GlobalMean, ItemMean, SkillMean, FrozenAbility, FrozenIRT,
    RunningAbility, PFA, BestLR, BestLR_I, BKT,
]