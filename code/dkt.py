"""Deep Knowledge Tracing (Piech et al., NeurIPS 2015) in pure NumPy,
plus the proposed within-student ranking objective (DKT-W).

The container used for these experiments has no GPU and no access to a deep
learning framework, so the LSTM and its backward pass are implemented directly.
Correctness of every gradient is verified against central finite differences in
``tests/test_gradients.py``.

Model
-----
    x_t  = E[ 2*k_t + y_t ]                      (interaction embedding)
    h_t  = LSTM(x_t, h_{t-1}, c_{t-1})
    z_t  = <h_t, W_o[:, k_{t+1}]> + b_o[k_{t+1}]  (logit for the *next* item's KC)
    p_t  = sigma(z_t)

Objectives
----------
    L_BCE   = mean binary cross-entropy over observed steps          (DKT)
    L_RANK  = mean over sampled same-student (positive, negative) pairs of
              -log sigma(z_i - z_j)                                  (proposed)
    L       = L_BCE + alpha * L_RANK                                 (DKT-W)

``L_RANK`` is the standard convex pairwise surrogate for the Mann-Whitney
statistic (Herbrich et al. 2000; Burges et al. 2005; Cortes & Mohri 2004),
here restricted to pairs drawn from a single student's sequence so that it is a
surrogate for the *within-student* concordance AUC_W rather than the pooled AUC.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


# --------------------------------------------------------------------------
class DKT:
    def __init__(
        self,
        n_skills: int,
        hidden: int = 64,
        emb: int = 64,
        lr: float = 1e-2,
        l2: float = 1e-5,
        epochs: int = 15,
        batch: int = 32,
        maxlen: int = 200,
        n_items: int = 0,              # >0 enables a per-item easiness bias
        alpha: float = 0.0,            # weight of the within-student ranking loss
        pairs_per_seq: int = 64,
        pair_scope: str = "student",   # 'student' | 'skill'
        seed: int = 0,
        verbose: bool = False,
    ):
        self.K, self.H, self.D = n_skills, hidden, emb
        self.NI = int(n_items)
        self.lr, self.l2, self.epochs, self.B, self.L = lr, l2, epochs, batch, maxlen
        self.alpha, self.npair, self.pair_scope = alpha, pairs_per_seq, pair_scope
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        tag = "DKT+I" if self.NI > 0 else "DKT"
        self.name = tag if alpha == 0 else f"{tag}-W(a={alpha:g},{pair_scope})"
        self._init_params()

    # ---------------- parameters ----------------
    def _init_params(self):
        r = self.rng
        H, D, K = self.H, self.D, self.K
        s = lambda a, b: (r.standard_normal((a, b)) * (1.0 / np.sqrt(a))).astype(np.float64)
        self.p = {
            "E": (r.standard_normal((2 * K + 1, D)) * 0.1),
            "Wx": s(D, 4 * H),
            "Wh": s(H, 4 * H),
            "b": np.zeros(4 * H),
            "Wo": s(H, K),
            "bo": np.zeros(K),
        }
        if self.NI > 0:
            self.p["d"] = np.zeros(self.NI)
        self.p["b"][H : 2 * H] = 1.0                      # forget-gate bias = 1
        self.p["E"][0] = 0.0                              # padding row
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    # ---------------- batching ----------------
    def _chunks(self, seqs):
        """Split each student's sequence into fixed-length chunks.

        A sequence is ``(kc, y, row_index, item)`` or, when an ablation has
        replaced the information the model may condition on,
        ``(kc, y, row_index, item, kc_hist, y_hist)``.  The first four entries
        always define the *prediction targets* and are never altered by an
        ablation; the last two define the *input* stream fed to the recurrence.
        """
        out = []
        for sq in seqs:
            k, y, _idx, itm = sq[0], sq[1], sq[2], sq[3]
            kh, yh = (sq[4], sq[5]) if len(sq) >= 6 else (k, y)
            for a in range(0, len(k), self.L):
                kk = k[a : a + self.L]
                if len(kk) < 2:
                    continue
                out.append((kk, y[a : a + self.L], _idx[a : a + self.L],
                            itm[a : a + self.L], kh[a : a + self.L], yh[a : a + self.L]))
        return out

    def _pack(self, chunk_list):
        B, L = len(chunk_list), self.L
        Xi = np.zeros((B, L), dtype=np.int64)     # embedding index of interaction t
        Kt = np.zeros((B, L), dtype=np.int64)     # KC of the *target* (step t+1)
        Y = np.zeros((B, L))
        M = np.zeros((B, L))
        R = np.full((B, L), -1, dtype=np.int64)   # row id in the original frame
        It = np.zeros((B, L), dtype=np.int64)     # item of the *target* (step t+1)
        for b, (k, y, idx, itm, kh, yh) in enumerate(chunk_list):
            n = len(k) - 1
            Xi[b, :n] = 2 * kh[:n] + yh[:n] + 1
            Kt[b, :n] = k[1 : n + 1]
            Y[b, :n] = y[1 : n + 1]
            M[b, :n] = 1.0
            R[b, :n] = idx[1 : n + 1]
            It[b, :n] = itm[1 : n + 1]
        return Xi, Kt, Y, M, R, It

    # ---------------- forward ----------------
    def _forward(self, Xi, Kt, M, It=None):
        p = self.p
        B, L = Xi.shape
        H = self.H
        X = p["E"][Xi]                                    # (B,L,D)
        h = np.zeros((B, H))
        c = np.zeros((B, H))
        cache = {"X": X, "h": np.zeros((B, L, H)), "c": np.zeros((B, L, H)),
                 "hprev": np.zeros((B, L, H)), "cprev": np.zeros((B, L, H)),
                 "i": np.zeros((B, L, H)), "f": np.zeros((B, L, H)),
                 "g": np.zeros((B, L, H)), "o": np.zeros((B, L, H)),
                 "tc": np.zeros((B, L, H))}
        for t in range(L):
            cache["hprev"][:, t] = h
            cache["cprev"][:, t] = c
            z = X[:, t] @ p["Wx"] + h @ p["Wh"] + p["b"]
            i = sigmoid(z[:, :H]); f = sigmoid(z[:, H:2 * H])
            g = np.tanh(z[:, 2 * H:3 * H]); o = sigmoid(z[:, 3 * H:])
            c_new = f * c + i * g
            tc = np.tanh(c_new)
            h_new = o * tc
            m = M[:, t][:, None]
            h = m * h_new + (1 - m) * h                  # freeze state on padding
            c = m * c_new + (1 - m) * c
            cache["i"][:, t], cache["f"][:, t] = i, f
            cache["g"][:, t], cache["o"][:, t] = g, o
            cache["tc"][:, t], cache["h"][:, t], cache["c"][:, t] = tc, h, c
        Hf = cache["h"].reshape(-1, H)
        Kf = Kt.reshape(-1)
        Z = np.einsum("nh,nh->n", Hf, p["Wo"][:, Kf].T) + p["bo"][Kf]
        if self.NI > 0 and It is not None:
            Z = Z + p["d"][It.reshape(-1)]
        return Z.reshape(B, L), cache

    # ---------------- loss + backward ----------------
    def _rank_pairs(self, Y, M, Kt):
        """Sample within-sequence (positive, negative) index pairs."""
        B, L = Y.shape
        bi, pi, ni = [], [], []
        for b in range(B):
            valid = np.flatnonzero(M[b] > 0)
            if valid.size < 2:
                continue
            yv = Y[b, valid]
            pos, neg = valid[yv == 1], valid[yv == 0]
            if pos.size == 0 or neg.size == 0:
                continue
            if self.pair_scope == "skill":
                kk = Kt[b]
                common = np.intersect1d(kk[pos], kk[neg])
                if common.size == 0:
                    continue
                sel = self.rng.choice(common, size=min(self.npair, common.size * 4))
                for s_ in sel:
                    pp = pos[kk[pos] == s_]; nn = neg[kk[neg] == s_]
                    if pp.size and nn.size:
                        bi.append(b)
                        pi.append(self.rng.choice(pp))
                        ni.append(self.rng.choice(nn))
                continue
            m = min(self.npair, 4 * min(pos.size, neg.size))
            bi.extend([b] * m)
            pi.extend(self.rng.choice(pos, m))
            ni.extend(self.rng.choice(neg, m))
        return np.asarray(bi, int), np.asarray(pi, int), np.asarray(ni, int)

    def _backward(self, Z, cache, Xi, Kt, Y, M, It=None):
        p = self.p
        B, L = Z.shape
        H, D = self.H, self.D
        n_obs = max(M.sum(), 1.0)

        P = sigmoid(Z)
        dZ = (P - Y) * M / n_obs
        loss = -(M * (Y * np.log(np.clip(P, 1e-12, 1)) +
                      (1 - Y) * np.log(np.clip(1 - P, 1e-12, 1)))).sum() / n_obs

        rank_loss = 0.0
        if self.alpha > 0:
            bi, pi, ni = self._rank_pairs(Y, M, Kt)
            if bi.size:
                d = Z[bi, pi] - Z[bi, ni]
                sg = sigmoid(d)
                rank_loss = -np.log(np.clip(sg, 1e-12, 1)).mean()
                w = self.alpha * (sg - 1.0) / bi.size
                np.add.at(dZ, (bi, pi), w)
                np.add.at(dZ, (bi, ni), -w)

        g = {k: np.zeros_like(v) for k, v in p.items()}
        Hf = cache["h"].reshape(-1, H)
        Kf = Kt.reshape(-1)
        dZf = dZ.reshape(-1)
        S = sparse.csr_matrix((dZf, (np.arange(dZf.size), Kf)), shape=(dZf.size, self.K))
        g["Wo"] = np.asarray(S.T @ Hf).T
        g["bo"] = np.bincount(Kf, weights=dZf, minlength=self.K)
        if self.NI > 0 and It is not None:
            g["d"] = np.bincount(It.reshape(-1), weights=dZf, minlength=self.NI)
        dH = (dZf[:, None] * p["Wo"][:, Kf].T).reshape(B, L, H)

        dh = np.zeros((B, H))
        dc = np.zeros((B, H))
        dX = np.zeros_like(cache["X"])
        for t in range(L - 1, -1, -1):
            m = M[:, t][:, None]
            dh_t = dH[:, t] + dh
            dh_eff = m * dh_t
            dc_eff = m * dc
            o, tc = cache["o"][:, t], cache["tc"][:, t]
            i, f, gg = cache["i"][:, t], cache["f"][:, t], cache["g"][:, t]
            cprev = cache["cprev"][:, t]
            do = dh_eff * tc
            dcn = dc_eff + dh_eff * o * (1 - tc * tc)
            di = dcn * gg
            dg = dcn * i
            df = dcn * cprev
            dcprev = dcn * f
            dz = np.concatenate(
                [di * i * (1 - i), df * f * (1 - f), dg * (1 - gg * gg), do * o * (1 - o)],
                axis=1,
            )
            g["Wx"] += cache["X"][:, t].T @ dz
            g["Wh"] += cache["hprev"][:, t].T @ dz
            g["b"] += dz.sum(0)
            dX[:, t] = dz @ p["Wx"].T
            dh = dz @ p["Wh"].T + (1 - m) * dh_t
            dc = dcprev + (1 - m) * dc
        np.add.at(g["E"], Xi.reshape(-1), dX.reshape(-1, D))
        g["E"][0] = 0.0
        return loss + self.alpha * rank_loss, g

    # ---------------- optimisation ----------------
    def _adam(self, g):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k in self.p:
            gk = g[k] + self.l2 * self.p[k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * gk
            self.v[k] = b2 * self.v[k] + (1 - b2) * gk * gk
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            self.p[k] -= self.lr * mh / (np.sqrt(vh) + eps)
        for pad in ("E", "E_int"):          # padding embedding row stays at zero
            if pad in self.p:
                self.p[pad][0] = 0.0

    def fit_seqs(self, seqs, val=None):
        ch = self._chunks(seqs)
        for ep in range(self.epochs):
            order = self.rng.permutation(len(ch))
            tot, nb = 0.0, 0
            for a in range(0, len(order), self.B):
                bl = [ch[j] for j in order[a : a + self.B]]
                Xi, Kt, Y, M, _, It = self._pack(bl)
                Z, cache = self._forward(Xi, Kt, M, It)
                loss, g = self._backward(Z, cache, Xi, Kt, Y, M, It)
                self._adam(g)
                tot += loss
                nb += 1
            if self.verbose:
                print(f"  epoch {ep + 1}/{self.epochs} loss={tot / max(nb, 1):.4f}", flush=True)
        return self

    def predict_seqs(self, seqs, n_rows: int, prior: float = 0.5):
        out = np.full(n_rows, prior)
        ch = self._chunks(seqs)
        for a in range(0, len(ch), self.B):
            bl = ch[a : a + self.B]
            Xi, Kt, Y, M, R, It = self._pack(bl)
            Z, _ = self._forward(Xi, Kt, M, It)
            P = sigmoid(Z)
            sel = R >= 0
            out[R[sel]] = P[sel]
        return out