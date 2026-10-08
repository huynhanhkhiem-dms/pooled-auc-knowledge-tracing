"""SAKT -- a self-attentive knowledge tracing model (Pandey & Karypis, EDM 2019)
implemented in pure NumPy with hand-derived gradients.

The evaluation container has no GPU and no deep-learning framework, so this is a
faithful but compact re-implementation of the published architecture: a single
causal self-attention block in which the query is the embedding of the exercise
being predicted and the keys/values are the embeddings of the learner's past
interactions, followed by a position-wise feed-forward network, each sublayer
wrapped in a residual connection and layer normalisation.

    Q_t    = E_exe[k_{t+1}]                            (query = target exercise)
    KV_s   = E_int[2 k_s + y_s] + E_pos[s]             (keys/values = history)
    A      = softmax( Q K^T / sqrt(d) + causal mask )
    H1     = LayerNorm( A V + Q )
    H2     = LayerNorm( FFN(H1) + H1 )
    z_t    = <H2_t, w> + b_o[k_{t+1}] ( + d[i_{t+1}] )

Every analytic gradient is checked against central finite differences in
``tests/test_sakt_gradients.py``.

The optional within-student pairwise ranking term of the accompanying paper is
supported through the same ``alpha`` / ``pair_scope`` interface as ``dkt.DKT``.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from dkt import DKT, sigmoid


def _softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def _layernorm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xh = xc * inv
    return xh * g + b, (xh, inv)


def _layernorm_backward(dy, cache, g):
    xh, inv = cache
    d = xh.shape[-1]
    dg = (dy * xh).reshape(-1, d).sum(0)
    db = dy.reshape(-1, d).sum(0)
    dxh = dy * g
    dx = inv / d * (d * dxh - dxh.sum(-1, keepdims=True)
                    - xh * (dxh * xh).sum(-1, keepdims=True))
    return dx, dg, db


class SAKT(DKT):
    """Self-attentive KT.  Reuses DKT's batching, Adam and ranking-loss plumbing."""

    def __init__(self, n_skills, d_model=64, d_ff=128, maxlen=100, **kw):
        self.D_MODEL, self.D_FF = d_model, d_ff
        kw.setdefault("hidden", d_model)
        kw.setdefault("emb", d_model)
        kw["maxlen"] = maxlen
        super().__init__(n_skills, **kw)
        tag = "SAKT+I" if self.NI > 0 else "SAKT"
        self.name = tag if self.alpha == 0 else f"{tag}-W(a={self.alpha:g},{self.pair_scope})"

    # ---------------- parameters ----------------
    def _init_params(self):
        r = self.rng
        d, dff, K = self.D_MODEL, self.D_FF, self.K
        sc = lambda a, b: (r.standard_normal((a, b)) * (1.0 / np.sqrt(a)))
        self.p = {
            "E_int": r.standard_normal((2 * K + 1, d)) * 0.1,
            "E_exe": r.standard_normal((K + 1, d)) * 0.1,
            "E_pos": r.standard_normal((self.L, d)) * 0.1,
            "g1": np.ones(d), "b1": np.zeros(d),
            "W1": sc(d, dff), "bff1": np.zeros(dff),
            "W2": sc(dff, d), "bff2": np.zeros(d),
            "g2": np.ones(d), "b2": np.zeros(d),
            "w_out": np.zeros(d),
            "bo": np.zeros(K),
        }
        if self.NI > 0:
            self.p["d"] = np.zeros(self.NI)
        self.p["E_int"][0] = 0.0
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    # ---------------- forward ----------------
    def _forward(self, Xi, Kt, M, It=None):
        p = self.p
        B, L = Xi.shape
        d = self.D_MODEL
        Q = p["E_exe"][Kt]                                  # (B,L,d)
        KV = p["E_int"][Xi] + p["E_pos"][None, :L, :]       # (B,L,d)
        KV = KV * M[:, :, None]                             # padded steps carry nothing

        scale = 1.0 / np.sqrt(d)
        S = np.einsum("btd,bsd->bts", Q, KV) * scale
        # step t predicts step t+1, so it may attend to history slots 0..t
        causal = np.tril(np.ones((L, L), dtype=bool))
        valid = causal[None] & (M[:, None, :] > 0)
        S = np.where(valid, S, -1e9)
        A = _softmax(S, axis=-1)
        A = np.where(valid, A, 0.0)
        O = np.einsum("bts,bsd->btd", A, KV)

        R1 = O + Q
        H1, c1 = _layernorm(R1, p["g1"], p["b1"])
        F1 = H1 @ p["W1"] + p["bff1"]
        Fr = np.maximum(F1, 0.0)
        F2 = Fr @ p["W2"] + p["bff2"]
        R2 = F2 + H1
        H2, c2 = _layernorm(R2, p["g2"], p["b2"])

        Kf = Kt.reshape(-1)
        Z = H2.reshape(-1, d) @ p["w_out"] + p["bo"][Kf]
        if self.NI > 0 and It is not None:
            Z = Z + p["d"][It.reshape(-1)]
        cache = dict(Q=Q, KV=KV, A=A, valid=valid, O=O, H1=H1, c1=c1,
                     F1=F1, Fr=Fr, H2=H2, c2=c2, scale=scale, M=M)
        return Z.reshape(B, L), cache

    # ---------------- backward ----------------
    def _backward(self, Z, cache, Xi, Kt, Y, M, It=None):
        p = self.p
        B, L = Z.shape
        d, dff = self.D_MODEL, self.D_FF
        n_obs = max(M.sum(), 1.0)

        P = sigmoid(Z)
        dZ = (P - Y) * M / n_obs
        loss = -(M * (Y * np.log(np.clip(P, 1e-12, 1)) +
                      (1 - Y) * np.log(np.clip(1 - P, 1e-12, 1)))).sum() / n_obs

        rank_loss = 0.0
        if self.alpha > 0:
            bi, pi, ni = self._rank_pairs(Y, M, Kt)
            if bi.size:
                dd = Z[bi, pi] - Z[bi, ni]
                sg = sigmoid(dd)
                rank_loss = -np.log(np.clip(sg, 1e-12, 1)).mean()
                w = self.alpha * (sg - 1.0) / bi.size
                np.add.at(dZ, (bi, pi), w)
                np.add.at(dZ, (bi, ni), -w)

        g = {k: np.zeros_like(v) for k, v in p.items()}
        Kf = Kt.reshape(-1)
        dZf = dZ.reshape(-1)
        H2f = cache["H2"].reshape(-1, d)

        g["w_out"] = H2f.T @ dZf
        g["bo"] = np.bincount(Kf, weights=dZf, minlength=self.K)
        if self.NI > 0 and It is not None:
            g["d"] = np.bincount(It.reshape(-1), weights=dZf, minlength=self.NI)
        dH2 = (dZf[:, None] * p["w_out"][None, :]).reshape(B, L, d)

        dR2, g["g2"], g["b2"] = _layernorm_backward(dH2, cache["c2"], p["g2"])
        dF2 = dR2
        g["W2"] = cache["Fr"].reshape(-1, dff).T @ dF2.reshape(-1, d)
        g["bff2"] = dF2.reshape(-1, d).sum(0)
        dFr = dF2 @ p["W2"].T
        dF1 = dFr * (cache["F1"] > 0)
        g["W1"] = cache["H1"].reshape(-1, d).T @ dF1.reshape(-1, dff)
        g["bff1"] = dF1.reshape(-1, dff).sum(0)
        dH1 = dF1 @ p["W1"].T + dR2

        dR1, g["g1"], g["b1"] = _layernorm_backward(dH1, cache["c1"], p["g1"])
        dO = dR1
        dQ = dR1.copy()

        A, KV = cache["A"], cache["KV"]
        dA = np.einsum("btd,bsd->bts", dO, KV)
        dKV = np.einsum("bts,btd->bsd", A, dO)
        # softmax backward
        dS = A * (dA - (dA * A).sum(-1, keepdims=True))
        dS = np.where(cache["valid"], dS, 0.0)
        dQ += np.einsum("bts,bsd->btd", dS, KV) * cache["scale"]
        dKV += np.einsum("bts,btd->bsd", dS, cache["Q"]) * cache["scale"]

        dKV = dKV * M[:, :, None]
        np.add.at(g["E_exe"], Kt.reshape(-1), dQ.reshape(-1, d))
        np.add.at(g["E_int"], Xi.reshape(-1), dKV.reshape(-1, d))
        g["E_pos"][:L] = dKV.sum(0)
        g["E_int"][0] = 0.0
        return loss + self.alpha * rank_loss, g