"""Stage 3: tables, statistics and figures from results/metrics.csv."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load, meta  # noqa: E402
from decomp import decompose_auc  # noqa: E402
from models_classic import burn_in_mask  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
pd.set_option("display.width", 250)

DS_ORDER = ["assistments09", "assistments12", "assistments15", "assistments17",
            "algebra05", "bridge_algebra06", "spanish", "statics"]
MODEL_ORDER = ["GlobalMean", "ItemMean", "SkillMean", "FrozenAbility", "FrozenIRT",
               "RunningAbility", "PFA-shared", "BKT", "Best-LR", "Best-LR+I",
               "DKT", "DKT+I", "SAKT", "SAKT+I"]
MIN_PAIRS = 5000   # reliability gate for reporting a component


def budget_table(df):
    """Dataset-level 'AUC budget': model-free properties of the evaluation."""
    rows = []
    for ds in DS_ORDER:
        d = df[(df.dataset == ds) & (df.model == "GlobalMean")].iloc[0]
        tr, te = load(ds)
        S = te.user_id.nunique()
        n_s = te.groupby("user_id").size().to_numpy().astype(float)
        pi_s = te.groupby("user_id")["correct"].mean().to_numpy()
        kappa = (np.mean(n_s ** 2 * pi_s * (1 - pi_s)) /
                 (np.mean(n_s * pi_s) * np.mean(n_s * (1 - pi_s))))
        rows.append(dict(
            dataset=ds, n_test=int(d.n), students=S,
            items=int(te.item_id.nunique()), kcs=int(te.skill_id.nunique()),
            acc=float(te.correct.mean()),
            informative_students=int(d.n_students_informative),
            lam_W=d.lambda_within, S_eff=d.S_eff, one_over_S=1.0 / S, kappa=kappa,
            lam_WK=d.lambda_within_skill, lam_WI=d.lambda_within_item,
            n_pairs=d.n_pairs_total, n_W=d.n_pairs_within,
            n_WK=d.n_pairs_within_skill, n_WI=d.n_pairs_within_item,
        ))
    return pd.DataFrame(rows)


def rank_disagreement(df):
    """Kendall tau and top-1 disagreement between pooled and within rankings."""
    from scipy.stats import kendalltau, spearmanr
    out = []
    real = [m for m in MODEL_ORDER if m not in ("GlobalMean",)]
    for ds in DS_ORDER:
        d = df[(df.dataset == ds) & (df.model.isin(real))].set_index("model")
        a, w = d["auc_pooled"], d["auc_within"]
        wk = d["auc_within_skill"]
        tau = kendalltau(a, w).statistic
        tau_k = kendalltau(a, wk).statistic
        inv = sum(1 for i in range(len(a)) for j in range(i + 1, len(a))
                  if np.sign(a.iloc[i] - a.iloc[j]) != np.sign(w.iloc[i] - w.iloc[j]))
        out.append(dict(dataset=ds, tau_pooled_vs_W=tau, tau_pooled_vs_WK=tau_k,
                        inversions=inv, n_pairs_of_models=len(a) * (len(a) - 1) // 2,
                        best_pooled=a.idxmax(), best_W=w.idxmax(), best_WK=wk.idxmax(),
                        spearman_P5_pooled=spearmanr(a, d["wP5"]).statistic,
                        spearman_P5_W=spearmanr(w, d["wP5"]).statistic))
    return pd.DataFrame(out)


def headroom_table(df):
    rows = []
    for ds in DS_ORDER:
        d = df[df.dataset == ds].set_index("model")
        lam = d.loc["GlobalMean", "lambda_within"]
        best = d["auc_pooled"].drop("GlobalMean").idxmax()
        aw = d.loc[best, "auc_within"]
        rows.append(dict(dataset=ds, lam_W=lam, best_model=best,
                         auc_pooled=d.loc[best, "auc_pooled"], auc_W=aw,
                         headroom=lam * (1 - aw)))
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(os.path.join(RES, "metrics.csv"))
    df["model"] = pd.Categorical(df["model"], MODEL_ORDER, ordered=True)

    bt = budget_table(df)
    bt.to_csv(os.path.join(RES, "table_budget.csv"), index=False)
    print("\n=== TABLE 1: the AUC budget of each benchmark (model-free) ===")
    print(bt.to_string(index=False, float_format=lambda v: f"{v:,.6g}"))

    print("\n=== TABLE 2: components per dataset x model ===")
    for ds in DS_ORDER:
        d = df[df.dataset == ds].sort_values("model")
        nwi = d["n_pairs_within_item"].iloc[0]
        nwk = d["n_pairs_within_skill"].iloc[0]
        print(f"\n--- {ds}   (within-KC pairs={nwk:,.0f}, within-item pairs={nwi:,.0f}"
              f"{'  [item pairs too few: not interpreted]' if nwi < MIN_PAIRS else ''})")
        show = d[["model", "auc_pooled", "auc_within", "auc_between",
                  "auc_within_skill", "auc_within_item", "wP5", "rmse", "nll"]]
        print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    rd = rank_disagreement(df)
    rd.to_csv(os.path.join(RES, "table_rank_disagreement.csv"), index=False)
    print("\n=== TABLE 3: pooled vs within ranking disagreement ===")
    print(rd.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    hr = headroom_table(df)
    hr.to_csv(os.path.join(RES, "table_headroom.csv"), index=False)
    print("\n=== TABLE 4: tracing headroom lambda_W (1 - AUC_W) ===")
    print(hr.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

    # --- targeted repair: Best-LR -> Best-LR+I on reliable datasets ---
    print("\n=== TABLE 5: item-repetition repair (Best-LR -> Best-LR+I) ===")
    rows = []
    for ds in DS_ORDER:
        d = df[df.dataset == ds].set_index("model")
        if d.loc["Best-LR", "n_pairs_within_item"] < MIN_PAIRS:
            continue
        rows.append(dict(dataset=ds,
                         d_pooled=d.loc["Best-LR+I", "auc_pooled"] - d.loc["Best-LR", "auc_pooled"],
                         d_W=d.loc["Best-LR+I", "auc_within"] - d.loc["Best-LR", "auc_within"],
                         d_WI=d.loc["Best-LR+I", "auc_within_item"] - d.loc["Best-LR", "auc_within_item"],
                         WI_before=d.loc["Best-LR", "auc_within_item"],
                         WI_after=d.loc["Best-LR+I", "auc_within_item"]))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    # --- non-tracing reference, evaluated on the POST-BURN-IN window ---
    # The frozen baselines spend their first k observations estimating an ability;
    # they are only frozen afterwards.  Every model is therefore re-scored on the
    # same window t >= k so the comparison is like for like, and so that
    # FrozenAbility is exactly constant within a learner there (Prop. 5(i)).
    K_BURN = 10
    print(f"\n=== TABLE 7: non-tracing reference, evaluated on t >= {K_BURN} "
          f"(post burn-in, identical window for every model) ===")
    rows = []
    for ds in DS_ORDER:
        g = np.load(os.path.join(RES, "preds", f"{ds}__GT.npz"))
        y, u, sk, it = g["y"], g["user"], g["skill"], g["item"]
        m = burn_in_mask(u, K_BURN)
        sub = {}
        for mod in MODEL_ORDER:
            f = os.path.join(RES, "preds", f"{ds}__{mod}.npy")
            if not os.path.exists(f):
                continue
            p = np.load(f).astype(np.float64)
            sub[mod] = decompose_auc(y[m], p[m], u[m], skill=sk[m], item=it[m])
        best_model = max((k for k in sub if k != "GlobalMean"),
                         key=lambda k: sub[k]["auc_pooled"])
        best = sub[best_model]["auc_pooled"]
        rows.append(dict(
            dataset=ds, n_eval=int(m.sum()),
            FrozenAbility=sub["FrozenAbility"]["auc_pooled"],
            FrozenAbility_AUC_W=sub["FrozenAbility"]["auc_within"],
            FrozenIRT=sub["FrozenIRT"]["auc_pooled"],
            FrozenIRT_AUC_WI=sub["FrozenIRT"]["auc_within_item"],
            best_model=best_model, best_model_auc=best,
            share_FrozenAbility=(sub["FrozenAbility"]["auc_pooled"] - .5) / (best - .5),
            share_FrozenIRT=(sub["FrozenIRT"]["auc_pooled"] - .5) / (best - .5)))
    t7 = pd.DataFrame(rows)
    t7.to_csv(os.path.join(RES, "table_nontracing_share.csv"), index=False)
    print(t7.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"  median share recovered by FrozenIRT: {t7.share_FrozenIRT.median():.3f}")
    print(f"  Prop.5(i)  FrozenAbility AUC_W  on this window: "
          f"max deviation from 1/2 = {(t7.FrozenAbility_AUC_W - 0.5).abs().max():.2e}")
    print(f"  Prop.5(iii) FrozenIRT  AUC_W|I on this window: "
          f"max deviation from 1/2 = {(t7.FrozenIRT_AUC_WI - 0.5).abs().max():.2e}")

    # --- how often is a within component below chance? ---
    print("\n=== TABLE 6: below-chance within-item concordance (reliable datasets) ===")
    sub = df[df.n_pairs_within_item >= MIN_PAIRS]
    piv = sub.pivot_table(index="model", columns="dataset", values="auc_within_item",
                          observed=True)
    print(piv.to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()