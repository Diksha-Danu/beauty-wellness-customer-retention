"""Step 3b (Task 4) - Exploratory data analysis and hypothesis tests (H1, H2)."""
import numpy as np, pandas as pd, seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from .common import PROC, update_json, save_fig, NAVY, CORAL, SAGE
from .features import add_features


def wald_logit(X, y):
    """Unpenalised logistic regression with Wald p-values (no statsmodels needed)."""
    Xm = np.column_stack([np.ones(len(X)), X])
    lg = LogisticRegression(C=1e10, max_iter=5000, fit_intercept=False).fit(Xm, y)
    b = lg.coef_[0]; p = 1 / (1 + np.exp(-Xm @ b)); W = p * (1 - p)
    se = np.sqrt(np.diag(np.linalg.inv((Xm * W[:, None]).T @ Xm)))
    return b, 2 * (1 - stats.norm.cdf(abs(b / se)))


def main():
    d = add_features(pd.read_csv(PROC / "clean.csv"))
    d.to_csv(PROC / "features.csv", index=False)
    R = {"churn_rate_pct": round(d.is_churned.mean() * 100, 1)}

    # H1: compatibility -> retention
    hi, lo = d[d.formula_match_score >= .6], d[d.formula_match_score < .6]
    r_hi, r_lo = 1 - hi.is_churned.mean(), 1 - lo.is_churned.mean()
    chi = stats.chi2_contingency(pd.crosstab(d.formula_match_score >= .6, d.is_churned))
    R["H1"] = dict(retention_high=round(r_hi * 100, 1), retention_low=round(r_lo * 100, 1),
                   relative_delta_pct=round((r_hi / r_lo - 1) * 100, 1), p_value=float(chi[1]))

    # H2: sentiment -> churn (reviewers only)
    rv = d[d.has_review == 1]
    b, pv = wald_logit(rv[["sentiment_polarity", "formula_match_score", "discount_reliance"]].values, rv.is_churned)
    R["H2"] = dict(coef=round(float(b[1]), 3), odds_ratio=round(float(np.exp(b[1])), 3), p_value=float(pv[1]), n=len(rv))
    R["churn_by_adverse_flag"] = d.groupby("adverse_reaction_flag").is_churned.mean().mul(100).round(1).to_dict()
    R["churn_by_match_band"] = d.groupby("match_band", observed=True).is_churned.mean().mul(100).round(1).to_dict()
    R["churn_by_skin_type"] = d.groupby("skin_type").is_churned.mean().mul(100).round(1).to_dict()

    # charts
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    mb = d.groupby("match_band", observed=True).is_churned.mean() * 100
    ax[0].bar(range(4), mb.values, color=[CORAL, "#F2CC8F", SAGE, "#5B9279"])
    ax[0].set_xticks(range(4)); ax[0].set_xticklabels(["Low", "Medium", "High", "Very high"], fontsize=8)
    ax[0].set_title("Churn % by formula-match band"); ax[0].set_ylim(0, 100)
    sk = d.groupby("skin_type").is_churned.mean().sort_values() * 100
    ax[1].barh(sk.index, sk.values, color=NAVY); ax[1].set_title("Churn % by skin type"); ax[1].set_xlim(0, 100)
    for k, c, l in [(0, SAGE, "Retained"), (1, CORAL, "Churned")]:
        sns.kdeplot(rv[rv.is_churned == k].sentiment_polarity, ax=ax[2], color=c, fill=True, alpha=.35, label=l)
    ax[2].set_title("Review sentiment by outcome"); ax[2].legend(frameon=False)
    save_fig("t4_eda1")

    cols = ["formula_match_score", "sensitivity_score", "recency_days", "frequency", "discount_reliance",
            "sentiment_polarity", "adverse_reaction_flag", "total_clv", "rfm_score", "mismatch_x_sensitivity", "is_churned"]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(d[cols].corr(), cmap="RdBu_r", center=0, annot=True, fmt=".2f", annot_kws={"size": 7}, ax=ax, cbar_kws={"shrink": .7})
    ax.set_title("Correlation matrix (Pearson)"); save_fig("t4_corr")
    R["corr_with_churn"] = d[cols].corr()["is_churned"].drop("is_churned").round(2).to_dict()

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    db = d.groupby(pd.cut(d.discount_reliance, [0, .2, .4, .6, 1], include_lowest=True), observed=True).is_churned.mean() * 100
    ax[0].bar(["0-.2", ".2-.4", ".4-.6", ".6-1"], db.values, color=CORAL); ax[0].set_title("Churn % by discount reliance"); ax[0].set_ylim(0, 100)
    fb = d.groupby("frequency").is_churned.mean().head(6) * 100
    ax[1].plot(fb.index, fb.values, marker="o", color=NAVY); ax[1].set_title("Churn % by order frequency"); ax[1].set_ylim(0, 100)
    save_fig("t4_eda2")
    update_json("eda_results.json", R)
    print(R["H1"], R["H2"])


if __name__ == "__main__":
    main()
