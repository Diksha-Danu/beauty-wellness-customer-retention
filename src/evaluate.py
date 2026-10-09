"""Step 5 (Task 6) - Final evaluation on the sealed test set + business simulation.

Input : data/processed/features.csv, models/rf_params.json
Output: models/best_rf.joblib, reports/evaluation_results.json, figures t6_*.png
"""
import json
import numpy as np, pandas as pd, seaborn as sns
import matplotlib.pyplot as plt, joblib
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, confusion_matrix,
                             accuracy_score, precision_score, recall_score, f1_score, brier_score_loss)
from .common import MODELS, SEED, update_json, save_fig, flowchart, NAVY, CORAL, SAGE, SAND, GREY
from .train_models import load_split, candidate_models

# Illustrative business assumptions - REPLACE with real figures
COST_PER_CONTACT, MARGIN_PER_SAVE = 6, 45
SAVE_RATE_TARGETED, SAVE_RATE_BLANKET, TOP_SHARE = .25, .10, .30


def main():
    d, X, y, Xtr, Xte, ytr, yte = load_split()
    params = {k: (None if v == "None" else v) for k, v in json.loads((MODELS / "rf_params.json").read_text()).items()}
    rf = RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1, **params)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)

    # threshold chosen on out-of-fold TRAINING predictions only (Youden's J)
    oof = cross_val_predict(rf, Xtr, ytr, cv=skf, method="predict_proba")[:, 1]
    ths = np.linspace(.2, .8, 61)
    youden = [recall_score(ytr, oof >= t) + recall_score(1 - ytr, oof < t) - 1 for t in ths]
    thr = float(ths[int(np.argmax(youden))])

    rf.fit(Xtr, ytr); joblib.dump(rf, MODELS / "best_rf.joblib")
    p = rf.predict_proba(Xte)[:, 1]; pred = (p >= thr).astype(int); yt = yte.values
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    R = {"threshold": round(thr, 2), "test": dict(
        roc_auc=round(roc_auc_score(yte, p), 3), pr_auc=round(average_precision_score(yte, p), 3),
        accuracy=round(accuracy_score(yte, pred), 3), precision=round(precision_score(yte, pred), 3),
        recall=round(recall_score(yte, pred), 3), f1=round(f1_score(yte, pred), 3), brier=round(brier_score_loss(yte, p), 3),
        base_rate=round(yte.mean(), 3), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))}

    # benchmark models on the same test set
    R["test_auc_all_models"] = {n: round(roc_auc_score(yte, m.fit(Xtr, ytr).predict_proba(Xte)[:, 1]), 3)
                                for n, m in candidate_models().items() if n != "Random Forest"}
    R["test_auc_all_models"]["Random Forest (tuned)"] = R["test"]["roc_auc"]

    # bootstrap CI for AUC
    rs = np.random.default_rng(1)
    bs = [roc_auc_score(yt[i], p[i]) for i in (rs.integers(0, len(yt), len(yt)) for _ in range(500))]
    R["auc_95ci"] = [round(np.percentile(bs, 2.5), 3), round(np.percentile(bs, 97.5), 3)]

    # ---- charts ----
    fig, ax = plt.subplots(1, 2, figsize=(10, 4)); fpr, tpr, _ = roc_curve(yte, p)
    ax[0].plot(fpr, tpr, color=NAVY, lw=2, label=f"Random Forest (AUC = {R['test']['roc_auc']})")
    ax[0].plot([0, 1], [0, 1], "--", color=GREY, label="Random guess")
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (hold-out test set)"); ax[0].legend(frameon=False, loc="lower right")
    pr, rc, _ = precision_recall_curve(yte, p)
    ax[1].plot(rc, pr, color=CORAL, lw=2, label=f"AP = {R['test']['pr_auc']}"); ax[1].axhline(yte.mean(), ls="--", color=GREY, label=f"Baseline = {yte.mean():.2f}")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve", ylim=(0, 1.02)); ax[1].legend(frameon=False)
    save_fig("t6_roc")

    fig, ax = plt.subplots(figsize=(4.3, 3.8))
    sns.heatmap(confusion_matrix(yte, pred), annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"], annot_kws={"size": 14})
    ax.set(xlabel="Predicted", ylabel="Actual", title=f"Confusion matrix (threshold {thr:.2f})"); save_fig("t6_cm")

    pi = permutation_importance(rf, Xte, yte, scoring="roc_auc", n_repeats=8, random_state=SEED, n_jobs=-1)
    imp = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False).head(10)
    R["top_features"] = imp.round(3).to_dict()
    fig, ax = plt.subplots(figsize=(7, 4)); imp[::-1].plot.barh(color=NAVY, ax=ax)
    ax.set_title("Top 10 drivers of churn (permutation importance, AUC drop)"); save_fig("t6_imp")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8)); fp_, mp_ = calibration_curve(yte, p, n_bins=10)
    ax[0].plot(mp_, fp_, marker="o", color=NAVY, label="Model"); ax[0].plot([0, 1], [0, 1], "--", color=GREY, label="Perfect")
    ax[0].set(xlabel="Mean predicted probability", ylabel="Observed churn rate", title="Calibration"); ax[0].legend(frameon=False)
    order = np.argsort(-p); dec = np.array_split(yt[order], 10); lift = [x.mean() / yt.mean() for x in dec]
    R["lift_by_decile"] = [round(x, 2) for x in lift]; R["churn_rate_by_decile_pct"] = [round(x.mean() * 100, 1) for x in dec]
    ax[1].bar(range(1, 11), lift, color=[CORAL if i < 3 else SAND for i in range(10)]); ax[1].axhline(1, ls="--", color=GREY)
    ax[1].set(xlabel="Risk decile (1 = highest predicted risk)", ylabel="Lift vs average", title="Lift by risk decile"); save_fig("t6_calib")

    te = d.loc[Xte.index]; segs = {}
    for g in ["skin_type", "skin_tone"]:
        for v, gr in te.groupby(g):
            if len(gr) > 40 and gr.is_churned.nunique() == 2:
                segs[f"{g}={v}"] = round(roc_auc_score(gr.is_churned, p[Xte.index.get_indexer(gr.index)]), 3)
    R["auc_by_segment"] = segs
    fig, ax = plt.subplots(figsize=(6.5, 3.6)); s = pd.Series(segs).sort_values()
    ax.barh(s.index, s.values, color=SAGE); ax.axvline(R["test"]["roc_auc"], ls="--", color=NAVY); ax.set_xlim(.5, 1)
    ax.set_title("ROC-AUC by customer segment (dashed = overall)"); save_fig("t6_seg")

    # ---- business simulation ----
    n = len(p); top = int(n * TOP_SHARE); caught = int(yt[order[:top]].sum())
    net_t = caught * SAVE_RATE_TARGETED * MARGIN_PER_SAVE - top * COST_PER_CONTACT
    net_b = yt.sum() * SAVE_RATE_BLANKET * MARGIN_PER_SAVE - n * COST_PER_CONTACT
    R["business"] = dict(test_customers=n, contacted_targeted=top, leavers_reached=caught,
                         capture_pct=round(caught / yt.sum() * 100, 1), net_targeted=round(net_t), net_blanket=round(net_b),
                         assumptions=dict(cost=COST_PER_CONTACT, margin=MARGIN_PER_SAVE, save_targeted=SAVE_RATE_TARGETED, save_blanket=SAVE_RATE_BLANKET))
    fig, ax = plt.subplots(figsize=(5.5, 3.4)); ax.bar(["Blanket offer\n(everyone)", "Targeted offer\n(top 30% risk)"], [net_b, net_t], color=[GREY, SAGE])
    ax.axhline(0, color="#555", lw=.8); ax.set_ylabel("Net benefit, USD (test cohort)"); ax.set_title("Illustrative campaign economics"); save_fig("t6_biz")
    flowchart([("1 Hold-out", "20% test set sealed"), ("2 Cross-validate", "5-fold stratified"), ("3 Tune", "grid search on AUC"), ("4 Threshold", "Youden J on OOF"),
               ("5 Test once", "AUC, PR, F1, Brier"), ("6 Segment audit", "AUC by skin type/tone"), ("7 Business sim", "cost vs benefit"), ("8 Monitor", "drift & retrain")],
              "t6_flow", "Validation and performance-measurement plan")
    update_json("evaluation_results.json", R)
    print(json.dumps({k: R[k] for k in ["threshold", "test", "auc_95ci", "business"]}, indent=1))


if __name__ == "__main__":
    main()
