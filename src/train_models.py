"""Step 4 (Task 5) - Compare candidate models, tune Random Forest, diagnose overfitting, cluster customers.

Input : data/processed/features.csv
Output: models/rf_params.json, reports/model_results.json, figures t5_*.png
"""
import json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, silhouette_score
from .common import PROC, MODELS, SEED, update_json, save_fig, flowchart, NAVY, CORAL, SAND
from .features import make_xy


def load_split():
    d = pd.read_csv(PROC / "features.csv")
    X, y = make_xy(d)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=SEED)
    return d, X, y, Xtr, Xte, ytr, yte


def candidate_models():
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=.5)),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=30, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=10, max_features="sqrt", random_state=SEED, n_jobs=-1),
        "Gradient Boosting": HistGradientBoostingClassifier(learning_rate=.05, max_iter=200, max_depth=3, l2_regularization=1.0, random_state=SEED),
    }


def main():
    d, X, y, Xtr, Xte, ytr, yte = load_split()
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    R = {"split": dict(train=len(Xtr), test=len(Xte), n_features=X.shape[1])}

    cv = {}
    for name, m in candidate_models().items():
        auc = cross_val_score(m, Xtr, ytr, cv=skf, scoring="roc_auc")
        f1 = cross_val_score(m, Xtr, ytr, cv=skf, scoring="f1")
        cv[name] = dict(auc_mean=round(auc.mean(), 3), auc_sd=round(auc.std(), 3), f1=round(f1.mean(), 3), folds=auc.tolist())
    R["cv"] = cv

    gs = GridSearchCV(RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
                      {"max_depth": [4, 6, 8, None], "min_samples_leaf": [5, 10, 25]},
                      cv=skf, scoring="roc_auc").fit(Xtr, ytr)
    best = {k: ("None" if v is None else v) for k, v in gs.best_params_.items()}
    R["rf_best"] = dict(params=best, cv_auc=round(gs.best_score_, 3))
    (MODELS / "rf_params.json").write_text(json.dumps(best))

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.boxplot([cv[n]["folds"] for n in cv], label=[n.replace(" ", "\n") for n in cv], patch_artist=True,
               boxprops=dict(facecolor=SAND), medianprops=dict(color=NAVY))
    ax.set_ylabel("ROC-AUC (5-fold)"); ax.set_title("Candidate models: cross-validated ROC-AUC")
    save_fig("t5_cv")

    # over/under-fitting diagnostic
    depths = [2, 3, 4, 6, 8, 12, None]; tr, va = [], []
    for dp in depths:
        m = RandomForestClassifier(n_estimators=150, max_depth=dp, min_samples_leaf=3, random_state=SEED, n_jobs=-1)
        tr.append(roc_auc_score(ytr, m.fit(Xtr, ytr).predict_proba(Xtr)[:, 1]))
        va.append(cross_val_score(m, Xtr, ytr, cv=skf, scoring="roc_auc").mean())
    R["depth_curve"] = dict(depths=[str(x) for x in depths], train=[round(x, 3) for x in tr], val=[round(x, 3) for x in va])
    fig, ax = plt.subplots(figsize=(6.5, 3.6)); xl = [str(x) for x in depths]
    ax.plot(xl, tr, marker="o", color=CORAL, label="Training AUC"); ax.plot(xl, va, marker="s", color=NAVY, label="Validation AUC (CV)")
    ax.set_xlabel("Random Forest max_depth"); ax.set_ylabel("ROC-AUC"); ax.set_title("Over-/under-fitting diagnostic"); ax.legend(frameon=False)
    save_fig("t5_fit")

    # K-means segmentation
    sc = StandardScaler().fit_transform(d[["recency_days", "frequency", "spend_per_order", "discount_reliance", "formula_match_score", "sensitivity_score"]])
    ks = range(2, 8); inertia, sil = [], []
    for k in ks:
        km = KMeans(k, n_init=10, random_state=SEED).fit(sc)
        inertia.append(km.inertia_); sil.append(silhouette_score(sc, km.labels_, sample_size=2000, random_state=SEED))
    d["segment"] = KMeans(4, n_init=10, random_state=SEED).fit(sc).labels_
    seg = d.groupby("segment").agg(customers=("customer_id", "count"), churn=("is_churned", "mean"),
                                   avg_orders=("frequency", "mean"), avg_recency=("recency_days", "mean")).round(2)
    seg["churn"] = (seg.churn * 100).round(1)
    R["silhouette"] = [round(s, 3) for s in sil]; R["segments"] = seg.reset_index().to_dict("records")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
    ax[0].plot(list(ks), inertia, marker="o", color=NAVY); ax[0].set_title("Elbow (inertia)"); ax[0].set_xlabel("k")
    ax[1].plot(list(ks), sil, marker="o", color=CORAL); ax[1].set_title("Silhouette score"); ax[1].set_xlabel("k")
    save_fig("t5_cluster")
    flowchart([("Raw data", "events, reviews"), ("Clean", "Task 3 pipeline"), ("Features", "Task 4 RFM + NLP"), ("Model", "Random Forest"),
               ("Risk score", "0-1 per customer"), ("Trigger", "CRM workflow"), ("Measure", "A/B holdout"), ("Retrain", "monthly")],
              "t5_arch", "From data to retention action")
    update_json("model_results.json", R)
    print({k: (v["auc_mean"], v["auc_sd"]) for k, v in cv.items()}, R["rf_best"])


if __name__ == "__main__":
    main()
