"""Score customers with the saved model:  python -m src.predict data/processed/features.csv out.csv"""
import sys, joblib, pandas as pd, json
from .common import MODELS, REP
from .features import make_xy


def score(features_csv, out_csv):
    d = pd.read_csv(features_csv)
    X, _ = make_xy(d)
    model = joblib.load(MODELS / "best_rf.joblib")
    X = X.reindex(columns=model.feature_names_in_, fill_value=0)
    thr = json.loads((REP / "evaluation_results.json").read_text())["threshold"]
    out = d[["customer_id"]].assign(churn_risk=model.predict_proba(X)[:, 1])
    out["risk_band"] = pd.cut(out.churn_risk, [0, .4, thr, 1], labels=["low", "medium", "high"], include_lowest=True)
    out.to_csv(out_csv, index=False); print(out.risk_band.value_counts())


if __name__ == "__main__":
    score(sys.argv[1], sys.argv[2])
