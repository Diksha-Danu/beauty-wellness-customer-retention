"""Step 3a (Task 4) - Feature engineering shared by EDA, training and evaluation."""
import numpy as np, pandas as pd

CAT = ["skin_type", "skin_tone", "age_bracket"]
NUM = ["sensitivity_score", "recency_days", "frequency", "spend_per_order", "total_clv", "discount_reliance",
       "formula_match_score", "has_review", "adverse_reaction_flag", "sentiment_polarity", "keyword_texture_clash",
       "subjectivity", "rfm_score", "order_cadence", "mismatch_x_sensitivity", "discount_x_mismatch",
       "neg_sentiment", "oily_texture_clash"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["recency_score"] = pd.qcut(-d.recency_days, 5, labels=False, duplicates="drop") + 1
    d["frequency_score"] = pd.cut(d.frequency, [0, 1, 2, 3, 5, 99], labels=False) + 1
    d["monetary_score"] = pd.qcut(d.total_clv.rank(method="first"), 5, labels=False) + 1
    d["rfm_score"] = d.recency_score + d.frequency_score + d.monetary_score
    d["order_cadence"] = (d.recency_days / d.frequency).round(2)
    d["mismatch_x_sensitivity"] = (1 - d.formula_match_score) * d.sensitivity_score
    d["discount_x_mismatch"] = d.discount_reliance * (1 - d.formula_match_score)
    d["neg_sentiment"] = np.maximum(0, -d.sentiment_polarity)
    d["log_spend"] = np.log1p(d.spend_per_order)
    d["log_clv"] = np.log1p(d.total_clv)
    d["oily_texture_clash"] = ((d.skin_type == "oily") & (d.keyword_texture_clash == 1)).astype(int)
    d["match_band"] = pd.cut(d.formula_match_score, [0, .4, .6, .8, 1.0],
                             labels=["Low (<0.4)", "Medium (0.4-0.6)", "High (0.6-0.8)", "Very high (>0.8)"],
                             include_lowest=True)
    return d


def make_xy(d: pd.DataFrame):
    X = pd.get_dummies(d[NUM + CAT], columns=CAT, drop_first=True).astype(float)
    return X, d["is_churned"]
