import pandas as pd
from src.generate_data import generate
from src.clean_data import clean
from src.features import add_features, make_xy


def test_cleaning_removes_problems():
    _, raw = generate()
    df, log, _ = clean(raw)
    assert df.customer_id.is_unique and df.isna().sum().sum() == 0
    assert (df.recency_days >= 0).all()
    assert df.skin_type.nunique() == 5
    assert log["duplicates_removed"] > 0


def test_features_shape():
    _, raw = generate()
    d = add_features(clean(raw)[0])
    X, y = make_xy(d)
    assert len(X) == len(y) and "mismatch_x_sensitivity" in X.columns
    assert X.isna().sum().sum() == 0
