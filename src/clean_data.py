"""Step 2 (Task 3) - Data cleaning and preprocessing pipeline.

Input : data/raw/raw.csv
Output: data/processed/clean.csv, reports/cleaning_log.json, figures t3_*.png
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from .common import RAW, PROC, update_json, save_fig, flowchart, CORAL, SAND, NAVY


def clean(raw: pd.DataFrame):
    log = {"raw_rows": len(raw)}
    miss = raw.drop(columns=["sentiment_polarity", "subjectivity"]).isna().mean() * 100
    log["missing_pct_before"] = miss[miss > 0].round(1).to_dict()

    # 1. de-duplicate on business key
    df = raw.drop_duplicates(subset="customer_id").copy()
    log["duplicates_removed"] = len(raw) - len(df)

    # 2. standardise categories
    log["skin_type_labels_before"] = int(raw.skin_type.nunique())
    df["skin_type"] = df.skin_type.str.strip().str.lower().replace({"combo": "combination"})
    log["skin_type_labels_after"] = int(df.skin_type.nunique())

    # 3. validate ranges -> impossible values become missing
    log["negative_recency_fixed"] = int((df.recency_days < 0).sum())
    df.loc[df.recency_days < 0, "recency_days"] = np.nan

    # 4. impute (group median + missing flags); categorical -> 'unknown'
    for c in ["sensitivity_score", "formula_match_score", "recency_days"]:
        df[c + "_was_missing"] = df[c].isna().astype(int)
        df[c] = df[c].fillna(df.groupby("skin_type")[c].transform("median"))
    df["skin_tone"] = df.skin_tone.fillna("unknown")
    df["age_bracket"] = df.age_bracket.fillna("unknown")
    df["sentiment_polarity"] = df.sentiment_polarity.fillna(0)   # no review -> no text signal
    df["subjectivity"] = df.subjectivity.fillna(0)

    # 5. outliers: IQR capping
    before = df.spend_per_order.copy()
    q1, q3 = df.spend_per_order.quantile([.25, .75])
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    n_out = int(((before < lo) | (before > hi)).sum())
    df["spend_per_order"] = before.clip(lo, hi)
    df["total_clv"] = (df.spend_per_order * df.frequency).round(2)
    log["spend_iqr_fences"] = [round(lo, 2), round(hi, 2)]
    log["spend_outliers_capped"] = n_out

    # 6. validation gates (fail loudly)
    assert df.customer_id.is_unique
    assert df.isna().sum().sum() == 0
    assert (df.recency_days >= 0).all()
    assert df.skin_type.isin(["dry", "oily", "combination", "normal", "sensitive"]).all()
    log["clean_rows"] = len(df)
    return df, log, before


def main():
    raw = pd.read_csv(RAW / "raw.csv")
    df, log, before = clean(raw)
    df.to_csv(PROC / "clean.csv", index=False)
    update_json("cleaning_log.json", log)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    mb = pd.Series(log["missing_pct_before"]).sort_values()
    ax[0].barh(mb.index, mb.values, color=CORAL); ax[0].set_title("Missing values before cleaning (%)")
    ax[1].boxplot([before, df.spend_per_order], label=["Before", "After IQR cap"], patch_artist=True,
                  boxprops=dict(facecolor=SAND), medianprops=dict(color=NAVY))
    ax[1].set_title("spend_per_order (USD)")
    save_fig("t3_quality")
    flowchart([("1 Ingest", "raw CSV / API"), ("2 Deduplicate", "customer_id key"), ("3 Standardise", "case, spacing, labels"),
               ("4 Validate", "range & logic checks"), ("5 Impute", "group median / mode"), ("6 Outliers", "IQR capping"),
               ("7 Text clean", "regex normalise"), ("8 Export", "clean.csv + log")],
              "t3_flow", "Data cleaning and preprocessing pipeline")
    print(log)


if __name__ == "__main__":
    main()
