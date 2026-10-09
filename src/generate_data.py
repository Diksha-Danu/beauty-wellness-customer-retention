"""Step 1 - Simulate a customer dataset that mirrors the Task 1 schema.

Produces
  data/raw/raw.csv            messy data (missing values, duplicates, outliers, inconsistent labels)
  data/processed/clean_truth.csv   the underlying clean data (for reference only)

Replace this step with real data (e.g. Sephora Kaggle + first-party CRM) when available.
"""
import numpy as np, pandas as pd
from .common import RAW, PROC

N = 6000


def generate(seed=42):
    rng = np.random.default_rng(seed)
    skin = rng.choice(["dry", "oily", "combination", "normal", "sensitive"], N, p=[.22, .24, .28, .12, .14])
    tone = rng.choice(["fair", "light", "medium", "tan", "deep"], N, p=[.15, .25, .28, .18, .14])
    age = rng.choice(["18-24", "25-34", "35-44", "45-54", "55+"], N, p=[.2, .33, .24, .14, .09])
    sens = np.clip(rng.normal(4.5, 2.2, N) + (skin == "sensitive") * 2.5, 0, 10).round(1)
    match = np.clip(rng.beta(4, 2.6, N), 0, 1)
    disc = np.clip(rng.beta(2, 3.5, N), 0, 1)
    freq = 1 + rng.poisson(1.1, N)
    recency = np.clip(rng.gamma(2.2, 18, N), 1, 120).round().astype(int)
    spend = np.clip(rng.lognormal(3.9, .35, N), 15, 400).round(2)
    clv = (spend * freq * rng.uniform(.9, 1.1, N)).round(2)
    has_review = rng.random(N) < 0.42
    p_adv = 1 / (1 + np.exp(-(-2.6 + 2.2 * (1 - match) + 0.18 * (sens - 4.5))))
    adv = (rng.random(N) < p_adv).astype(int)
    sent = np.clip(0.45 - 0.9 * adv - 0.5 * (1 - match) + rng.normal(0, .35, N), -1, 1)
    clash = (rng.random(N) < (0.08 + 0.35 * (1 - match) * (skin == "oily") + 0.2 * adv)).astype(int)
    subj = np.clip(rng.beta(3, 2.5, N), 0, 1)
    z = (0.75 - 3.0 * (match - 0.6) + 1.1 * adv - 1.3 * np.where(has_review, sent, 0.15)
         + 0.7 * clash + 1.6 * (disc - 0.3) - 0.45 * np.log(freq) + 0.012 * (recency - 40)
         + 0.12 * (sens - 4.5) * (1 - match) + rng.normal(0, .35, N))
    churn = (rng.random(N) < 1 / (1 + np.exp(-z))).astype(int)
    df = pd.DataFrame(dict(
        customer_id=np.arange(1, N + 1), skin_type=skin, skin_tone=tone, age_bracket=age,
        sensitivity_score=sens, recency_days=recency, frequency=freq, spend_per_order=spend,
        total_clv=clv, discount_reliance=disc.round(3), formula_match_score=match.round(3),
        has_review=has_review.astype(int), adverse_reaction_flag=np.where(has_review, adv, 0),
        sentiment_polarity=np.where(has_review, sent.round(3), np.nan),
        keyword_texture_clash=np.where(has_review, clash, 0),
        subjectivity=np.where(has_review, subj.round(3), np.nan), is_churned=churn))

    # ---- corrupt a copy to mimic real-world raw data ----
    raw = df.copy()
    for c, p in [("skin_tone", .08), ("age_bracket", .06), ("sensitivity_score", .10), ("formula_match_score", .03)]:
        raw.loc[rng.random(N) < p, c] = np.nan
    idx = rng.random(N) < .015
    raw.loc[idx, "spend_per_order"] *= rng.uniform(8, 15, idx.sum())
    idx = rng.random(N) < .008
    raw.loc[idx, "recency_days"] = -rng.integers(1, 30, idx.sum())

    def mess(s):
        r = rng.random()
        if r < .06: return s.upper()
        if r < .11: return " " + s.capitalize() + " "
        if s == "combination" and r < .2: return "combo"
        return s
    raw["skin_type"] = raw["skin_type"].map(mess)
    dups = raw.sample(int(.02 * N), random_state=1)
    raw = pd.concat([raw, dups]).sample(frac=1, random_state=3).reset_index(drop=True)
    return df, raw


def main():
    df, raw = generate()
    df.to_csv(PROC / "clean_truth.csv", index=False)
    raw.to_csv(RAW / "raw.csv", index=False)
    print(f"raw.csv: {raw.shape}, churn rate {df.is_churned.mean():.3f}")


if __name__ == "__main__":
    main()
