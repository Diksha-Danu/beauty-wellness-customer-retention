# Predictive Customer Retention in Premium Beauty & Skincare

End-to-end data-science project: predict which first-time D2C beauty customers will **not repurchase within 90 days** (churn), explain why, and turn the scores into retention actions.

Built for the Beauty & Wellness Analytics internship (Weeks 1-6).

> **Data note:** the repo ships with a **simulated 6,000-customer dataset** that mirrors the project schema, so every step runs offline. All reported numbers come from this simulation. To use real data, replace `src/generate_data.py` with a loader that writes `data/raw/raw.csv` using the same columns (see *Data dictionary*).

## Business problem
75% of new beauty buyers never place a second order. Three causes drive this: **formula/biometric mismatch**, **silent attrition**, and **blanket discounts** that erode margin.

| Hypothesis | Test |
|---|---|
| **H1** skin-formula compatibility raises retention | retention gap between high and low `formula_match_score` (target >= 25% relative, p < 0.05) |
| **H2** negative review sentiment predicts churn early | negative, significant `sentiment_polarity` coefficient in logistic regression |

## Pipeline (maps to the six internship tasks)

| Task | Module | Output |
|---|---|---|
| 1. Problem definition | (documentation) | Strategic planning document |
| 2. Data sourcing | `src/source_scoring.py` | source scores, sourcing workflow chart |
| 3. Cleaning | `src/clean_data.py` | `data/processed/clean.csv`, `reports/cleaning_log.json` |
| 4. EDA + features | `src/features.py`, `src/eda.py` | hypothesis tests, EDA figures, `features.csv` |
| 5. Modeling | `src/train_models.py` | model comparison, tuned Random Forest, K-means segments |
| 6. Evaluation | `src/evaluate.py` | test metrics, ROC/PR, calibration, lift, business simulation |

## Quick start
```bash
git clone https://github.com/<your-username>/beauty-churn-retention.git
cd beauty-churn-retention
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_all.py            # runs all steps (about 1-2 minutes)
python -m pytest -q tests    # optional tests
```
Run a single step: `python -m src.clean_data`

Score customers with the saved model:
```bash
python -m src.predict data/processed/features.csv reports/scored_customers.csv
```

## Results (simulated data, hold-out test set of 1,200)
| Metric | Value |
|---|---|
| ROC-AUC | 0.714 (95% CI 0.683-0.739) |
| PR-AUC | 0.807 (baseline 0.625) |
| Precision / Recall @ 0.61 | 74.8% / 66.8% |
| Top risk decile churn vs bottom | 93.3% vs 35.8% |
| H1 retention (high vs low match) | 46.0% vs 27.1% (+69.9% relative, p < 0.001) |
| H2 sentiment odds ratio | 0.17 per unit (p < 0.001) |
| Illustrative campaign net benefit | Targeted +$1,204 vs blanket -$3,825 |

Logistic Regression (0.727), Random Forest (0.714) and Gradient Boosting (0.712) perform similarly; Random Forest is the primary model and Logistic Regression the explainable benchmark. Cost, margin and save-rate figures in `src/evaluate.py` are **placeholders**.

## Project structure
```
beauty-churn-retention/
├── run_all.py
├── requirements.txt
├── src/
│   ├── common.py          paths, plot style, helpers
│   ├── generate_data.py   simulated raw data
│   ├── source_scoring.py  Task 2 source scoring
│   ├── clean_data.py      Task 3 pipeline
│   ├── features.py        Task 4 feature engineering
│   ├── eda.py             Task 4 EDA + H1/H2 tests
│   ├── train_models.py    Task 5 model comparison
│   ├── evaluate.py        Task 6 evaluation
│   └── predict.py         batch scoring
├── tests/test_pipeline.py
├── data/{raw,processed}/  (generated, git-ignored)
├── models/                (generated, git-ignored)
└── reports/               metrics JSON + figures/
```

## Data dictionary
| Column | Description |
|---|---|
| `customer_id` | unique key |
| `skin_type`, `skin_tone`, `age_bracket` | profile categories |
| `sensitivity_score` | 0-10 |
| `recency_days`, `frequency`, `spend_per_order`, `total_clv` | RFM behaviour |
| `discount_reliance` | 0-1 share of orders using discounts |
| `formula_match_score` | 0-1 compatibility of purchased formula with skin profile |
| `has_review`, `adverse_reaction_flag`, `sentiment_polarity`, `keyword_texture_clash`, `subjectivity` | review NLP features |
| `is_churned` | **target**: 1 = no repurchase within 90 days |

## Using real data
1. Public sources: Sephora Products and Skincare Reviews (Kaggle), Open Beauty Facts, EU CosIng, FDA CAERS. First-party order logs are needed for the true churn label.
2. Build the same columns (sentiment via VADER/TextBlob or a transformer; match score from skin profile vs ingredient contraindications).
3. Save as `data/raw/raw.csv` and run steps 3-6.

## Limitations
Associations are observational, not causal: validate with an A/B test. Use skin attributes only to improve product fit, never to deny service or set prices. Follow GDPR / India DPDP Act for personal data.

## License
MIT
