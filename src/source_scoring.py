"""Task 2 - weighted scoring of candidate public data sources (scores are analyst judgement, 1-5)."""
import json
import matplotlib.pyplot as plt
from .common import REP, save_fig, SAGE, SAND, flowchart

# (relevance, reliability, accessibility, update frequency)
SOURCES = {
    "Sephora Kaggle dataset": (5, 4, 5, 1), "Open Beauty Facts": (3, 3, 5, 4),
    "EU CosIng ingredients": (3, 5, 4, 2), "Amazon Reviews 2023 (Beauty)": (4, 3, 4, 1),
    "FDA CAERS adverse events": (3, 3, 4, 3), "Google Trends": (2, 3, 5, 5),
    "Reddit (skincare communities)": (3, 2, 3, 5), "BLS Consumer Expenditure": (2, 5, 5, 1)}
WEIGHTS = (.4, .25, .2, .15)


def main():
    sc = {k: round(sum(a * b for a, b in zip(v, WEIGHTS)), 2) for k, v in SOURCES.items()}
    (REP / "source_scores.json").write_text(json.dumps(sc, indent=1))
    s = sorted(sc.items(), key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.barh([a for a, _ in s], [b for _, b in s], color=[SAGE if b >= 3.5 else SAND for _, b in s])
    for i, (_, b) in enumerate(s): ax.text(b + .04, i, f"{b:.2f}", va="center", fontsize=9)
    ax.set_xlim(0, 5); ax.set_title("Weighted source suitability score (max 5.0)")
    save_fig("t2_scores")
    flowchart([("Define needs", "variables from H1/H2"), ("Discover", "catalogues, APIs"), ("Assess", "licence, quality, fit"),
               ("Acquire", "download / API"), ("Profile", "schema, nulls, drift"), ("Integrate", "join on product_id"),
               ("Store", "versioned raw zone"), ("Hand-off", "to cleaning")], "t2_flow", "Data sourcing workflow")
    print(sc)


if __name__ == "__main__":
    main()
