"""Run the full pipeline:  python run_all.py"""
from src import generate_data, source_scoring, clean_data, eda, train_models, evaluate

for name, step in [("1 Generate data", generate_data), ("2 Source scoring (Task 2)", source_scoring), ("3 Clean (Task 3)", clean_data),
                   ("4 EDA + features (Task 4)", eda), ("5 Train models (Task 5)", train_models), ("6 Evaluate (Task 6)", evaluate)]:
    print(f"\n=== {name} ===")
    step.main()
print("\nDone. See reports/ and reports/figures/")
