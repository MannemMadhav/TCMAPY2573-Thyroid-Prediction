import pandas as pd

# ============================================================
# TCMAPY2573 - Final Model Comparison
# ============================================================

results = {
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
        "Tuned XGBoost"
    ],
    "Accuracy": [
        0.9621,
        0.9937,
        0.9953,
        0.9921
    ],
    "Precision": [
        0.5600,
        0.9333,
        0.9655,
        0.9032
    ],
    "Recall": [
        0.9333,
        0.9333,
        0.9333,
        0.9333
    ],
    "F1-Score": [
        0.7000,
        0.9333,
        0.9492,
        0.9180
    ],
    "ROC-AUC": [
        0.9669,
        0.9783,
        0.9800,
        0.9697
    ]
}

comparison = pd.DataFrame(results)

print("=" * 90)
print("FINAL MODEL COMPARISON")
print("=" * 90)

print(comparison.to_string(index=False))

# Save final comparison
comparison.to_csv(
    r".\data\model_comparison.csv",
    index=False
)

print("\nFinal model comparison saved successfully!")
print("Saved to: data/model_comparison.csv")