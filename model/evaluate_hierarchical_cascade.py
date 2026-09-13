import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from hierarchical_cascade import (
    stage1_model,
    stage2_model
)


# ============================================================
# Load test data
# ============================================================

X_test = pd.read_csv(
    r".\data\ann_X_test.csv"
)

y_test = pd.read_csv(
    r".\data\ann_y_test.csv"
)["target"]


# Reset indexes for safe positional selection
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)


# ============================================================
# Stage 1 prediction for ALL samples
#
# 0 = normal
# 1 = thyroid dysfunction
# ============================================================

stage1_predictions = stage1_model.predict(
    X_test
).astype(int)


# ============================================================
# Start final predictions
#
# Stage 1 normal -> final class = 1
# ============================================================

final_predictions = pd.Series(
    1,
    index=X_test.index,
    dtype=int
)


# ============================================================
# Find samples requiring Stage 2
# ============================================================

dysfunction_mask = (
    stage1_predictions == 1
)

dysfunction_count = int(
    dysfunction_mask.sum()
)

normal_count = int(
    (~dysfunction_mask).sum()
)


# ============================================================
# Run Stage 2 ONLY for dysfunction samples
#
# Stage 2:
# 1 = normal
# 2 = hyperfunction
# 3 = subnormal
# ============================================================

if dysfunction_count > 0:

    X_stage2 = X_test.loc[
        dysfunction_mask
    ]

    stage2_predictions = (
        stage2_model.predict(X_stage2)
        .astype(int)
    )

    final_predictions.loc[
        dysfunction_mask
    ] = stage2_predictions


# ============================================================
# Evaluation
# ============================================================

accuracy = accuracy_score(
    y_test,
    final_predictions
)

macro_f1 = f1_score(
    y_test,
    final_predictions,
    average="macro"
)


# ============================================================
# Results
# ============================================================

print("=" * 60)
print("FULL HIERARCHICAL CASCADE EVALUATION")
print("=" * 60)

print("\nTotal test samples:")
print(len(X_test))

print("\nStage 2 used:")
print(dysfunction_count)

print("\nStage 2 skipped:")
print(normal_count)

print("\nHierarchical Accuracy:")
print(
    f"{accuracy:.4f}"
)

print("\nHierarchical Macro F1:")
print(
    f"{macro_f1:.4f}"
)


# ============================================================
# Confusion Matrix
# ============================================================

print("\nHierarchical Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        final_predictions
    )
)


# ============================================================
# Classification Report
# ============================================================

print("\nHierarchical Classification Report:")

print(
    classification_report(
        y_test,
        final_predictions,
        target_names=[
            "normal",
            "hyperfunction",
            "subnormal"
        ]
    )
)


print(
    "\nFull hierarchical cascade evaluation completed!"
)