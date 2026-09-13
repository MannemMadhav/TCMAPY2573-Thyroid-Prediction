import pandas as pd

from hierarchical_cascade import (
    HierarchicalThyroidCascade,
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


# ============================================================
# Reset indexes so X_test and y_test use the same positions
# ============================================================

X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)


# ============================================================
# Find the first normal test sample
# ============================================================

normal_positions = y_test[
    y_test == 1
].index

normal_position = int(
    normal_positions[0]
)


# ============================================================
# Select exactly one row
# ============================================================

sample = X_test.iloc[
    [normal_position]
]


# ============================================================
# Create cascade
# ============================================================

cascade = HierarchicalThyroidCascade(
    stage1_model,
    stage2_model
)


# ============================================================
# Run prediction
# ============================================================

result = cascade.predict(
    sample
)


# ============================================================
# Display result
# ============================================================

print("=" * 60)
print("HIERARCHICAL NORMAL-BRANCH TEST")
print("=" * 60)

print("\nTest sample position:")
print(normal_position)

print("\nTrue class:")
print("normal")

print("\nStage 1 prediction:")
print(
    result["stage1_prediction"]
)

print("\nStage 1 class:")
print(
    result["stage1_class"]
)

print("\nFinal class:")
print(
    result["final_class"]
)

print("\nStage 2 prediction:")
print(
    result["stage2_prediction"]
)

if result["stage2_prediction"] is None:
    print(
        "\nSUCCESS: Stage 2 was correctly skipped."
    )
else:
    print(
        "\nWARNING: Stage 2 was executed."
    )