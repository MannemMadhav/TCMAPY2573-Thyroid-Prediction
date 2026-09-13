import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split

from ann_thyroid_loader import load_ann_thyroid_data


# ============================================================
# Load dataset
# ============================================================

X, y = load_ann_thyroid_data()


# ============================================================
# Train / test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# Display results
# ============================================================

print("=" * 60)
print("STAGE 2 TRAIN / TEST SPLIT")
print("=" * 60)

print("\nOriginal dataset:")
print("X:", X.shape)
print("y:", y.shape)

print("\nTraining set:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting set:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\nTraining class distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting class distribution:")
print(y_test.value_counts().sort_index())


# ============================================================
# Save split datasets
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

X_train.to_csv(
    DATA_DIR / "ann_X_train.csv",
    index=False
)

X_test.to_csv(
    DATA_DIR / "ann_X_test.csv",
    index=False
)

y_train.to_csv(
    DATA_DIR / "ann_y_train.csv",
    index=False,
    header=["target"]
)

y_test.to_csv(
    DATA_DIR / "ann_y_test.csv",
    index=False,
    header=["target"]
)


print("\nStage 2 split saved successfully!")