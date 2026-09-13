import pandas as pd
from pathlib import Path

from imblearn.over_sampling import SMOTE


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"


# ============================================================
# Load Stage 2 training data
# ============================================================

X_train = pd.read_csv(
    DATA_DIR / "ann_X_train.csv"
)

y_train = pd.read_csv(
    DATA_DIR / "ann_y_train.csv"
)["target"]


# ============================================================
# Display original distribution
# ============================================================

print("=" * 60)
print("STAGE 2 SMOTE BALANCING")
print("=" * 60)

print("\nOriginal training shape:")
print(X_train.shape)

print("\nOriginal class distribution:")
print(y_train.value_counts().sort_index())


# ============================================================
# Apply SMOTE
# ============================================================

smote = SMOTE(
    random_state=42,
    k_neighbors=5
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)


# ============================================================
# Display balanced results
# ============================================================

print("\nAfter SMOTE:")
print("X_train_smote:", X_train_smote.shape)
print("y_train_smote:", y_train_smote.shape)

print("\nBalanced class distribution:")
print(
    pd.Series(y_train_smote)
    .value_counts()
    .sort_index()
)


# ============================================================
# Save balanced training data
# ============================================================

X_train_smote = pd.DataFrame(
    X_train_smote,
    columns=X_train.columns
)

y_train_smote = pd.Series(
    y_train_smote,
    name="target"
)

X_train_smote.to_csv(
    DATA_DIR / "ann_X_train_smote.csv",
    index=False
)

y_train_smote.to_csv(
    DATA_DIR / "ann_y_train_smote.csv",
    index=False
)


print("\nSMOTE training data saved successfully!")