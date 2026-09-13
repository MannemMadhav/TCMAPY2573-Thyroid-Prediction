import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"


# ============================================================
# Load SMOTE training data
# ============================================================

X_train = pd.read_csv(
    DATA_DIR / "ann_X_train_smote.csv"
)

y_train = pd.read_csv(
    DATA_DIR / "ann_y_train_smote.csv"
)["target"]


# ============================================================
# Load untouched test data
# ============================================================

X_test = pd.read_csv(
    DATA_DIR / "ann_X_test.csv"
)

y_test = pd.read_csv(
    DATA_DIR / "ann_y_test.csv"
)["target"]


# ============================================================
# Create Random Forest
# ============================================================

rf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# Train
# ============================================================

print("=" * 60)
print("STAGE 2 - RANDOM FOREST TRAINING")
print("=" * 60)

print("\nTraining data shape:")
print(X_train.shape)

print("\nTest data shape:")
print(X_test.shape)

print("\nTraining Random Forest...")

rf_model.fit(
    X_train,
    y_train
)

print("Random Forest trained successfully!")


# ============================================================
# Prediction
# ============================================================

y_pred = rf_model.predict(
    X_test
)


# ============================================================
# Evaluation
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)

print("\nRandom Forest Evaluation:")

print(
    f"Accuracy: {accuracy:.4f}"
)

print(
    f"Macro F1-Score: {macro_f1:.4f}"
)


# ============================================================
# Confusion Matrix
# ============================================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# Classification Report
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "normal",
            "hyperfunction",
            "subnormal"
        ]
    )
)