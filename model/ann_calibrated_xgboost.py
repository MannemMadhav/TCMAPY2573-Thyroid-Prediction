import pandas as pd
from pathlib import Path

from xgboost import XGBClassifier

from sklearn.calibration import CalibratedClassifierCV
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
# Convert labels for XGBoost
#
# Original:
# 1 = normal
# 2 = hyperfunction
# 3 = subnormal
#
# XGBoost:
# 0, 1, 2
# ============================================================

y_train_xgb = y_train - 1
y_test_xgb = y_test - 1


# ============================================================
# Base XGBoost
# ============================================================

base_xgb = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)


# ============================================================
# Calibrated XGBoost
# ============================================================

calibrated_xgb = CalibratedClassifierCV(
    estimator=base_xgb,
    method="sigmoid",
    cv=3
)


# ============================================================
# Train
# ============================================================

print("=" * 60)
print("STAGE 2 - CALIBRATED XGBOOST")
print("=" * 60)

print("\nTraining data shape:")
print(X_train.shape)

print("\nTest data shape:")
print(X_test.shape)

print("\nTraining calibrated XGBoost...")

calibrated_xgb.fit(
    X_train,
    y_train_xgb
)

print("Calibrated XGBoost trained successfully!")


# ============================================================
# Prediction
# ============================================================

y_pred_xgb = calibrated_xgb.predict(
    X_test
)

# Convert back to original labels
y_pred = y_pred_xgb + 1


# ============================================================
# Probabilities
# ============================================================

probabilities = calibrated_xgb.predict_proba(
    X_test
)

print("\nProbability output shape:")
print(probabilities.shape)


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

print("\nCalibrated XGBoost Evaluation:")

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