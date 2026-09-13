import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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
# Base SVM
# ============================================================

base_svm = Pipeline(
    [
        (
            "scaler",
            StandardScaler()
        ),
        (
            "svm",
            SVC(
                kernel="rbf",
                C=1.0,
                random_state=42
            )
        )
    ]
)


# ============================================================
# Calibrated SVM
# ============================================================

calibrated_svm = CalibratedClassifierCV(
    estimator=base_svm,
    method="sigmoid",
    cv=3
)


# ============================================================
# Train
# ============================================================

print("=" * 60)
print("STAGE 2 - CALIBRATED SVM")
print("=" * 60)

print("\nTraining data shape:")
print(X_train.shape)

print("\nTest data shape:")
print(X_test.shape)

print("\nTraining calibrated SVM...")

calibrated_svm.fit(
    X_train,
    y_train
)

print("Calibrated SVM trained successfully!")


# ============================================================
# Prediction
# ============================================================

y_pred = calibrated_svm.predict(
    X_test
)


# ============================================================
# Probabilities
# ============================================================

probabilities = calibrated_svm.predict_proba(
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

print("\nCalibrated SVM Evaluation:")

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