import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

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
# Load ANN multiclass dataset
# ============================================================

from ann_thyroid_loader import load_ann_thyroid_data

X, y = load_ann_thyroid_data()


# ============================================================
# Create Stage 1 binary target
#
# 0 = normal
# 1 = thyroid dysfunction
#
# Class 1 = normal
# Classes 2 and 3 = dysfunction
# ============================================================

y_stage1 = y.map(
    {
        1: 0,
        2: 1,
        3: 1
    }
)


# ============================================================
# Display distribution
# ============================================================

print("=" * 60)
print("STAGE 1 - THYROID DYSFUNCTION SCREENING")
print("=" * 60)

print("\nOriginal multiclass distribution:")
print(y.value_counts().sort_index())

print("\nStage 1 binary distribution:")
print(y_stage1.value_counts().sort_index())


# ============================================================
# Train/test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_stage1,
    test_size=0.20,
    random_state=42,
    stratify=y_stage1
)


print("\nTraining set:")
print(X_train.shape)

print("\nTesting set:")
print(X_test.shape)


# ============================================================
# SMOTE ONLY on training data
# ============================================================

smote = SMOTE(
    random_state=42,
    k_neighbors=5
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)


print("\nAfter SMOTE:")

print(
    pd.Series(y_train_smote)
    .value_counts()
    .sort_index()
)


# ============================================================
# Stage 1 XGBoost
# ============================================================

stage1_model = XGBClassifier(
    objective="binary:logistic",
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)


# ============================================================
# Train
# ============================================================

print("\nTraining Stage 1 XGBoost...")

stage1_model.fit(
    X_train_smote,
    y_train_smote
)

print(
    "Stage 1 XGBoost trained successfully!"
)


# ============================================================
# Save hierarchical Stage 1 model
# ============================================================

MODEL_PATH = (
    BASE_DIR
    / "model"
    / "hierarchical_stage1_xgboost.pkl"
)

joblib.dump(
    stage1_model,
    MODEL_PATH
)

print(
    "\nHierarchical Stage 1 model saved successfully!"
)

print(
    "Saved to:",
    MODEL_PATH
)

MODEL_PATH = (
    BASE_DIR
    / "model"
    / "hierarchical_stage1_xgboost.pkl"
)

joblib.dump(
    stage1_model,
    MODEL_PATH
)

print(
    "\nHierarchical Stage 1 model saved successfully!"
)

print(
    "Saved to:",
    MODEL_PATH
)

print(
    "Stage 1 XGBoost trained successfully!"
)


# ============================================================
# Prediction
# ============================================================

y_pred = stage1_model.predict(
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

print("\nStage 1 Evaluation:")

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
            "thyroid_dysfunction"
        ]
    )
)