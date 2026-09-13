import pandas as pd
import joblib
from pathlib import Path

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

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
# Load ORIGINAL Stage 2 training data
# ============================================================

X_train = pd.read_csv(
    DATA_DIR / "ann_X_train.csv"
)

y_train = pd.read_csv(
    DATA_DIR / "ann_y_train.csv"
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
# Base Model 1 - Random Forest
#
# SMOTE is inside the pipeline.
# ============================================================

rf_pipeline = ImbPipeline(
    steps=[
        (
            "smote",
            SMOTE(
                random_state=42,
                k_neighbors=5
            )
        ),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

calibrated_rf = CalibratedClassifierCV(
    estimator=rf_pipeline,
    method="sigmoid",
    cv=3
)


# ============================================================
# Base Model 2 - XGBoost
#
# SMOTE is inside the pipeline.
# ============================================================

xgb_pipeline = ImbPipeline(
    steps=[
        (
            "smote",
            SMOTE(
                random_state=42,
                k_neighbors=5
            )
        ),
        (
            "xgb",
            XGBClassifier(
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
        )
    ]
)

calibrated_xgb = CalibratedClassifierCV(
    estimator=xgb_pipeline,
    method="sigmoid",
    cv=3
)


# ============================================================
# Base Model 3 - SVM
#
# SMOTE + scaling are inside the pipeline.
# ============================================================

svm_pipeline = ImbPipeline(
    steps=[
        (
            "smote",
            SMOTE(
                random_state=42,
                k_neighbors=5
            )
        ),
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

calibrated_svm = CalibratedClassifierCV(
    estimator=svm_pipeline,
    method="sigmoid",
    cv=3
)


# ============================================================
# Calibrated Stacking Ensemble
#
# Logistic Regression is the meta-learner.
# ============================================================

stacking_model = StackingClassifier(
    estimators=[
        ("rf", calibrated_rf),
        ("xgb", calibrated_xgb),
        ("svm", calibrated_svm)
    ],
    final_estimator=LogisticRegression(
        max_iter=2000,
        random_state=42
    ),
    stack_method="predict_proba",
    cv=3,
    n_jobs=-1
)


# ============================================================
# Train
# ============================================================

print("=" * 60)
print("STAGE 2 - CALIBRATED STACKING ENSEMBLE")
print("=" * 60)

print("\nOriginal training data:")
print(X_train.shape)

print("\nUntouched test data:")
print(X_test.shape)

print("\nTraining calibrated stacking ensemble...")
print("This may take several minutes.")

stacking_model.fit(
    X_train,
    y_train
)
# ============================================================
# Save trained stacking model
# ============================================================

MODEL_DIR = BASE_DIR / "model"

MODEL_DIR.mkdir(
    exist_ok=True
)

joblib.dump(
    stacking_model,
    MODEL_DIR / "final_calibrated_stacking_model.pkl"
)

print(
    "\nCalibrated stacking model saved successfully!"
)

print(
    "Saved to:",
    MODEL_DIR / "final_calibrated_stacking_model.pkl"
)

print("\nCalibrated stacking ensemble trained successfully!")


# ============================================================
# Prediction
# ============================================================

y_pred = stacking_model.predict(
    X_test
)


# ============================================================
# Probability output
# ============================================================

probabilities = stacking_model.predict_proba(
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

print("\nCalibrated Stacking Evaluation:")

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