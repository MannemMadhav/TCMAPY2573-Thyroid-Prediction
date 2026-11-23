from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier


# ---------------------------------------------------------
# TCMAPY2573
# Enhanced Stage 2 Calibrated Stacking Ensemble
# Step 355
# ---------------------------------------------------------

X_TRAIN_FILE = Path(
    "data/enhanced_stage2_smote_X_train_processed.csv"
)

Y_TRAIN_FILE = Path(
    "data/enhanced_stage2_smote_y_train.csv"
)

MODEL_OUTPUT = Path(
    "model/enhanced_stage2_calibrated_stacking.pkl"
)

ARCHITECTURE_OUTPUT = Path(
    "data/enhanced_stage2_model_architecture.json"
)

FEATURE_OUTPUT = Path(
    "data/enhanced_stage2_model_features.csv"
)


RANDOM_STATE = 42


# ---------------------------------------------------------
# Class definitions
# ---------------------------------------------------------

CLASS_NAMES = {
    0: "Normal",
    1: "Hyperthyroid",
    2: "Hypothyroid",
    3: "Other / Related Diagnosis",
}


def main():

    print("=" * 80)
    print(
        "TCMAPY2573 - ENHANCED STAGE 2 CALIBRATED STACKING"
    )
    print("=" * 80)

    # -----------------------------------------------------
    # File checks
    # -----------------------------------------------------

    if not X_TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training feature file not found: {X_TRAIN_FILE}"
        )

    if not Y_TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training target file not found: {Y_TRAIN_FILE}"
        )

    # -----------------------------------------------------
    # Load balanced training data
    # -----------------------------------------------------

    X_train = pd.read_csv(
        X_TRAIN_FILE
    )

    y_train = pd.read_csv(
        Y_TRAIN_FILE
    )["stage2_target"].astype(int)

    print()
    print("TRAINING DATA")
    print("-" * 80)

    print(
        "X shape:",
        X_train.shape
    )

    print(
        "y shape:",
        y_train.shape
    )

    if X_train.shape[0] != 8610:
        raise ValueError(
            "Expected 8610 balanced training samples."
        )

    if X_train.shape[1] != 56:
        raise ValueError(
            "Expected 56 model features."
        )

    if len(y_train) != 8610:
        raise ValueError(
            "Expected 8610 target values."
        )

    # -----------------------------------------------------
    # Verify no missing / infinite values
    # -----------------------------------------------------

    print()
    print("INPUT VALIDATION")
    print("-" * 80)

    missing = int(
        X_train.isna().sum().sum()
    )

    infinite = int(
        X_train.to_numpy().astype(float)
        .__class__
        and 0
    )

    # More reliable explicit infinity check
    import numpy as np

    infinite = int(
        np.isinf(
            X_train.to_numpy(
                dtype=float
            )
        ).sum()
    )

    print(
        "Missing values:",
        missing
    )

    print(
        "Infinite values:",
        infinite
    )

    if missing != 0:
        raise ValueError(
            "Missing values detected."
        )

    if infinite != 0:
        raise ValueError(
            "Infinite values detected."
        )

    # -----------------------------------------------------
    # Class distribution
    # -----------------------------------------------------

    print()
    print("TRAINING CLASS DISTRIBUTION")
    print("-" * 80)

    class_counts = (
        y_train
        .value_counts()
        .sort_index()
    )

    for class_id, name in CLASS_NAMES.items():

        print(
            f"{name:30} "
            f"{int(class_counts.get(class_id, 0))}"
        )

    # -----------------------------------------------------
    # Base learner 1
    # Random Forest
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("CREATING CALIBRATED RANDOM FOREST")
    print("=" * 80)

    random_forest = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    calibrated_rf = CalibratedClassifierCV(
        estimator=random_forest,
        method="sigmoid",
        cv=3,
    )

    # -----------------------------------------------------
    # Base learner 2
    # XGBoost
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("CREATING CALIBRATED XGBOOST")
    print("=" * 80)

    xgboost = XGBClassifier(
        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss",
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        min_child_weight=2,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=3.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    calibrated_xgb = CalibratedClassifierCV(
        estimator=xgboost,
        method="sigmoid",
        cv=3,
    )

    # -----------------------------------------------------
    # Base learner 3
    # SVM
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("CREATING CALIBRATED SVM")
    print("=" * 80)

    svm = SVC(
        kernel="rbf",
        C=2.0,
        gamma="scale",
        class_weight=None,
        random_state=RANDOM_STATE,
    )

    calibrated_svm = CalibratedClassifierCV(
        estimator=svm,
        method="sigmoid",
        cv=3,
    )

    # -----------------------------------------------------
    # Meta learner
    # Logistic Regression
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("CREATING LOGISTIC REGRESSION META-LEARNER")
    print("=" * 80)

    meta_learner = LogisticRegression(
    max_iter=2000,
    solver="lbfgs",
    random_state=RANDOM_STATE,
)

    # -----------------------------------------------------
    # Stacking ensemble
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("CREATING CALIBRATED STACKING ENSEMBLE")
    print("=" * 80)

    stacking_model = StackingClassifier(
        estimators=[
            (
                "calibrated_rf",
                calibrated_rf,
            ),
            (
                "calibrated_xgb",
                calibrated_xgb,
            ),
            (
                "calibrated_svm",
                calibrated_svm,
            ),
        ],
        final_estimator=meta_learner,
        stack_method="predict_proba",
        cv=3,
        n_jobs=-1,
        passthrough=False,
    )

    print(
        "Base learners:"
    )

    print(
        "1. Calibrated Random Forest"
    )

    print(
        "2. Calibrated XGBoost"
    )

    print(
        "3. Calibrated SVM"
    )

    print()
    print(
        "Meta-learner:"
    )

    print(
        "Logistic Regression"
    )

    print()
    print(
        "Stack method:",
        "predict_proba"
    )

    print(
        "Stacking CV folds:",
        3
    )

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("TRAINING ENHANCED STAGE 2 ENSEMBLE")
    print("=" * 80)

    print(
        "Training may take some time because"
        " calibration and stacking use cross-validation."
    )

    stacking_model.fit(
        X_train,
        y_train,
    )

    print()
    print(
        "Stage 2 ensemble training completed."
    )

    # -----------------------------------------------------
    # Verify fitted architecture
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("MODEL ARCHITECTURE VERIFICATION")
    print("=" * 80)

    print(
        "Model type:",
        type(stacking_model).__name__,
    )

    print(
        "Number of base learners:",
        len(stacking_model.estimators),
    )

    print(
        "Stack method:",
        stacking_model.stack_method,
    )

    print(
        "CV folds:",
        stacking_model.cv,
    )

    print(
        "Meta-learner:",
        type(
            stacking_model.final_estimator
        ).__name__,
    )

    for name, estimator in zip(
        [
            "Random Forest",
            "XGBoost",
            "SVM",
        ],
        stacking_model.estimators_,
    ):

        print(
            f"{name} wrapper:",
            type(estimator).__name__,
        )

    # -----------------------------------------------------
    # Verify classes
    # -----------------------------------------------------

    print()
    print("MODEL CLASSES")
    print("-" * 80)

    print(
        stacking_model.classes_
    )

    expected_classes = [
        0,
        1,
        2,
        3,
    ]

    if list(
        stacking_model.classes_
    ) != expected_classes:

        raise ValueError(
            "Expected four Stage 2 classes."
        )

    # -----------------------------------------------------
    # Save model
    # -----------------------------------------------------

    MODEL_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        stacking_model,
        MODEL_OUTPUT,
    )

    # -----------------------------------------------------
    # Save feature names
    # -----------------------------------------------------

    pd.DataFrame(
        {
            "feature_index": range(
                len(X_train.columns)
            ),
            "feature_name": X_train.columns,
        }
    ).to_csv(
        FEATURE_OUTPUT,
        index=False,
    )

    # -----------------------------------------------------
    # Save architecture metadata
    # -----------------------------------------------------

    architecture = {
        "model_type": type(
            stacking_model
        ).__name__,
        "base_learners": {
            "random_forest": {
                "wrapper": "CalibratedClassifierCV",
                "calibration": "sigmoid",
                "cv": 3,
            },
            "xgboost": {
                "wrapper": "CalibratedClassifierCV",
                "calibration": "sigmoid",
                "cv": 3,
            },
            "svm": {
                "wrapper": "CalibratedClassifierCV",
                "calibration": "sigmoid",
                "cv": 3,
            },
        },
        "meta_learner": {
            "type": "LogisticRegression",
            "max_iter": 2000,
        },
        "stack_method": "predict_proba",
        "stacking_cv": 3,
        "number_of_features": 56,
        "number_of_classes": 4,
        "classes": CLASS_NAMES,
        "training_rows": len(X_train),
    }

    with open(
        ARCHITECTURE_OUTPUT,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            architecture,
            file,
            indent=4,
        )

    # -----------------------------------------------------
    # Final
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 355 COMPLETED")
    print("=" * 80)

    print(
        "Enhanced Stage 2 model:",
        MODEL_OUTPUT,
    )

    print(
        "Architecture metadata:",
        ARCHITECTURE_OUTPUT,
    )

    print(
        "Feature list:",
        FEATURE_OUTPUT,
    )

    print()
    print("VERIFIED ARCHITECTURE:")
    print(
        "Calibrated RF + Calibrated XGBoost + Calibrated SVM"
    )

    print(
        "            ↓"
    )

    print(
        "Logistic Regression Meta-Learner"
    )

    print(
        "            ↓"
    )

    print(
        "Normal / Hyperthyroid / Hypothyroid / Other"
    )

    print()
    print("IMPORTANT:")
    print("- Only the balanced TRAINING data was used.")
    print("- The 1835-record test set was NOT used.")
    print("- Logistic Regression is the explicit meta-learner.")
    print("- Base learners provide calibrated probabilities.")
    print("- No Flask code was modified.")
    print("- No final test performance claim is made yet.")


if __name__ == "__main__":
    main()