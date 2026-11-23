from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


# ---------------------------------------------------------
# TCMAPY2573
# Enhanced Stage 1 XGBoost
# Step 347
# ---------------------------------------------------------

X_TRAIN_FILE = Path(
    "data/enhanced_X_train_processed.csv"
)

X_TEST_FILE = Path(
    "data/enhanced_X_test_processed.csv"
)

TRAIN_RAW_FILE = Path(
    "data/enhanced_train_with_targets.csv"
)

TEST_RAW_FILE = Path(
    "data/enhanced_test_with_targets.csv"
)

MODEL_OUTPUT = Path(
    "model/enhanced_stage1_xgboost.pkl"
)

VALIDATION_RESULTS_OUTPUT = Path(
    "data/enhanced_stage1_validation_results.csv"
)

TEST_METRICS_OUTPUT = Path(
    "data/enhanced_stage1_test_metrics.csv"
)

TEST_PREDICTIONS_OUTPUT = Path(
    "data/enhanced_stage1_test_predictions.csv"
)

TEST_CONFUSION_OUTPUT = Path(
    "data/enhanced_stage1_test_confusion_matrix.csv"
)


RANDOM_STATE = 42


# ---------------------------------------------------------
# Candidate XGBoost configurations
# ---------------------------------------------------------

CANDIDATES = [
    {
        "name": "XGB_A",
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.05,
        "min_child_weight": 2,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "reg_lambda": 2.0,
    },
    {
        "name": "XGB_B",
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_child_weight": 2,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "reg_lambda": 2.0,
    },
    {
        "name": "XGB_C",
        "n_estimators": 500,
        "max_depth": 4,
        "learning_rate": 0.03,
        "min_child_weight": 2,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "reg_lambda": 3.0,
    },
    {
        "name": "XGB_D",
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.03,
        "min_child_weight": 2,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "reg_lambda": 3.0,
    },
]


def create_model(config):
    """Create an XGBoost binary classifier."""

    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        min_child_weight=config["min_child_weight"],
        subsample=config["subsample"],
        colsample_bytree=config["colsample_bytree"],
        reg_lambda=config["reg_lambda"],
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def calculate_metrics(y_true, y_pred, y_probability):

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "precision_macro": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            y_probability,
        ),
    }


def main():

    print("=" * 80)
    print("TCMAPY2573 - ENHANCED STAGE 1 XGBOOST")
    print("=" * 80)

    # -----------------------------------------------------
    # File checks
    # -----------------------------------------------------

    required_files = [
        X_TRAIN_FILE,
        X_TEST_FILE,
        TRAIN_RAW_FILE,
        TEST_RAW_FILE,
    ]

    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    # -----------------------------------------------------
    # Load processed features
    # -----------------------------------------------------

    X_train = pd.read_csv(
        X_TRAIN_FILE
    )

    X_test = pd.read_csv(
        X_TEST_FILE
    )

    train_raw = pd.read_csv(
        TRAIN_RAW_FILE
    )

    test_raw = pd.read_csv(
        TEST_RAW_FILE
    )

    # -----------------------------------------------------
    # Basic verification
    # -----------------------------------------------------

    print()
    print("DATASET")
    print("-" * 80)

    print(
        "Training shape:",
        X_train.shape
    )

    print(
        "Testing shape :",
        X_test.shape
    )

    if X_train.shape[0] != len(train_raw):
        raise ValueError(
            "Training feature rows do not match training target rows."
        )

    if X_test.shape[0] != len(test_raw):
        raise ValueError(
            "Testing feature rows do not match testing target rows."
        )

    if X_train.shape[1] != 56:
        raise ValueError(
            "Expected 56 processed features."
        )

    # -----------------------------------------------------
    # Verify patient ordering
    # -----------------------------------------------------

    print()
    print("PATIENT ORDER VERIFICATION")
    print("-" * 80)

    # These raw files came from the same train/test split.
    # We verify their lengths and unique IDs before training.

    train_unique_ids = train_raw[
        "patient_id"
    ].nunique()

    test_unique_ids = test_raw[
        "patient_id"
    ].nunique()

    print(
        "Training unique IDs:",
        train_unique_ids
    )

    print(
        "Testing unique IDs:",
        test_unique_ids
    )

    if train_unique_ids != len(train_raw):
        raise ValueError(
            "Duplicate training patient IDs detected."
        )

    if test_unique_ids != len(test_raw):
        raise ValueError(
            "Duplicate testing patient IDs detected."
        )

    # -----------------------------------------------------
    # Targets
    # -----------------------------------------------------

    y_train = train_raw[
        "stage1_target"
    ].astype(int)

    y_test = test_raw[
        "stage1_target"
    ].astype(int)

    print()
    print("STAGE 1 TARGET")
    print("-" * 80)

    print(
        "Training:",
        y_train.value_counts()
        .sort_index()
        .to_dict()
    )

    print(
        "Testing :",
        y_test.value_counts()
        .sort_index()
        .to_dict()
    )

    # -----------------------------------------------------
    # Internal validation split
    # -----------------------------------------------------

    (
        X_inner_train,
        X_validation,
        y_inner_train,
        y_validation,
    ) = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    print()
    print("=" * 80)
    print("INTERNAL VALIDATION SPLIT")
    print("=" * 80)

    print(
        "Inner training:",
        X_inner_train.shape
    )

    print(
        "Validation:",
        X_validation.shape
    )

    # -----------------------------------------------------
    # Candidate model comparison
    # -----------------------------------------------------

    validation_results = []

    print()
    print("=" * 80)
    print("XGBOOST CANDIDATE EVALUATION")
    print("=" * 80)

    for config in CANDIDATES:

        print()
        print(
            f"Training {config['name']}..."
        )

        model = create_model(
            config
        )

        # IMPORTANT:
        # Class weights are computed from INNER TRAINING ONLY.
        sample_weights = compute_sample_weight(
            class_weight="balanced",
            y=y_inner_train,
        )

        model.fit(
            X_inner_train,
            y_inner_train,
            sample_weight=sample_weights,
        )

        validation_pred = model.predict(
            X_validation
        )

        validation_probability = model.predict_proba(
            X_validation
        )[:, 1]

        metrics = calculate_metrics(
            y_validation,
            validation_pred,
            validation_probability,
        )

        row = {
            "model": config["name"],
            "n_estimators": config["n_estimators"],
            "max_depth": config["max_depth"],
            "learning_rate": config["learning_rate"],
            "min_child_weight": config["min_child_weight"],
            "subsample": config["subsample"],
            "colsample_bytree": config["colsample_bytree"],
            "reg_lambda": config["reg_lambda"],
            **metrics,
        }

        validation_results.append(
            row
        )

        print(
            f"Accuracy: {metrics['accuracy']:.4f}"
        )

        print(
            f"Macro F1: {metrics['f1_macro']:.4f}"
        )

        print(
            f"ROC-AUC: {metrics['roc_auc']:.4f}"
        )

    # -----------------------------------------------------
    # Select best candidate
    # -----------------------------------------------------

    validation_df = pd.DataFrame(
        validation_results
    )

    validation_df = validation_df.sort_values(
        by="f1_macro",
        ascending=False,
    )

    print()
    print("=" * 80)
    print("VALIDATION MODEL COMPARISON")
    print("=" * 80)

    print(
        validation_df[
            [
                "model",
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "roc_auc",
            ]
        ].to_string(
            index=False
        )
    )

    best_name = validation_df.iloc[
        0
    ]["model"]

    best_config = next(
        config
        for config in CANDIDATES
        if config["name"] == best_name
    )

    print()
    print(
        "Selected model:",
        best_name
    )

    print(
        "Selection criterion:",
        "highest validation Macro F1"
    )

    # -----------------------------------------------------
    # Retrain selected model on ALL training data
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL STAGE 1 MODEL TRAINING")
    print("=" * 80)

    final_model = create_model(
        best_config
    )

    final_sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )

    final_model.fit(
        X_train,
        y_train,
        sample_weight=final_sample_weights,
    )

    # -----------------------------------------------------
    # Final test evaluation
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("UNTOUCHED TEST EVALUATION")
    print("=" * 80)

    test_pred = final_model.predict(
        X_test
    )

    test_probability = final_model.predict_proba(
        X_test
    )[:, 1]

    test_metrics = calculate_metrics(
        y_test,
        test_pred,
        test_probability,
    )

    print(
        f"Accuracy       : {test_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision Macro : {test_metrics['precision_macro']:.4f}"
    )

    print(
        f"Recall Macro    : {test_metrics['recall_macro']:.4f}"
    )

    print(
        f"F1 Macro        : {test_metrics['f1_macro']:.4f}"
    )

    print(
        f"ROC-AUC         : {test_metrics['roc_auc']:.4f}"
    )

    print()
    print("CLASSIFICATION REPORT")
    print("-" * 80)

    print(
        classification_report(
            y_test,
            test_pred,
            target_names=[
                "Normal",
                "Non-Normal",
            ],
            zero_division=0,
        )
    )

    print()
    print("CONFUSION MATRIX")
    print("-" * 80)

    confusion = confusion_matrix(
        y_test,
        test_pred,
    )

    print(
        confusion
    )

    # -----------------------------------------------------
    # Save model
    # -----------------------------------------------------

    MODEL_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        final_model,
        MODEL_OUTPUT,
    )

    # -----------------------------------------------------
    # Save validation results
    # -----------------------------------------------------

    validation_df.to_csv(
        VALIDATION_RESULTS_OUTPUT,
        index=False,
    )

    # -----------------------------------------------------
    # Save test metrics
    # -----------------------------------------------------

    pd.DataFrame(
        [
            {
                "model": best_name,
                **test_metrics,
            }
        ]
    ).to_csv(
        TEST_METRICS_OUTPUT,
        index=False,
    )

    # -----------------------------------------------------
    # Save test predictions
    # -----------------------------------------------------

    pd.DataFrame(
        {
            "actual_stage1": y_test,
            "predicted_stage1": test_pred,
            "non_normal_probability": test_probability,
        }
    ).to_csv(
        TEST_PREDICTIONS_OUTPUT,
        index=False,
    )

    # -----------------------------------------------------
    # Save confusion matrix
    # -----------------------------------------------------

    pd.DataFrame(
        confusion,
        index=[
            "Actual_Normal",
            "Actual_Non_Normal",
        ],
        columns=[
            "Predicted_Normal",
            "Predicted_Non_Normal",
        ],
    ).to_csv(
        TEST_CONFUSION_OUTPUT
    )

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 347 COMPLETED")
    print("=" * 80)

    print(
        "Selected XGBoost:",
        best_name
    )

    print(
        "Model saved:",
        MODEL_OUTPUT
    )

    print(
        "Validation results:",
        VALIDATION_RESULTS_OUTPUT
    )

    print(
        "Test metrics:",
        TEST_METRICS_OUTPUT
    )

    print(
        "Test predictions:",
        TEST_PREDICTIONS_OUTPUT
    )

    print(
        "Confusion matrix:",
        TEST_CONFUSION_OUTPUT
    )

    print()
    print("IMPORTANT:")
    print("- Class weights were used for Stage 1.")
    print("- Hyperparameter selection used INTERNAL VALIDATION.")
    print("- The 1835-record TEST SET was not used for model selection.")
    print("- The selected model was retrained on all 7337 training records.")
    print("- No SMOTE was applied to Stage 1.")
    print("- No confidence threshold was tuned yet.")
    print("- No reject/uncertain option has been implemented yet.")
    print("- No Stage 2 model was modified.")
    print("- No Flask code was modified.")


if __name__ == "__main__":
    main()