from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ---------------------------------------------------------
# TCMAPY2573
# FINAL ENHANCED STAGE 2 EVALUATION
# Step 357
# ---------------------------------------------------------

MODEL_FILE = Path(
    "model/enhanced_stage2_calibrated_stacking.pkl"
)

X_TEST_FILE = Path(
    "data/enhanced_X_test_processed.csv"
)

Y_TEST_SOURCE = Path(
    "data/enhanced_test_with_targets.csv"
)

METRICS_FILE = Path(
    "data/enhanced_stage2_final_metrics.csv"
)

PREDICTIONS_FILE = Path(
    "data/enhanced_stage2_test_predictions.csv"
)

CONFUSION_FILE = Path(
    "data/enhanced_stage2_final_confusion_matrix.csv"
)

REPORT_FILE = Path(
    "data/enhanced_stage2_classification_report.csv"
)

PROBABILITY_FILE = Path(
    "data/enhanced_stage2_test_probabilities.csv"
)


CLASS_NAMES = {
    0: "Normal",
    1: "Hyperthyroid",
    2: "Hypothyroid",
    3: "Other / Related Diagnosis",
}


EXPECTED_TEST_ROWS = 1835
EXPECTED_FEATURES = 56


def multiclass_brier_score(
    y_true,
    probabilities,
    classes,
):
    """
    Multiclass Brier score using one-hot targets.
    Lower is better.
    """

    y_true = np.asarray(
        y_true
    )

    probabilities = np.asarray(
        probabilities
    )

    classes = list(classes)

    one_hot = np.zeros_like(
        probabilities
    )

    for row_index, label in enumerate(
        y_true
    ):

        class_index = classes.index(
            label
        )

        one_hot[
            row_index,
            class_index,
        ] = 1.0

    return float(
        np.mean(
            np.sum(
                (
                    probabilities
                    - one_hot
                ) ** 2,
                axis=1,
            )
        )
    )


def main():

    print("=" * 80)
    print(
        "TCMAPY2573 - FINAL ENHANCED STAGE 2 EVALUATION"
    )
    print("=" * 80)

    # -----------------------------------------------------
    # File checks
    # -----------------------------------------------------

    required_files = [
        MODEL_FILE,
        X_TEST_FILE,
        Y_TEST_SOURCE,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    # -----------------------------------------------------
    # Load model and test data
    # -----------------------------------------------------

    print()
    print("LOADING FINAL TEST SET")
    print("-" * 80)

    model = joblib.load(
        MODEL_FILE
    )

    X_test = pd.read_csv(
        X_TEST_FILE
    )

    test_source = pd.read_csv(
        Y_TEST_SOURCE
    )

    y_test = (
        test_source[
            "stage2_target"
        ]
        .astype(int)
    )

    print(
        "Model:",
        type(model).__name__
    )

    print(
        "X_test:",
        X_test.shape
    )

    print(
        "y_test:",
        y_test.shape
    )

    # -----------------------------------------------------
    # Test-set safety checks
    # -----------------------------------------------------

    if X_test.shape != (
        EXPECTED_TEST_ROWS,
        EXPECTED_FEATURES,
    ):

        raise ValueError(
            "Unexpected final test feature shape."
        )

    if len(y_test) != EXPECTED_TEST_ROWS:

        raise ValueError(
            "Unexpected final test target length."
        )

    missing = int(
        X_test.isna().sum().sum()
    )

    infinite = int(
        np.isinf(
            X_test.to_numpy(
                dtype=float
            )
        ).sum()
    )

    print()
    print("TEST DATA CHECK")
    print("-" * 80)

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
            "Final test data contains missing values."
        )

    if infinite != 0:
        raise ValueError(
            "Final test data contains infinite values."
        )

    # -----------------------------------------------------
    # Verify test class distribution
    # -----------------------------------------------------

    print()
    print("FINAL TEST CLASS DISTRIBUTION")
    print("-" * 80)

    test_counts = (
        y_test
        .value_counts()
        .sort_index()
    )

    for class_id in CLASS_NAMES:

        print(
            f"{CLASS_NAMES[class_id]:30} "
            f"{int(test_counts.get(class_id, 0))}"
        )

    # -----------------------------------------------------
    # Predict
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL PREDICTION")
    print("=" * 80)

    predictions = model.predict(
        X_test
    )

    probabilities = (
        model.predict_proba(
            X_test
        )
    )

    print(
        "Prediction complete."
    )

    print(
        "Probability matrix:",
        probabilities.shape
    )

    # -----------------------------------------------------
    # Basic metrics
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            predictions,
        )
    )

    precision_macro = (
        precision_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    recall_macro = (
        recall_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    f1_macro = (
        f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    f1_weighted = (
        f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )
    )

    # -----------------------------------------------------
    # Probability metrics
    # -----------------------------------------------------

    logloss = log_loss(
        y_test,
        probabilities,
        labels=model.classes_,
    )

    brier = multiclass_brier_score(
        y_test,
        probabilities,
        list(model.classes_),
    )

    try:

        roc_auc_ovr = roc_auc_score(
            y_test,
            probabilities,
            multi_class="ovr",
            average="macro",
            labels=model.classes_,
        )

    except ValueError:

        roc_auc_ovr = np.nan

    # -----------------------------------------------------
    # Classification report
    # -----------------------------------------------------

    report = classification_report(
        y_test,
        predictions,
        labels=list(model.classes_),
        target_names=[
            CLASS_NAMES[int(x)]
            for x in model.classes_
        ],
        output_dict=True,
        zero_division=0,
    )

    report_df = (
        pd.DataFrame(report)
        .transpose()
    )

    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=list(model.classes_),
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            CLASS_NAMES[int(x)]
            for x in model.classes_
        ],
        columns=[
            CLASS_NAMES[int(x)]
            for x in model.classes_
        ],
    )

    # -----------------------------------------------------
    # Prediction table
    # -----------------------------------------------------

    prediction_df = pd.DataFrame(
        {
            "actual_class": y_test.to_numpy(),
            "predicted_class": predictions,
            "actual_label": [
                CLASS_NAMES[int(x)]
                for x in y_test
            ],
            "predicted_label": [
                CLASS_NAMES[int(x)]
                for x in predictions
            ],
        }
    )

    # -----------------------------------------------------
    # Probability table
    # -----------------------------------------------------

    probability_data = {}

    for index, class_id in enumerate(
        model.classes_
    ):

        probability_data[
            f"probability_class_{int(class_id)}"
        ] = probabilities[
            :,
            index,
        ]

    probability_df = pd.DataFrame(
        probability_data
    )

    # -----------------------------------------------------
    # Combined probability + prediction
    # -----------------------------------------------------

    probability_df.insert(
        0,
        "actual_class",
        y_test.to_numpy(),
    )

    probability_df.insert(
        1,
        "predicted_class",
        predictions,
    )

    # -----------------------------------------------------
    # Save metrics
    # -----------------------------------------------------

    metrics_df = pd.DataFrame(
        [
            {
                "accuracy": accuracy,
                "balanced_accuracy": (
                    balanced_accuracy
                ),
                "precision_macro": (
                    precision_macro
                ),
                "recall_macro": (
                    recall_macro
                ),
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted,
                "roc_auc_ovr_macro": (
                    roc_auc_ovr
                ),
                "log_loss": logloss,
                "multiclass_brier_score": brier,
                "test_samples": len(y_test),
                "features": X_test.shape[1],
            }
        ]
    )

    # -----------------------------------------------------
    # Save all outputs
    # -----------------------------------------------------

    METRICS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_df.to_csv(
        METRICS_FILE,
        index=False,
    )

    prediction_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    cm_df.to_csv(
        CONFUSION_FILE
    )

    report_df.to_csv(
        REPORT_FILE
    )

    probability_df.to_csv(
        PROBABILITY_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # Print final metrics
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL STAGE 2 METRICS")
    print("=" * 80)

    print(
        f"Accuracy             : {accuracy:.6f}"
    )

    print(
        f"Balanced Accuracy    : {balanced_accuracy:.6f}"
    )

    print(
        f"Precision Macro      : {precision_macro:.6f}"
    )

    print(
        f"Recall Macro         : {recall_macro:.6f}"
    )

    print(
        f"F1 Macro             : {f1_macro:.6f}"
    )

    print(
        f"F1 Weighted          : {f1_weighted:.6f}"
    )

    print(
        f"ROC-AUC OVR Macro    : {roc_auc_ovr:.6f}"
    )

    print(
        f"Log Loss             : {logloss:.6f}"
    )

    print(
        f"Multiclass Brier     : {brier:.6f}"
    )

    # -----------------------------------------------------
    # Classification report
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("CLASSIFICATION REPORT")
    print("=" * 80)

    print(
        classification_report(
            y_test,
            predictions,
            labels=list(model.classes_),
            target_names=[
                CLASS_NAMES[int(x)]
                for x in model.classes_
            ],
            zero_division=0,
        )
    )

    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("CONFUSION MATRIX")
    print("=" * 80)

    print(
        cm
    )

    print()
    print("=" * 80)
    print("STEP 357 COMPLETED")
    print("=" * 80)

    print(
        "Final metrics:",
        METRICS_FILE,
    )

    print(
        "Predictions:",
        PREDICTIONS_FILE,
    )

    print(
        "Confusion matrix:",
        CONFUSION_FILE,
    )

    print(
        "Classification report:",
        REPORT_FILE,
    )

    print(
        "Probabilities:",
        PROBABILITY_FILE,
    )

    print()
    print("IMPORTANT:")
    print("- This is the first evaluation on the untouched final test set.")
    print("- Test data was not used for training.")
    print("- Test data was not used for calibration.")
    print("- Test data was not used for threshold tuning.")
    print("- Do NOT tune the model using these results.")
    print("- No Flask code was modified.")


if __name__ == "__main__":
    main()