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
    precision_score,
    recall_score,
)


# ---------------------------------------------------------
# TCMAPY2573
# FINAL ENHANCED CONFIDENCE-GUIDED CASCADE EVALUATION
# Step 359
# ---------------------------------------------------------

CASCADE_FILE = Path(
    "model/enhanced_hierarchical_cascade.py"
)

STAGE1_MODEL_FILE = Path(
    "model/enhanced_stage1_calibrated_model.pkl"
)

STAGE2_MODEL_FILE = Path(
    "model/enhanced_stage2_calibrated_stacking.pkl"
)

X_TEST_FILE = Path(
    "data/enhanced_X_test_processed.csv"
)

TEST_TARGET_FILE = Path(
    "data/enhanced_test_with_targets.csv"
)

OUTPUT_METRICS = Path(
    "data/enhanced_cascade_final_metrics.csv"
)

OUTPUT_PREDICTIONS = Path(
    "data/enhanced_cascade_test_predictions.csv"
)

OUTPUT_ROUTING = Path(
    "data/enhanced_cascade_routing_analysis.csv"
)

OUTPUT_CONFUSION = Path(
    "data/enhanced_cascade_confusion_matrix.csv"
)

OUTPUT_REPORT = Path(
    "data/enhanced_cascade_classification_report.csv"
)


CONFIDENCE_THRESHOLD = 0.95

EXPECTED_TEST_ROWS = 1835
EXPECTED_FEATURES = 56


CLASS_NAMES = {
    0: "Normal",
    1: "Hyperthyroid",
    2: "Hypothyroid",
    3: "Other / Related Diagnosis",
}


def main():

    print("=" * 80)
    print(
        "TCMAPY2573 - FINAL ENHANCED CASCADE EVALUATION"
    )
    print("=" * 80)

    # -----------------------------------------------------
    # File checks
    # -----------------------------------------------------

    required_files = [
        STAGE1_MODEL_FILE,
        STAGE2_MODEL_FILE,
        X_TEST_FILE,
        TEST_TARGET_FILE,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    # -----------------------------------------------------
    # Load models
    # -----------------------------------------------------

    print()
    print("MODEL LOADING")
    print("-" * 80)

    stage1 = joblib.load(
        STAGE1_MODEL_FILE
    )

    stage2 = joblib.load(
        STAGE2_MODEL_FILE
    )

    print(
        "Stage 1:",
        type(stage1).__name__
    )

    print(
        "Stage 2:",
        type(stage2).__name__
    )

    # -----------------------------------------------------
    # Load test data
    # -----------------------------------------------------

    X_test = pd.read_csv(
        X_TEST_FILE
    )

    test_data = pd.read_csv(
        TEST_TARGET_FILE
    )

    y_test = (
        test_data[
            "stage2_target"
        ]
        .astype(int)
        .to_numpy()
    )

    y_stage1 = (
        test_data[
            "stage1_target"
        ]
        .astype(int)
        .to_numpy()
    )

    # -----------------------------------------------------
    # Test validation
    # -----------------------------------------------------

    print()
    print("TEST DATA VALIDATION")
    print("-" * 80)

    print(
        "X_test shape:",
        X_test.shape
    )

    print(
        "y_test length:",
        len(y_test)
    )

    print(
        "Expected rows:",
        EXPECTED_TEST_ROWS
    )

    print(
        "Expected features:",
        EXPECTED_FEATURES
    )

    if X_test.shape != (
        EXPECTED_TEST_ROWS,
        EXPECTED_FEATURES,
    ):

        raise ValueError(
            "Unexpected test-set shape."
        )

    if len(y_test) != EXPECTED_TEST_ROWS:

        raise ValueError(
            "Unexpected test target length."
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
            "Missing values found in test data."
        )

    if infinite != 0:
        raise ValueError(
            "Infinite values found in test data."
        )

    # -----------------------------------------------------
    # Stage 1 calibrated probabilities
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("STAGE 1 CALIBRATED SCREENING")
    print("=" * 80)

    stage1_probabilities = (
        stage1.predict_proba(
            X_test
        )[:, 1]
    )

    probability_normal = (
        1.0
        - stage1_probabilities
    )

    # -----------------------------------------------------
    # Confidence-guided routing
    # -----------------------------------------------------

    normal_mask = (
        stage1_probabilities
        <= 1.0 - CONFIDENCE_THRESHOLD
    )

    stage2_mask = (
        ~normal_mask
    )

    confident_normal_count = int(
        normal_mask.sum()
    )

    stage2_count = int(
        stage2_mask.sum()
    )

    uncertain_mask = (
        (
            stage1_probabilities
            > 1.0 - CONFIDENCE_THRESHOLD
        )
        &
        (
            stage1_probabilities
            < CONFIDENCE_THRESHOLD
        )
    )

    confident_non_normal_mask = (
        stage1_probabilities
        >= CONFIDENCE_THRESHOLD
    )

    uncertain_count = int(
        uncertain_mask.sum()
    )

    confident_non_normal_count = int(
        confident_non_normal_mask.sum()
    )

    print(
        f"Confidence threshold: "
        f"{CONFIDENCE_THRESHOLD:.2f}"
    )

    print(
        f"Final Normal routing: "
        f"{confident_normal_count}"
    )

    print(
        f"Stage 2 routing: "
        f"{stage2_count}"
    )

    print(
        f"Uncertain cases: "
        f"{uncertain_count}"
    )

    print(
        f"Confident Non-Normal cases: "
        f"{confident_non_normal_count}"
    )

    # -----------------------------------------------------
    # Final predictions
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("END-TO-END CASCADE PREDICTION")
    print("=" * 80)

    final_predictions = np.zeros(
        EXPECTED_TEST_ROWS,
        dtype=int,
    )

    # Confident Normal cases
    final_predictions[
        normal_mask
    ] = 0

    # Stage 2 cases
    if stage2_count > 0:

        stage2_predictions = (
            stage2.predict(
                X_test[
                    stage2_mask
                ]
            )
        )

        final_predictions[
            stage2_mask
        ] = stage2_predictions

    print(
        "End-to-end prediction completed."
    )

    # -----------------------------------------------------
    # Stage 1 screening evaluation
    # -----------------------------------------------------

    stage1_predictions = (
        stage1_probabilities
        >= 0.50
    ).astype(int)

    stage1_accuracy = accuracy_score(
        y_stage1,
        stage1_predictions,
    )

    stage1_f1 = f1_score(
        y_stage1,
        stage1_predictions,
        average="macro",
        zero_division=0,
    )

    # -----------------------------------------------------
    # Final cascade metrics
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        final_predictions,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            final_predictions,
        )
    )

    precision_macro = (
        precision_score(
            y_test,
            final_predictions,
            average="macro",
            zero_division=0,
        )
    )

    recall_macro = (
        recall_score(
            y_test,
            final_predictions,
            average="macro",
            zero_division=0,
        )
    )

    f1_macro = (
        f1_score(
            y_test,
            final_predictions,
            average="macro",
            zero_division=0,
        )
    )

    f1_weighted = (
        f1_score(
            y_test,
            final_predictions,
            average="weighted",
            zero_division=0,
        )
    )

    # -----------------------------------------------------
    # Routing error analysis
    # -----------------------------------------------------

    confident_normal_correct = int(
        (
            normal_mask
            &
            (y_test == 0)
        ).sum()
    )

    confident_normal_incorrect = int(
        (
            normal_mask
            &
            (y_test != 0)
        ).sum()
    )

    confident_normal_precision = (
        confident_normal_correct
        / confident_normal_count
        if confident_normal_count > 0
        else 0.0
    )

    # -----------------------------------------------------
    # Classification report
    # -----------------------------------------------------

    report = classification_report(
        y_test,
        final_predictions,
        labels=[0, 1, 2, 3],
        target_names=[
            CLASS_NAMES[0],
            CLASS_NAMES[1],
            CLASS_NAMES[2],
            CLASS_NAMES[3],
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
        final_predictions,
        labels=[0, 1, 2, 3],
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            CLASS_NAMES[i]
            for i in range(4)
        ],
        columns=[
            CLASS_NAMES[i]
            for i in range(4)
        ],
    )

    # -----------------------------------------------------
    # Routing data
    # -----------------------------------------------------

    routing_labels = np.full(
        EXPECTED_TEST_ROWS,
        "STAGE_2",
        dtype=object,
    )

    routing_labels[
        normal_mask
    ] = "FINAL_NORMAL"

    routing_df = pd.DataFrame(
        {
            "stage1_probability_non_normal":
                stage1_probabilities,
            "stage1_probability_normal":
                probability_normal,
            "routing_decision":
                routing_labels,
            "actual_stage1":
                y_stage1,
            "actual_stage2":
                y_test,
            "final_prediction":
                final_predictions,
        }
    )

    # -----------------------------------------------------
    # Prediction data
    # -----------------------------------------------------

    prediction_df = pd.DataFrame(
        {
            "actual_class":
                y_test,
            "predicted_class":
                final_predictions,
            "actual_label": [
                CLASS_NAMES[int(x)]
                for x in y_test
            ],
            "predicted_label": [
                CLASS_NAMES[int(x)]
                for x in final_predictions
            ],
            "stage1_probability_non_normal":
                stage1_probabilities,
            "stage1_probability_normal":
                probability_normal,
            "stage2_used":
                stage2_mask,
        }
    )

    # -----------------------------------------------------
    # Metrics dataframe
    # -----------------------------------------------------

    metrics_df = pd.DataFrame(
        [
            {
                "stage1_accuracy":
                    stage1_accuracy,
                "stage1_macro_f1":
                    stage1_f1,
                "cascade_accuracy":
                    accuracy,
                "cascade_balanced_accuracy":
                    balanced_accuracy,
                "cascade_precision_macro":
                    precision_macro,
                "cascade_recall_macro":
                    recall_macro,
                "cascade_f1_macro":
                    f1_macro,
                "cascade_f1_weighted":
                    f1_weighted,
                "confidence_threshold":
                    CONFIDENCE_THRESHOLD,
                "final_test_samples":
                    EXPECTED_TEST_ROWS,
                "final_normal_count":
                    confident_normal_count,
                "stage2_count":
                    stage2_count,
                "uncertain_count":
                    uncertain_count,
                "confident_non_normal_count":
                    confident_non_normal_count,
                "confident_normal_correct":
                    confident_normal_correct,
                "confident_normal_incorrect":
                    confident_normal_incorrect,
                "confident_normal_precision":
                    confident_normal_precision,
            }
        ]
    )

    # -----------------------------------------------------
    # Save outputs
    # -----------------------------------------------------

    OUTPUT_METRICS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_df.to_csv(
        OUTPUT_METRICS,
        index=False,
    )

    prediction_df.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
    )

    routing_df.to_csv(
        OUTPUT_ROUTING,
        index=False,
    )

    cm_df.to_csv(
        OUTPUT_CONFUSION
    )

    report_df.to_csv(
        OUTPUT_REPORT
    )

    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL CASCADE METRICS")
    print("=" * 80)

    print(
        f"Stage 1 Accuracy        : "
        f"{stage1_accuracy:.6f}"
    )

    print(
        f"Stage 1 Macro F1        : "
        f"{stage1_f1:.6f}"
    )

    print(
        f"Cascade Accuracy        : "
        f"{accuracy:.6f}"
    )

    print(
        f"Balanced Accuracy       : "
        f"{balanced_accuracy:.6f}"
    )

    print(
        f"Macro Precision         : "
        f"{precision_macro:.6f}"
    )

    print(
        f"Macro Recall            : "
        f"{recall_macro:.6f}"
    )

    print(
        f"Macro F1                : "
        f"{f1_macro:.6f}"
    )

    print(
        f"Weighted F1             : "
        f"{f1_weighted:.6f}"
    )

    print()
    print("=" * 80)
    print("ROUTING SUMMARY")
    print("=" * 80)

    print(
        f"Final Normal            : "
        f"{confident_normal_count}"
    )

    print(
        f"Stage 2 used            : "
        f"{stage2_count}"
    )

    print(
        f"Uncertain cases          : "
        f"{uncertain_count}"
    )

    print(
        f"Confident Non-Normal    : "
        f"{confident_non_normal_count}"
    )

    print(
        f"Final Normal correct    : "
        f"{confident_normal_correct}"
    )

    print(
        f"Final Normal incorrect  : "
        f"{confident_normal_incorrect}"
    )

    print(
        f"Confident Normal precision: "
        f"{confident_normal_precision:.6f}"
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
            final_predictions,
            labels=[0, 1, 2, 3],
            target_names=[
                CLASS_NAMES[0],
                CLASS_NAMES[1],
                CLASS_NAMES[2],
                CLASS_NAMES[3],
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
    print("STEP 359 COMPLETED")
    print("=" * 80)

    print(
        "Metrics:",
        OUTPUT_METRICS
    )

    print(
        "Predictions:",
        OUTPUT_PREDICTIONS
    )

    print(
        "Routing analysis:",
        OUTPUT_ROUTING
    )

    print(
        "Confusion matrix:",
        OUTPUT_CONFUSION
    )

    print(
        "Classification report:",
        OUTPUT_REPORT
    )

    print()
    print("IMPORTANT:")
    print("- The 1835-record final test set was used only here.")
    print("- No test-set tuning was performed.")
    print("- Confidence threshold remained 0.95.")
    print("- Stage 1 and Stage 2 models were not modified.")
    print("- This is the end-to-end enhanced cascade evaluation.")


if __name__ == "__main__":
    main()