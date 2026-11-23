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
# ENHANCED CASCADE ABLATION STUDY
# Step 364
# ---------------------------------------------------------

X_TEST_FILE = Path(
    "data/enhanced_X_test_processed.csv"
)

TARGET_FILE = Path(
    "data/enhanced_test_with_targets.csv"
)

STAGE1_MODEL_FILE = Path(
    "model/enhanced_stage1_calibrated_model.pkl"
)

STAGE2_MODEL_FILE = Path(
    "model/enhanced_stage2_calibrated_stacking.pkl"
)

CASCADE_MODEL_FILE = Path(
    "model/final_enhanced_hierarchical_cascade.pkl"
)

OUTPUT_FILE = Path(
    "data/enhanced_cascade_ablation_results.csv"
)

STAGE2_CM_FILE = Path(
    "data/enhanced_ablation_stage2_only_confusion_matrix.csv"
)

CASCADE_CM_FILE = Path(
    "data/enhanced_ablation_cascade_confusion_matrix.csv"
)

EXPECTED_ROWS = 1835
EXPECTED_FEATURES = 56

CLASS_NAMES = [
    "Normal",
    "Hyperthyroid",
    "Hypothyroid",
    "Other / Related Diagnosis",
]


def load_true_targets():

    targets = pd.read_csv(
        TARGET_FILE
    )

    # Preferred numeric target
    if "stage2_target" in targets.columns:

        y_true = (
            targets[
                "stage2_target"
            ]
            .astype(int)
            .to_numpy()
        )

        return y_true

    # Fallback to stage2 label
    if "stage2_label" in targets.columns:

        mapping = {
            "Normal": 0,
            "Hyperthyroid": 1,
            "Hypothyroid": 2,
            "Other / Related Diagnosis": 3,
        }

        y_true = (
            targets[
                "stage2_label"
            ]
            .map(mapping)
        )

        if y_true.isna().any():

            raise ValueError(
                "Unknown value found in stage2_label."
            )

        return y_true.astype(int).to_numpy()

    raise ValueError(
        "Could not find stage2_target or stage2_label."
    )


def metrics(y_true, y_pred):

    return {
        "accuracy":
            accuracy_score(
                y_true,
                y_pred,
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred,
            ),

        "precision_macro":
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),

        "recall_macro":
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),

        "f1_macro":
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),

        "f1_weighted":
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
    }


def print_metrics(title, result):

    print()
    print(title)
    print("-" * 80)

    print(
        f"Accuracy          : "
        f"{result['accuracy']:.6f}"
    )

    print(
        f"Balanced Accuracy : "
        f"{result['balanced_accuracy']:.6f}"
    )

    print(
        f"Macro Precision   : "
        f"{result['precision_macro']:.6f}"
    )

    print(
        f"Macro Recall      : "
        f"{result['recall_macro']:.6f}"
    )

    print(
        f"Macro F1          : "
        f"{result['f1_macro']:.6f}"
    )

    print(
        f"Weighted F1       : "
        f"{result['f1_weighted']:.6f}"
    )


def main():

    print("=" * 80)
    print(
        "TCMAPY2573 - ENHANCED CASCADE ABLATION STUDY"
    )
    print("=" * 80)

    # -----------------------------------------------------
    # File validation
    # -----------------------------------------------------

    print()
    print("FILE CHECK")
    print("-" * 80)

    required_files = [
        X_TEST_FILE,
        TARGET_FILE,
        STAGE1_MODEL_FILE,
        STAGE2_MODEL_FILE,
        CASCADE_MODEL_FILE,
    ]

    for path in required_files:

        print(
            f"{path}: {path.exists()}"
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    # -----------------------------------------------------
    # Load test data
    # -----------------------------------------------------

    X_test = pd.read_csv(
        X_TEST_FILE
    )

    y_true = load_true_targets()

    if X_test.shape != (
        EXPECTED_ROWS,
        EXPECTED_FEATURES,
    ):

        raise ValueError(
            f"Unexpected X_test shape: "
            f"{X_test.shape}"
        )

    if len(y_true) != EXPECTED_ROWS:

        raise ValueError(
            f"Unexpected target length: "
            f"{len(y_true)}"
        )

    print()
    print("TEST DATA")
    print("-" * 80)

    print(
        "Rows:",
        len(X_test)
    )

    print(
        "Features:",
        X_test.shape[1]
    )

    print(
        "True target rows:",
        len(y_true)
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

    cascade = joblib.load(
        CASCADE_MODEL_FILE
    )

    print(
        "Stage 1:",
        type(stage1).__name__
    )

    print(
        "Stage 2:",
        type(stage2).__name__
    )

    print(
        "Cascade:",
        type(cascade).__name__
    )

    # -----------------------------------------------------
    # Configuration A
    # Stage 2 without confidence router
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print(
        "ABLATION A - STAGE 2 WITHOUT CONFIDENCE ROUTER"
    )
    print("=" * 80)

    print(
        "All 1835 test records are passed directly "
        "to Stage 2."
    )

    stage2_only_predictions = (
        stage2.predict(
            X_test
        )
        .astype(int)
    )

    stage2_metrics = metrics(
        y_true,
        stage2_only_predictions,
    )

    print_metrics(
        "STAGE 2 ONLY RESULTS",
        stage2_metrics,
    )

    stage2_cm = confusion_matrix(
        y_true,
        stage2_only_predictions,
        labels=[0, 1, 2, 3],
    )

    print()
    print("STAGE 2 ONLY CONFUSION MATRIX")
    print("-" * 80)

    print(
        stage2_cm
    )

    pd.DataFrame(
        stage2_cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    ).to_csv(
        STAGE2_CM_FILE
    )

    # -----------------------------------------------------
    # Configuration B
    # Full proposed cascade
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print(
        "ABLATION B - FULL CONFIDENCE-GUIDED CASCADE"
    )
    print("=" * 80)

    print(
        "Stage 1 confidence routing is enabled."
    )

    print(
        "Confidence threshold:",
        cascade.confidence_threshold
    )

    cascade_predictions = (
        cascade.predict(
            X_test
        )
        .astype(int)
    )

    cascade_metrics = metrics(
        y_true,
        cascade_predictions,
    )

    print_metrics(
        "FULL CASCADE RESULTS",
        cascade_metrics,
    )

    cascade_cm = confusion_matrix(
        y_true,
        cascade_predictions,
        labels=[0, 1, 2, 3],
    )

    print()
    print("FULL CASCADE CONFUSION MATRIX")
    print("-" * 80)

    print(
        cascade_cm
    )

    pd.DataFrame(
        cascade_cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    ).to_csv(
        CASCADE_CM_FILE
    )

    # -----------------------------------------------------
    # Component-level comparison
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print(
        "COMPONENT COMPARISON"
    )
    print("=" * 80)

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
    ]

    rows = []

    for metric_name in metric_names:

        stage2_value = (
            stage2_metrics[
                metric_name
            ]
        )

        cascade_value = (
            cascade_metrics[
                metric_name
            ]
        )

        difference = (
            cascade_value
            - stage2_value
        )

        rows.append(
            {
                "metric":
                    metric_name,

                "stage2_without_router":
                    stage2_value,

                "full_confidence_cascade":
                    cascade_value,

                "cascade_minus_stage2":
                    difference,
            }
        )

        print(
            f"{metric_name:22} "
            f"Stage2={stage2_value:.6f} "
            f"Cascade={cascade_value:.6f} "
            f"Difference={difference:+.6f}"
        )

    # -----------------------------------------------------
    # Stage 1 binary analysis
    # -----------------------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "STAGE 1 BINARY COMPONENT"
    )

    print(
        "=" * 80
    )

    stage1_predictions = (
        stage1.predict(
            X_test
        )
        .astype(int)
    )

    y_stage1_true = np.where(
        y_true == 0,
        0,
        1,
    )

    stage1_binary_accuracy = (
        accuracy_score(
            y_stage1_true,
            stage1_predictions,
        )
    )

    stage1_binary_f1 = (
        f1_score(
            y_stage1_true,
            stage1_predictions,
            average="macro",
            zero_division=0,
        )
    )

    print(
        f"Stage 1 Accuracy: "
        f"{stage1_binary_accuracy:.6f}"
    )

    print(
        f"Stage 1 Macro F1: "
        f"{stage1_binary_f1:.6f}"
    )

    # -----------------------------------------------------
    # Direct normal routing
    # -----------------------------------------------------

    print()
    print(
        "ROUTING COMPONENT"
    )

    print(
        "-" * 80
    )

    routing = pd.read_csv(
        Path(
            "data/enhanced_cascade_routing_analysis.csv"
        )
    )

    if "routing_decision" in routing.columns:

        direct_normal = int(
            (
                routing[
                    "routing_decision"
                ]
                == "FINAL_NORMAL"
            ).sum()
        )

        stage2_cases = (
            EXPECTED_ROWS
            - direct_normal
        )

        print(
            "Direct Final Normal:",
            direct_normal
        )

        print(
            "Stage 2 routed:",
            stage2_cases
        )

        print(
            f"Direct Normal rate: "
            f"{direct_normal / EXPECTED_ROWS * 100:.2f}%"
        )

        print(
            f"Stage 2 routing rate: "
            f"{stage2_cases / EXPECTED_ROWS * 100:.2f}%"
        )

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    result_df = pd.DataFrame(
        rows
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # Final completion
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print(
        "STEP 364 COMPLETED"
    )
    print("=" * 80)

    print(
        "Ablation study saved:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print(
        "Stage 2-only confusion matrix:"
    )

    print(
        STAGE2_CM_FILE
    )

    print()
    print(
        "Full cascade confusion matrix:"
    )

    print(
        CASCADE_CM_FILE
    )

    print()
    print("IMPORTANT:")
    print("- No model was retrained.")
    print("- No model was modified.")
    print("- Confidence threshold remained 0.95.")
    print("- The test set was used only for evaluation.")
    print("- No configuration was selected or tuned from these results.")
    print("- This step measures the contribution of the confidence router.")


if __name__ == "__main__":
    main()