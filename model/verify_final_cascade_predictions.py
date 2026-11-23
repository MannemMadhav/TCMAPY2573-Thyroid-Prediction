from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# TCMAPY2573
# FINAL CASCADE PREDICTION CONSISTENCY VERIFICATION
# Step 362
# ---------------------------------------------------------

MODEL_FILE = Path(
    "model/final_enhanced_hierarchical_cascade.pkl"
)

X_TEST_FILE = Path(
    "data/enhanced_X_test_processed.csv"
)

REFERENCE_FILE = Path(
    "data/enhanced_cascade_test_predictions.csv"
)

EXPECTED_ROWS = 1835
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
        "TCMAPY2573 - FINAL CASCADE PREDICTION CONSISTENCY CHECK"
    )
    print("=" * 80)

    # -----------------------------------------------------
    # File checks
    # -----------------------------------------------------

    print()
    print("FILE CHECK")
    print("-" * 80)

    for path in [
        MODEL_FILE,
        X_TEST_FILE,
        REFERENCE_FILE,
    ]:

        print(
            f"{path}: {path.exists()}"
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    # -----------------------------------------------------
    # Load final saved cascade
    # -----------------------------------------------------

    print()
    print("MODEL LOADING")
    print("-" * 80)

    cascade = joblib.load(
        MODEL_FILE
    )

    print(
        "Loaded model:",
        type(cascade).__name__
    )

    print(
        "Confidence threshold:",
        cascade.confidence_threshold
    )

    # -----------------------------------------------------
    # Load test data
    # -----------------------------------------------------

    X_test = pd.read_csv(
        X_TEST_FILE
    )

    reference = pd.read_csv(
        REFERENCE_FILE
    )

    if X_test.shape != (
        EXPECTED_ROWS,
        EXPECTED_FEATURES,
    ):

        raise ValueError(
            f"Unexpected X_test shape: "
            f"{X_test.shape}"
        )

    if len(reference) != EXPECTED_ROWS:

        raise ValueError(
            "Unexpected number of reference predictions."
        )

    print()
    print("TEST DATA")
    print("-" * 80)

    print(
        "X_test shape:",
        X_test.shape
    )

    print(
        "Reference prediction rows:",
        len(reference)
    )

    # -----------------------------------------------------
    # Reproduce predictions
    # -----------------------------------------------------

    print()
    print("RE-COMPUTING PREDICTIONS")
    print("-" * 80)

    predictions = cascade.predict(
        X_test
    )

    print(
        "Prediction completed."
    )

    # -----------------------------------------------------
    # Compare predictions
    # -----------------------------------------------------

    if "predicted_class" not in reference.columns:

        raise ValueError(
            "Reference file does not contain "
            "'predicted_class'."
        )

    reference_predictions = (
        reference[
            "predicted_class"
        ]
        .astype(int)
        .to_numpy()
    )

    mismatch_mask = (
        predictions
        != reference_predictions
    )

    mismatch_count = int(
        mismatch_mask.sum()
    )

    match_count = (
        EXPECTED_ROWS
        - mismatch_count
    )

    print()
    print("PREDICTION COMPARISON")
    print("-" * 80)

    print(
        "Reference predictions:",
        len(reference_predictions)
    )

    print(
        "Saved-model predictions:",
        len(predictions)
    )

    print(
        "Matching predictions:",
        match_count
    )

    print(
        "Mismatching predictions:",
        mismatch_count
    )

    consistency = (
        match_count
        / EXPECTED_ROWS
    )

    print(
        f"Prediction consistency: "
        f"{consistency:.6f}"
    )

    # -----------------------------------------------------
    # Show mismatches if any
    # -----------------------------------------------------

    if mismatch_count > 0:

        print()
        print(
            "FIRST MISMATCHES"
        )
        print("-" * 80)

        mismatch_indices = np.where(
            mismatch_mask
        )[0]

        for index in mismatch_indices[:20]:

            reference_class = int(
                reference_predictions[
                    index
                ]
            )

            saved_class = int(
                predictions[
                    index
                ]
            )

            print(
                f"Row {index}: "
                f"reference="
                f"{reference_class} "
                f"({CLASS_NAMES.get(reference_class, 'Unknown')}) "
                f"saved="
                f"{saved_class} "
                f"({CLASS_NAMES.get(saved_class, 'Unknown')})"
            )

        raise ValueError(
            "Saved cascade predictions do not "
            "match the Step 359 reference predictions."
        )

    # -----------------------------------------------------
    # Prediction distribution
    # -----------------------------------------------------

    print()
    print("PREDICTION DISTRIBUTION")
    print("-" * 80)

    counts = (
        pd.Series(
            predictions
        )
        .value_counts()
        .sort_index()
    )

    for class_id in range(4):

        count = int(
            counts.get(
                class_id,
                0
            )
        )

        print(
            f"{CLASS_NAMES[class_id]:30} "
            f"{count}"
        )

    # -----------------------------------------------------
    # Final verification
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 362 VERIFICATION PASSED")
    print("=" * 80)

    print(
        "Saved final cascade reproduces "
        "the Step 359 predictions exactly."
    )

    print(
        "Prediction consistency: 100%"
    )

    print()
    print("IMPORTANT:")
    print("- No training was performed.")
    print("- No model was modified.")
    print("- No threshold was changed.")
    print("- This is a consistency check only.")
    print("- The final test predictions were not used for tuning.")


if __name__ == "__main__":
    main()