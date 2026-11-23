from pathlib import Path

import joblib

from enhanced_hierarchical_cascade import (
    EnhancedHierarchicalCascade,
)


# ---------------------------------------------------------
# TCMAPY2573
# FINAL ENHANCED CASCADE MODEL SAVER
# Step 360
# ---------------------------------------------------------

OUTPUT_FILE = Path(
    "model/final_enhanced_hierarchical_cascade.pkl"
)

CONFIDENCE_THRESHOLD = 0.95


def main():

    print("=" * 80)
    print(
        "TCMAPY2573 - FINAL ENHANCED CASCADE MODEL SAVER"
    )
    print("=" * 80)

    # -----------------------------------------------------
    # Create complete cascade
    # -----------------------------------------------------

    cascade = EnhancedHierarchicalCascade(
        confidence_threshold=CONFIDENCE_THRESHOLD
    )

    print()
    print("CASCADE CREATED")
    print("-" * 80)

    print(
        "Stage 1:",
        type(cascade.stage1_model).__name__
    )

    print(
        "Stage 2:",
        type(cascade.stage2_model).__name__
    )

    print(
        "Confidence threshold:",
        cascade.confidence_threshold
    )

    # -----------------------------------------------------
    # Save complete cascade
    # -----------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        cascade,
        OUTPUT_FILE,
    )

    print()
    print("MODEL SAVED")
    print("-" * 80)

    print(
        "Output:",
        OUTPUT_FILE
    )

    # -----------------------------------------------------
    # Immediate reload verification
    # -----------------------------------------------------

    loaded = joblib.load(
        OUTPUT_FILE
    )

    print()
    print("RELOAD CHECK")
    print("-" * 80)

    print(
        "Loaded type:",
        type(loaded).__name__
    )

    print(
        "Loaded Stage 1:",
        type(
            loaded.stage1_model
        ).__name__
    )

    print(
        "Loaded Stage 2:",
        type(
            loaded.stage2_model
        ).__name__
    )

    print(
        "Loaded threshold:",
        loaded.confidence_threshold
    )

    if (
        loaded.confidence_threshold
        != CONFIDENCE_THRESHOLD
    ):

        raise ValueError(
            "Confidence threshold mismatch."
        )

    # -----------------------------------------------------
    # Completion
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 360 COMPLETED")
    print("=" * 80)

    print(
        "Final enhanced cascade saved successfully."
    )

    print(
        "Artifact:",
        OUTPUT_FILE
    )

    print()
    print("ARCHITECTURE:")
    print(
        "Calibrated Stage 1 XGBoost"
    )
    print(
        "        ↓"
    )
    print(
        "Confidence Router (0.95)"
    )
    print(
        "        ↓"
    )
    print(
        "Calibrated RF + Calibrated XGBoost + Calibrated SVM"
    )
    print(
        "        ↓"
    )
    print(
        "Logistic Regression Meta-Learner"
    )
    print(
        "        ↓"
    )
    print(
        "4-Class Diagnosis"
    )

    print()
    print("IMPORTANT:")
    print("- No model was retrained.")
    print("- No test data was used.")
    print("- No threshold was changed.")
    print("- Existing baseline model was not modified.")
    print("- Flask code was not modified.")


if __name__ == "__main__":
    main()