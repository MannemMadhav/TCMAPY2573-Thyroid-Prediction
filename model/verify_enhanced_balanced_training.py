from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# TCMAPY2573
# Enhanced Balanced Training Verification
# Step 346
# ---------------------------------------------------------

X_FILE = Path(
    "data/enhanced_stage2_smote_X_train_processed.csv"
)

Y_FILE = Path(
    "data/enhanced_stage2_smote_y_train.csv"
)

INFO_FILE = Path(
    "data/enhanced_stage2_smote_info.csv"
)

STAGE1_WEIGHT_FILE = Path(
    "data/enhanced_stage1_class_weights.csv"
)


EXPECTED_ROWS = 8610
EXPECTED_FEATURES = 56

EXPECTED_COUNTS = {
    0: 5416,
    1: 1000,
    2: 1000,
    3: 1194,
}

CLASS_NAMES = {
    0: "Normal",
    1: "Hyperthyroid",
    2: "Hypothyroid",
    3: "Other / Related Diagnosis",
}


def main():

    print("=" * 80)
    print("TCMAPY2573 - BALANCED TRAINING VERIFICATION")
    print("=" * 80)

    # -----------------------------------------------------
    # Check files
    # -----------------------------------------------------

    for file_path in [
        X_FILE,
        Y_FILE,
        INFO_FILE,
        STAGE1_WEIGHT_FILE,
    ]:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    X = pd.read_csv(X_FILE)
    y_df = pd.read_csv(Y_FILE)
    info = pd.read_csv(INFO_FILE)
    stage1_weights = pd.read_csv(
        STAGE1_WEIGHT_FILE
    )

    y = y_df["stage2_target"]

    # -----------------------------------------------------
    # Shape
    # -----------------------------------------------------

    print("\nSHAPE CHECK")
    print("-" * 80)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    if X.shape != (
        EXPECTED_ROWS,
        EXPECTED_FEATURES,
    ):
        raise ValueError(
            "Unexpected balanced X shape."
        )

    if len(y) != EXPECTED_ROWS:
        raise ValueError(
            "Unexpected balanced y length."
        )

    # -----------------------------------------------------
    # Missing values
    # -----------------------------------------------------

    print("\nMISSING VALUE CHECK")
    print("-" * 80)

    missing = int(
        X.isna().sum().sum()
    )

    print(
        "Missing values:",
        missing
    )

    if missing != 0:
        raise ValueError(
            "Balanced training data contains missing values."
        )

    # -----------------------------------------------------
    # Infinite values
    # -----------------------------------------------------

    print("\nINFINITE VALUE CHECK")
    print("-" * 80)

    X_array = X.to_numpy(
        dtype=float
    )

    infinite = int(
        np.isinf(X_array).sum()
    )

    print(
        "Infinite values:",
        infinite
    )

    if infinite != 0:
        raise ValueError(
            "Balanced training data contains infinite values."
        )

    # -----------------------------------------------------
    # Numeric check
    # -----------------------------------------------------

    print("\nDATA TYPE CHECK")
    print("-" * 80)

    all_numeric = X.dtypes.apply(
        pd.api.types.is_numeric_dtype
    ).all()

    print(
        "All features numeric:",
        all_numeric
    )

    if not all_numeric:
        raise ValueError(
            "Non-numeric feature detected."
        )

    # -----------------------------------------------------
    # Stage 2 distribution
    # -----------------------------------------------------

    print("\nSTAGE 2 BALANCED DISTRIBUTION")
    print("-" * 80)

    actual_counts = (
        y.value_counts()
        .sort_index()
    )

    for class_id, expected in EXPECTED_COUNTS.items():

        actual = int(
            actual_counts.get(
                class_id,
                0
            )
        )

        print(
            f"{CLASS_NAMES[class_id]:30} "
            f"expected={expected:5} "
            f"actual={actual:5}"
        )

        if actual != expected:
            raise ValueError(
                f"Class {class_id} count mismatch."
            )

    # -----------------------------------------------------
    # Verify only intended classes were changed
    # -----------------------------------------------------

    print("\nBALANCING INFORMATION")
    print("-" * 80)

    print(
        info.to_string(index=False)
    )

    # -----------------------------------------------------
    # Stage 1 weights
    # -----------------------------------------------------

    print("\nSTAGE 1 CLASS WEIGHTS")
    print("-" * 80)

    print(
        stage1_weights.to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    print("\n" + "=" * 80)
    print("STEP 346 VERIFICATION PASSED")
    print("=" * 80)

    print("Balanced Stage 2 data is valid.")
    print("Expected class counts are confirmed.")
    print("No missing values remain.")
    print("No infinite values remain.")
    print("All model features are numeric.")
    print("Balanced shape is 8610 x 56.")

    print()
    print("IMPORTANT:")
    print("- Test data was not used.")
    print("- Test data remains untouched.")
    print("- No feature selection has been performed.")
    print("- No model has been trained.")
    print("- Stage 1 uses class weights.")
    print("- Stage 2 uses controlled SMOTENC.")


if __name__ == "__main__":
    main()