from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# TCMAPY2573
# Enhanced Stage 2 Input Audit
# Step 354
# ---------------------------------------------------------

X_FILE = Path(
    "data/enhanced_stage2_smote_X_train_processed.csv"
)

Y_FILE = Path(
    "data/enhanced_stage2_smote_y_train.csv"
)

TEST_X_FILE = Path(
    "data/enhanced_X_test_processed.csv"
)

TEST_RAW_FILE = Path(
    "data/enhanced_test_with_targets.csv"
)


EXPECTED_TRAIN_ROWS = 8610
EXPECTED_TEST_ROWS = 1835
EXPECTED_FEATURES = 56

EXPECTED_TRAIN_COUNTS = {
    0: 5416,
    1: 1000,
    2: 1000,
    3: 1194,
}

EXPECTED_TEST_COUNTS = {
    0: 1355,
    1: 48,
    2: 133,
    3: 299,
}


CLASS_NAMES = {
    0: "Normal",
    1: "Hyperthyroid",
    2: "Hypothyroid",
    3: "Other / Related Diagnosis",
}


def main():

    print("=" * 80)
    print("TCMAPY2573 - ENHANCED STAGE 2 INPUT AUDIT")
    print("=" * 80)

    # -----------------------------------------------------
    # File checks
    # -----------------------------------------------------

    required_files = [
        X_FILE,
        Y_FILE,
        TEST_X_FILE,
        TEST_RAW_FILE,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    # -----------------------------------------------------
    # Load
    # -----------------------------------------------------

    X_train = pd.read_csv(
        X_FILE
    )

    y_train = pd.read_csv(
        Y_FILE
    )["stage2_target"].astype(int)

    X_test = pd.read_csv(
        TEST_X_FILE
    )

    test_raw = pd.read_csv(
        TEST_RAW_FILE
    )

    y_test = (
        test_raw["stage2_target"]
        .astype(int)
    )

    # -----------------------------------------------------
    # Shapes
    # -----------------------------------------------------

    print()
    print("SHAPE CHECK")
    print("-" * 80)

    print(
        "Enhanced Stage 2 X_train:",
        X_train.shape
    )

    print(
        "Enhanced Stage 2 y_train:",
        y_train.shape
    )

    print(
        "Untouched X_test:",
        X_test.shape
    )

    print(
        "Untouched y_test:",
        y_test.shape
    )

    if X_train.shape != (
        EXPECTED_TRAIN_ROWS,
        EXPECTED_FEATURES,
    ):
        raise ValueError(
            "Unexpected Stage 2 training shape."
        )

    if len(y_train) != EXPECTED_TRAIN_ROWS:
        raise ValueError(
            "Unexpected Stage 2 training target length."
        )

    if X_test.shape != (
        EXPECTED_TEST_ROWS,
        EXPECTED_FEATURES,
    ):
        raise ValueError(
            "Unexpected Stage 2 test shape."
        )

    if len(y_test) != EXPECTED_TEST_ROWS:
        raise ValueError(
            "Unexpected Stage 2 test target length."
        )

    # -----------------------------------------------------
    # Training feature checks
    # -----------------------------------------------------

    print()
    print("TRAINING FEATURE CHECK")
    print("-" * 80)

    train_missing = int(
        X_train.isna().sum().sum()
    )

    train_infinite = int(
        np.isinf(
            X_train.to_numpy(
                dtype=float
            )
        ).sum()
    )

    train_numeric = (
        X_train.dtypes
        .apply(
            pd.api.types.is_numeric_dtype
        )
        .all()
    )

    print(
        "Missing values:",
        train_missing
    )

    print(
        "Infinite values:",
        train_infinite
    )

    print(
        "All numeric:",
        train_numeric
    )

    # -----------------------------------------------------
    # Test feature checks
    # -----------------------------------------------------

    print()
    print("TEST FEATURE CHECK")
    print("-" * 80)

    test_missing = int(
        X_test.isna().sum().sum()
    )

    test_infinite = int(
        np.isinf(
            X_test.to_numpy(
                dtype=float
            )
        ).sum()
    )

    test_numeric = (
        X_test.dtypes
        .apply(
            pd.api.types.is_numeric_dtype
        )
        .all()
    )

    print(
        "Missing values:",
        test_missing
    )

    print(
        "Infinite values:",
        test_infinite
    )

    print(
        "All numeric:",
        test_numeric
    )

    if train_missing != 0:
        raise ValueError(
            "Training data contains missing values."
        )

    if test_missing != 0:
        raise ValueError(
            "Test data contains missing values."
        )

    if train_infinite != 0:
        raise ValueError(
            "Training data contains infinite values."
        )

    if test_infinite != 0:
        raise ValueError(
            "Test data contains infinite values."
        )

    if not train_numeric or not test_numeric:
        raise ValueError(
            "Non-numeric model input detected."
        )

    # -----------------------------------------------------
    # Training distribution
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("STAGE 2 TRAINING DISTRIBUTION")
    print("=" * 80)

    train_counts = (
        y_train.value_counts()
        .sort_index()
    )

    for class_id, expected in EXPECTED_TRAIN_COUNTS.items():

        actual = int(
            train_counts.get(
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
                "Training class distribution mismatch."
            )

    # -----------------------------------------------------
    # Untouched test distribution
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("STAGE 2 TEST DISTRIBUTION")
    print("=" * 80)

    test_counts = (
        y_test.value_counts()
        .sort_index()
    )

    for class_id, expected in EXPECTED_TEST_COUNTS.items():

        actual = int(
            test_counts.get(
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
                "Test class distribution mismatch."
            )

    # -----------------------------------------------------
    # Feature-column identity
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("FEATURE COLUMN CHECK")
    print("=" * 80)

    train_columns = list(
        X_train.columns
    )

    test_columns = list(
        X_test.columns
    )

    print(
        "Training features:",
        len(train_columns)
    )

    print(
        "Testing features:",
        len(test_columns)
    )

    print(
        "Columns match:",
        train_columns == test_columns
    )

    if train_columns != test_columns:
        raise ValueError(
            "Training and test feature columns differ."
        )

    # -----------------------------------------------------
    # Final
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 354 VERIFICATION PASSED")
    print("=" * 80)

    print(
        "Enhanced Stage 2 training input is valid."
    )

    print(
        "Enhanced Stage 2 test input is valid."
    )

    print(
        "Training shape:",
        X_train.shape
    )

    print(
        "Test shape:",
        X_test.shape
    )

    print()
    print("IMPORTANT:")
    print("- Training data is the controlled-SMOTENC dataset.")
    print("- Test data remains untouched.")
    print("- No model has been trained in this step.")
    print("- No confidence threshold was changed.")
    print("- No Flask code was changed.")


if __name__ == "__main__":
    main()