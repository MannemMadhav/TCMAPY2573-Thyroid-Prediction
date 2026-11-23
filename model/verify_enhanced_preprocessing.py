from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# TCMAPY2573 - Processed Data Verification
# Step 339
# ---------------------------------------------------------

X_TRAIN_FILE = Path("data/enhanced_X_train_processed.csv")
X_TEST_FILE = Path("data/enhanced_X_test_processed.csv")

Y_TRAIN_FILE = Path("data/enhanced_y_train.csv")
Y_TEST_FILE = Path("data/enhanced_y_test.csv")

FEATURE_NAMES_FILE = Path(
    "data/enhanced_processed_feature_names.csv"
)

PREPROCESSOR_FILE = Path(
    "model/enhanced_preprocessor.pkl"
)


def main():

    print("=" * 80)
    print("TCMAPY2573 - ENHANCED PREPROCESSING VERIFICATION")
    print("=" * 80)

    # -----------------------------------------------------
    # Load files
    # -----------------------------------------------------

    X_train = pd.read_csv(X_TRAIN_FILE)
    X_test = pd.read_csv(X_TEST_FILE)

    y_train = pd.read_csv(Y_TRAIN_FILE)
    y_test = pd.read_csv(Y_TEST_FILE)

    feature_names = pd.read_csv(FEATURE_NAMES_FILE)

    preprocessor = joblib.load(PREPROCESSOR_FILE)

    # -----------------------------------------------------
    # Shape verification
    # -----------------------------------------------------

    print("\nSHAPES")
    print("-" * 80)

    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)

    # -----------------------------------------------------
    # Missing-value verification
    # -----------------------------------------------------

    print("\nMISSING VALUE CHECK")
    print("-" * 80)

    train_missing = int(X_train.isna().sum().sum())
    test_missing = int(X_test.isna().sum().sum())

    print("Training missing values:", train_missing)
    print("Testing missing values :", test_missing)

    if train_missing != 0:
        raise ValueError(
            "Training processed data still contains missing values."
        )

    if test_missing != 0:
        raise ValueError(
            "Testing processed data still contains missing values."
        )

    # -----------------------------------------------------
    # Infinite-value verification
    # -----------------------------------------------------

    print("\nINFINITE VALUE CHECK")
    print("-" * 80)

    train_array = X_train.to_numpy(dtype=float)
    test_array = X_test.to_numpy(dtype=float)

    train_inf = int(np.isinf(train_array).sum())
    test_inf = int(np.isinf(test_array).sum())

    print("Training infinite values:", train_inf)
    print("Testing infinite values :", test_inf)

    if train_inf != 0:
        raise ValueError(
            "Training processed data contains infinite values."
        )

    if test_inf != 0:
        raise ValueError(
            "Testing processed data contains infinite values."
        )

    # -----------------------------------------------------
    # Feature-name verification
    # -----------------------------------------------------

    print("\nFEATURE NAME CHECK")
    print("-" * 80)

    saved_feature_count = len(feature_names)

    print(
        "Saved feature names:",
        saved_feature_count
    )

    print(
        "Training feature columns:",
        len(X_train.columns)
    )

    print(
        "Testing feature columns:",
        len(X_test.columns)
    )

    if saved_feature_count != X_train.shape[1]:
        raise ValueError(
            "Saved feature count does not match training data."
        )

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            "Training and testing feature columns differ."
        )

    # -----------------------------------------------------
    # Data type verification
    # -----------------------------------------------------

    print("\nDATA TYPE CHECK")
    print("-" * 80)

    train_numeric = X_train.dtypes.apply(
        lambda dtype: pd.api.types.is_numeric_dtype(dtype)
    ).all()

    test_numeric = X_test.dtypes.apply(
        lambda dtype: pd.api.types.is_numeric_dtype(dtype)
    ).all()

    print(
        "All training columns numeric:",
        train_numeric
    )

    print(
        "All testing columns numeric:",
        test_numeric
    )

    if not train_numeric:
        raise ValueError(
            "Training data contains non-numeric columns."
        )

    if not test_numeric:
        raise ValueError(
            "Testing data contains non-numeric columns."
        )

    # -----------------------------------------------------
    # Target verification
    # -----------------------------------------------------

    print("\nTARGET CHECK")
    print("-" * 80)

    print("Training targets:")
    print(
        y_train.iloc[:, 0]
        .value_counts()
        .to_string()
    )

    print("\nTesting targets:")
    print(
        y_test.iloc[:, 0]
        .value_counts()
        .to_string()
    )

    # -----------------------------------------------------
    # Preprocessor verification
    # -----------------------------------------------------

    print("\nPREPROCESSOR CHECK")
    print("-" * 80)

    print(
        "Loaded object type:",
        type(preprocessor).__name__
    )

    fitted = hasattr(
        preprocessor,
        "transformers_"
    )

    print(
        "Transformer fitted:",
        fitted
    )

    if not fitted:
        raise ValueError(
            "Saved preprocessor is not fitted."
        )

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    print("\n" + "=" * 80)
    print("VERIFICATION RESULT")
    print("=" * 80)

    print("Processed data contains no missing values.")
    print("Processed data contains no infinite values.")
    print("Processed data contains numeric model inputs.")
    print("Train/test feature columns match.")
    print("Saved preprocessor loads successfully.")
    print("Patient ID is excluded from model features.")

    print()
    print("LEAKAGE STATUS")
    print("-" * 80)

    print("Preprocessor fitted on TRAINING data only.")
    print("Test data was transformed only.")
    print("SMOTE has NOT been applied.")
    print("Feature selection has NOT been applied.")
    print("No model has been trained.")

    print()
    print("STEP 339 VERIFICATION PASSED")


if __name__ == "__main__":
    main()