from pathlib import Path

import joblib
import pandas as pd

from enhanced_preprocessing import (
    ALL_MODEL_FEATURES,
    TARGET_COLUMN,
    create_preprocessor,
)


# ---------------------------------------------------------
# TCMAPY2573
# Fit preprocessing using TRAINING DATA ONLY
# Step 338
# ---------------------------------------------------------

TRAIN_FILE = Path("data/enhanced_train_raw.csv")
TEST_FILE = Path("data/enhanced_test_raw.csv")

OUTPUT_DIR = Path("data")
MODEL_DIR = Path("model")

PREPROCESSOR_FILE = MODEL_DIR / "enhanced_preprocessor.pkl"

X_TRAIN_FILE = OUTPUT_DIR / "enhanced_X_train_processed.csv"
X_TEST_FILE = OUTPUT_DIR / "enhanced_X_test_processed.csv"

Y_TRAIN_FILE = OUTPUT_DIR / "enhanced_y_train.csv"
Y_TEST_FILE = OUTPUT_DIR / "enhanced_y_test.csv"

FEATURE_NAMES_FILE = OUTPUT_DIR / "enhanced_processed_feature_names.csv"


def main():

    print("=" * 80)
    print("TCMAPY2573 - TRAINING-ONLY PREPROCESSING FIT")
    print("=" * 80)

    # -----------------------------------------------------
    # Check input files
    # -----------------------------------------------------

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training file not found: {TRAIN_FILE}"
        )

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Testing file not found: {TEST_FILE}"
        )

    # -----------------------------------------------------
    # Load raw split data
    # -----------------------------------------------------

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    print(f"Training rows: {len(train_df)}")
    print(f"Testing rows : {len(test_df)}")

    # -----------------------------------------------------
    # Separate X and y
    # -----------------------------------------------------

    X_train = train_df[ALL_MODEL_FEATURES].copy()
    X_test = test_df[ALL_MODEL_FEATURES].copy()

    y_train = train_df[TARGET_COLUMN].copy()
    y_test = test_df[TARGET_COLUMN].copy()

    # -----------------------------------------------------
    # Create preprocessor
    # -----------------------------------------------------

    preprocessor = create_preprocessor()

    # -----------------------------------------------------
    # CRITICAL LEAKAGE RULE
    #
    # FIT ONLY ON TRAINING DATA
    # -----------------------------------------------------

    print()
    print("Fitting preprocessor on TRAINING data only...")

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    print("Training preprocessing fit complete.")

    # -----------------------------------------------------
    # TEST DATA IS ONLY TRANSFORMED
    # -----------------------------------------------------

    print()
    print("Transforming TEST data using the fitted training preprocessor...")

    X_test_processed = preprocessor.transform(
        X_test
    )

    print("Test transformation complete.")

    # -----------------------------------------------------
    # Feature names
    # -----------------------------------------------------

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    print()
    print(f"Processed training shape: {X_train_processed.shape}")
    print(f"Processed testing shape : {X_test_processed.shape}")
    print(f"Processed feature count : {len(feature_names)}")

    # -----------------------------------------------------
    # Save preprocessor
    # -----------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        preprocessor,
        PREPROCESSOR_FILE
    )

    # -----------------------------------------------------
    # Save processed matrices
    # -----------------------------------------------------

    X_train_processed_df = pd.DataFrame(
        X_train_processed,
        columns=feature_names
    )

    X_test_processed_df = pd.DataFrame(
        X_test_processed,
        columns=feature_names
    )

    X_train_processed_df.to_csv(
        X_TRAIN_FILE,
        index=False
    )

    X_test_processed_df.to_csv(
        X_TEST_FILE,
        index=False
    )

    y_train.to_csv(
        Y_TRAIN_FILE,
        index=False
    )

    y_test.to_csv(
        Y_TEST_FILE,
        index=False
    )

    pd.DataFrame(
        {
            "feature_name": feature_names
        }
    ).to_csv(
        FEATURE_NAMES_FILE,
        index=False
    )

    # -----------------------------------------------------
    # Verification
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 80)

    print(f"Preprocessor : {PREPROCESSOR_FILE}")
    print(f"X train      : {X_TRAIN_FILE}")
    print(f"X test       : {X_TEST_FILE}")
    print(f"y train      : {Y_TRAIN_FILE}")
    print(f"y test       : {Y_TEST_FILE}")
    print(f"Feature list : {FEATURE_NAMES_FILE}")

    print()
    print("LEAKAGE CHECK")
    print("-" * 80)
    print("Preprocessor FIT   : training data only")
    print("Preprocessor TEST  : transform only")
    print("SMOTE              : NOT applied")
    print("Feature selection  : NOT applied")
    print("Model training     : NOT performed")
    print("Patient ID         : NOT included")
    print()
    print("READY FOR NEXT STAGE")


if __name__ == "__main__":
    main()