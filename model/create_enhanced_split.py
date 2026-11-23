from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# TCMAPY2573 — Enhanced Stratified Train/Test Split
# Step 337
# ---------------------------------------------------------

INPUT_FILE = Path("data/thyroid_9172_quality_checked.csv")

TRAIN_FILE = Path("data/enhanced_train_raw.csv")
TEST_FILE = Path("data/enhanced_test_raw.csv")

TARGET_COLUMN = "target_group"
ID_COLUMN = "patient_id"

TEST_SIZE = 0.20
RANDOM_STATE = 42


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        TARGET_COLUMN,
        ID_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    print("=" * 80)
    print("TCMAPY2573 - ENHANCED STRATIFIED TRAIN/TEST SPLIT")
    print("=" * 80)

    print(f"Total records: {len(df)}")

    # -----------------------------------------------------
    # Verify patient IDs before splitting
    # -----------------------------------------------------

    duplicate_ids = int(
        df[ID_COLUMN].duplicated().sum()
    )

    print(f"Duplicate patient IDs before split: {duplicate_ids}")

    if duplicate_ids != 0:
        raise ValueError(
            "Duplicate patient IDs detected before splitting."
        )

    # -----------------------------------------------------
    # Stratified split
    # -----------------------------------------------------

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN],
    )

    # Reset row indexes for clean output.
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # -----------------------------------------------------
    # Patient overlap check
    # -----------------------------------------------------

    train_ids = set(train_df[ID_COLUMN])
    test_ids = set(test_df[ID_COLUMN])

    overlap = train_ids.intersection(test_ids)

    print(f"Training records: {len(train_df)}")
    print(f"Testing records : {len(test_df)}")
    print(f"Patient ID overlap: {len(overlap)}")

    if overlap:
        raise ValueError(
            "Patient ID leakage detected between train and test sets."
        )

    # -----------------------------------------------------
    # Class distribution
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("TRAINING CLASS DISTRIBUTION")
    print("=" * 80)

    print(
        train_df[TARGET_COLUMN]
        .value_counts()
        .to_string()
    )

    print()
    print("=" * 80)
    print("TEST CLASS DISTRIBUTION")
    print("=" * 80)

    print(
        test_df[TARGET_COLUMN]
        .value_counts()
        .to_string()
    )

    # -----------------------------------------------------
    # Class proportions
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("CLASS PROPORTIONS")
    print("=" * 80)

    overall = (
        df[TARGET_COLUMN]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    train_prop = (
        train_df[TARGET_COLUMN]
        .value_counts(normalize=True)
        .reindex(overall.index)
        * 100
    )

    test_prop = (
        test_df[TARGET_COLUMN]
        .value_counts(normalize=True)
        .reindex(overall.index)
        * 100
    )

    distribution = pd.DataFrame(
        {
            "Overall (%)": overall.round(2),
            "Train (%)": train_prop.round(2),
            "Test (%)": test_prop.round(2),
        }
    )

    print(distribution.to_string())

    # -----------------------------------------------------
    # Save raw split copies
    # -----------------------------------------------------

    TRAIN_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    train_df.to_csv(
        TRAIN_FILE,
        index=False
    )

    test_df.to_csv(
        TEST_FILE,
        index=False
    )

    # -----------------------------------------------------
    # Final verification
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("SPLIT CREATED SUCCESSFULLY")
    print("=" * 80)

    print(f"Training file: {TRAIN_FILE}")
    print(f"Testing file : {TEST_FILE}")

    print()
    print("IMPORTANT:")
    print("- Split was stratified by target_group.")
    print("- Random state = 42.")
    print("- Test data has not been imputed.")
    print("- Test data has not been encoded.")
    print("- Test data has not been balanced.")
    print("- No SMOTE was applied.")
    print("- No model was trained.")
    print("- Patient IDs do not overlap.")
    print("- Patient ID remains audit-only and must not be a model feature.")


if __name__ == "__main__":
    main()