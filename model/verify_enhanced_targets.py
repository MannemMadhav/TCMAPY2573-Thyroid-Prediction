from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# TCMAPY2573
# Enhanced Target / Split Alignment Verification
# Step 343
# ---------------------------------------------------------

TARGET_FILE = Path(
    "data/thyroid_9172_targets.csv"
)

TRAIN_FILE = Path(
    "data/enhanced_train_raw.csv"
)

TEST_FILE = Path(
    "data/enhanced_test_raw.csv"
)


def main():

    print("=" * 80)
    print("TCMAPY2573 - TARGET / SPLIT ALIGNMENT VERIFICATION")
    print("=" * 80)

    # -----------------------------------------------------
    # Load files
    # -----------------------------------------------------

    targets = pd.read_csv(TARGET_FILE)
    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    print(f"Target records: {len(targets)}")
    print(f"Training records: {len(train)}")
    print(f"Testing records : {len(test)}")

    # -----------------------------------------------------
    # Check total rows
    # -----------------------------------------------------

    if len(targets) != len(train) + len(test):
        raise ValueError(
            "Target records do not equal train + test records."
        )

    # -----------------------------------------------------
    # Patient ID checks
    # -----------------------------------------------------

    target_ids = set(
        targets["patient_id"]
    )

    train_ids = set(
        train["patient_id"]
    )

    test_ids = set(
        test["patient_id"]
    )

    print()
    print("PATIENT ID CHECK")
    print("-" * 80)

    print(
        "Target unique IDs:",
        len(target_ids)
    )

    print(
        "Train unique IDs:",
        len(train_ids)
    )

    print(
        "Test unique IDs:",
        len(test_ids)
    )

    train_test_overlap = (
        train_ids.intersection(test_ids)
    )

    target_train_missing = (
        train_ids.difference(target_ids)
    )

    target_test_missing = (
        test_ids.difference(target_ids)
    )

    print(
        "Train/Test ID overlap:",
        len(train_test_overlap)
    )

    print(
        "Train IDs missing from target file:",
        len(target_train_missing)
    )

    print(
        "Test IDs missing from target file:",
        len(target_test_missing)
    )

    if train_test_overlap:
        raise ValueError(
            "Patient leakage detected between train and test."
        )

    if target_train_missing or target_test_missing:
        raise ValueError(
            "Some split patient IDs are missing from target dataset."
        )

    # -----------------------------------------------------
    # Merge targets onto the existing raw splits
    # -----------------------------------------------------

    target_columns = [
        "patient_id",
        "stage1_target",
        "stage1_label",
        "stage2_target",
        "stage2_label",
    ]

    target_lookup = targets[target_columns].copy()

    train_verified = train.merge(
        target_lookup,
        on="patient_id",
        how="left",
        validate="one_to_one",
    )

    test_verified = test.merge(
        target_lookup,
        on="patient_id",
        how="left",
        validate="one_to_one",
    )

    # -----------------------------------------------------
    # Missing target check after merge
    # -----------------------------------------------------

    print()
    print("TARGET ALIGNMENT CHECK")
    print("-" * 80)

    for name, data in [
        ("Training", train_verified),
        ("Testing", test_verified),
    ]:
        s1_missing = int(
            data["stage1_target"].isna().sum()
        )

        s2_missing = int(
            data["stage2_target"].isna().sum()
        )

        print(
            f"{name} Stage 1 missing: {s1_missing}"
        )

        print(
            f"{name} Stage 2 missing: {s2_missing}"
        )

        if s1_missing != 0 or s2_missing != 0:
            raise ValueError(
                f"Missing targets found in {name.lower()} set."
            )

    # -----------------------------------------------------
    # Stage 1 distribution
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("STAGE 1 TRAINING DISTRIBUTION")
    print("=" * 80)

    print(
        train_verified["stage1_label"]
        .value_counts()
        .to_string()
    )

    print()
    print("=" * 80)
    print("STAGE 1 TEST DISTRIBUTION")
    print("=" * 80)

    print(
        test_verified["stage1_label"]
        .value_counts()
        .to_string()
    )

    # -----------------------------------------------------
    # Stage 2 distribution
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("STAGE 2 TRAINING DISTRIBUTION")
    print("=" * 80)

    print(
        train_verified["stage2_label"]
        .value_counts()
        .to_string()
    )

    print()
    print("=" * 80)
    print("STAGE 2 TEST DISTRIBUTION")
    print("=" * 80)

    print(
        test_verified["stage2_label"]
        .value_counts()
        .to_string()
    )

    # -----------------------------------------------------
    # Verify proportions
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("STAGE 2 CLASS PROPORTIONS")
    print("=" * 80)

    overall = (
        targets["stage2_label"]
        .value_counts(normalize=True)
        * 100
    )

    train_prop = (
        train_verified["stage2_label"]
        .value_counts(normalize=True)
        .reindex(overall.index)
        * 100
    )

    test_prop = (
        test_verified["stage2_label"]
        .value_counts(normalize=True)
        .reindex(overall.index)
        * 100
    )

    comparison = pd.DataFrame(
        {
            "Overall %": overall.round(2),
            "Train %": train_prop.round(2),
            "Test %": test_prop.round(2),
        }
    )

    print(
        comparison.to_string()
    )

    # -----------------------------------------------------
    # Save verified split targets
    # -----------------------------------------------------

    train_output = Path(
        "data/enhanced_train_with_targets.csv"
    )

    test_output = Path(
        "data/enhanced_test_with_targets.csv"
    )

    train_verified.to_csv(
        train_output,
        index=False
    )

    test_verified.to_csv(
        test_output,
        index=False
    )

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("TARGET / SPLIT VERIFICATION PASSED")
    print("=" * 80)

    print(f"Training output: {train_output}")
    print(f"Testing output : {test_output}")

    print()
    print("IMPORTANT:")
    print("- Original raw dataset was not modified.")
    print("- Patient IDs remain audit-only.")
    print("- Train/Test patient IDs do not overlap.")
    print("- Stage 1 targets align with the existing split.")
    print("- Stage 2 targets align with the existing split.")
    print("- No SMOTE was applied.")
    print("- No model was trained.")
    print("- No feature selection was performed.")


if __name__ == "__main__":
    main()