from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# TCMAPY2573
# Enhanced Hierarchical Target Construction
# Step 342
# ---------------------------------------------------------

INPUT_FILE = Path(
    "data/thyroid_9172_quality_checked.csv"
)

OUTPUT_FILE = Path(
    "data/thyroid_9172_targets.csv"
)


def create_stage2_target(target_group):
    """
    Four-class detailed diagnostic target.

    0 = Normal
    1 = Hyperthyroid
    2 = Hypothyroid
    3 = Other / Related Diagnosis
    """

    mapping = {
        "Normal": 0,
        "Hyperthyroid": 1,
        "Hypothyroid": 2,
        "Other / Related Diagnosis": 3,
    }

    return mapping[target_group]


def create_stage1_target(target_group):
    """
    Binary screening target.

    0 = Normal
    1 = Non-Normal

    Non-Normal includes:
        Hyperthyroid
        Hypothyroid
        Other / Related Diagnosis
    """

    if target_group == "Normal":
        return 0

    return 1


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print("=" * 80)
    print("TCMAPY2573 - ENHANCED HIERARCHICAL TARGET CONSTRUCTION")
    print("=" * 80)

    print(f"Input rows: {len(df)}")

    # -----------------------------------------------------
    # Required columns
    # -----------------------------------------------------

    required_columns = [
        "patient_id",
        "diagnosis_code",
        "target_group",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    # -----------------------------------------------------
    # Create Stage 1
    # -----------------------------------------------------

    df["stage1_target"] = (
        df["target_group"]
        .map(create_stage1_target)
    )

    df["stage1_label"] = df["stage1_target"].map(
        {
            0: "Normal",
            1: "Non-Normal",
        }
    )

    # -----------------------------------------------------
    # Create Stage 2
    # -----------------------------------------------------

    df["stage2_target"] = (
        df["target_group"]
        .map(create_stage2_target)
    )

    df["stage2_label"] = df["target_group"]

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("STAGE 1 TARGET")
    print("=" * 80)

    print(
        df["stage1_label"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Stage 1 numeric labels:")
    print(
        df["stage1_target"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("=" * 80)
    print("STAGE 2 TARGET")
    print("=" * 80)

    print(
        df["stage2_label"]
        .value_counts()
        .to_string()
    )

    print()
    print("Stage 2 numeric labels:")
    print(
        df["stage2_target"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # -----------------------------------------------------
    # Missing target check
    # -----------------------------------------------------

    stage1_missing = int(
        df["stage1_target"].isna().sum()
    )

    stage2_missing = int(
        df["stage2_target"].isna().sum()
    )

    print()
    print("=" * 80)
    print("TARGET VALIDATION")
    print("=" * 80)

    print(
        "Missing Stage 1 targets:",
        stage1_missing
    )

    print(
        "Missing Stage 2 targets:",
        stage2_missing
    )

    if stage1_missing != 0:
        raise ValueError(
            "Missing Stage 1 targets detected."
        )

    if stage2_missing != 0:
        raise ValueError(
            "Missing Stage 2 targets detected."
        )

    # -----------------------------------------------------
    # Patient ID uniqueness
    # -----------------------------------------------------

    duplicate_ids = int(
        df["patient_id"].duplicated().sum()
    )

    print(
        "Duplicate patient IDs:",
        duplicate_ids
    )

    if duplicate_ids != 0:
        raise ValueError(
            "Duplicate patient IDs detected."
        )

    # -----------------------------------------------------
    # Save target dataset
    # -----------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("TARGET DATASET CREATED SUCCESSFULLY")
    print("=" * 80)

    print(
        f"Output file: {OUTPUT_FILE}"
    )

    print()
    print("STAGE 1:")
    print("0 = Normal")
    print("1 = Non-Normal")

    print()
    print("STAGE 2:")
    print("0 = Normal")
    print("1 = Hyperthyroid")
    print("2 = Hypothyroid")
    print("3 = Other / Related Diagnosis")

    print()
    print("IMPORTANT:")
    print("- Original raw dataset was not modified.")
    print("- Diagnosis codes were preserved.")
    print("- Patient IDs were preserved for auditing only.")
    print("- No imputation was performed.")
    print("- No encoding was performed.")
    print("- No SMOTE was applied.")
    print("- No feature selection was performed.")
    print("- No model was trained.")


if __name__ == "__main__":
    main()