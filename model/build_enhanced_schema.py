from pathlib import Path
import re
import pandas as pd


# ---------------------------------------------------------
# TCMAPY2573 — Enhanced 9,172-Record Dataset Schema Builder
# Step 331
# ---------------------------------------------------------

INPUT_FILE = Path("data/thyroid+disease/thyroid0387.data")
OUTPUT_FILE = Path("data/thyroid_9172_enhanced_schema.csv")


# The 29 attributes documented for thyroid0387.data
FEATURE_COLUMNS = [
    "age",
    "sex",
    "on_thyroxine",
    "query_on_thyroxine",
    "on_antithyroid_medication",
    "sick",
    "pregnant",
    "thyroid_surgery",
    "I131_treatment",
    "query_hypothyroid",
    "query_hyperthyroid",
    "lithium",
    "goitre",
    "tumor",
    "hypopituitary",
    "psych",
    "TSH_measured",
    "TSH",
    "T3_measured",
    "T3",
    "TT4_measured",
    "TT4",
    "T4U_measured",
    "T4U",
    "FTI_measured",
    "FTI",
    "TBG_measured",
    "TBG",
    "referral_source",
]


def split_diagnosis_and_id(value: str) -> tuple[str, str]:
    """
    Convert values such as:

        C|I[840904009]

    into:

        diagnosis = CI
        patient_id = 840904009
    """

    value = str(value).strip()

    # The "|" character separates diagnosis codes.
    value = value.replace("|", "")

    match = re.fullmatch(r"([A-Z-]+)\[(\d+)\]", value)

    if not match:
        raise ValueError(
            f"Unable to parse diagnosis/patient ID value: {value!r}"
        )

    diagnosis = match.group(1)
    patient_id = match.group(2)

    return diagnosis, patient_id


def map_target(diagnosis: str) -> str:
    """
    Engineering target grouping for the enhanced Stage-2 study.

    -     -> Normal
    A-D   -> Hyperthyroid
    E-H   -> Hypothyroid
    Other -> Other / Related Diagnosis

    This derived target is NOT an original UCI target label.
    It is created for our proposed research pipeline.
    """

    if diagnosis == "-":
        return "Normal"

    if any(code in diagnosis for code in "ABCD"):
        return "Hyperthyroid"

    if any(code in diagnosis for code in "EFGH"):
        return "Hypothyroid"

    return "Other / Related Diagnosis"


def main() -> None:

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    print("=" * 70)
    print("TCMAPY2573 — ENHANCED DATASET SCHEMA BUILDER")
    print("=" * 70)

    # Load raw data.
    df = pd.read_csv(
        INPUT_FILE,
        header=None,
        na_values="?",
        dtype=object,
    )

    print(f"Raw rows    : {len(df)}")
    print(f"Raw columns : {len(df.columns)}")

    expected_raw_columns = 30

    if len(df.columns) != expected_raw_columns:
        raise ValueError(
            f"Expected {expected_raw_columns} raw columns, "
            f"but found {len(df.columns)}."
        )

    # Assign names to the 29 documented attributes.
    feature_df = df.iloc[:, :29].copy()
    feature_df.columns = FEATURE_COLUMNS

    # Separate diagnosis and patient ID.
    diagnosis_values = []
    patient_ids = []

    for value in df.iloc[:, 29]:

        diagnosis, patient_id = split_diagnosis_and_id(value)

        diagnosis_values.append(diagnosis)
        patient_ids.append(patient_id)

    feature_df["diagnosis_code"] = diagnosis_values
    feature_df["patient_id"] = patient_ids

    # Create the derived research target.
    feature_df["target_group"] = feature_df["diagnosis_code"].map(
        map_target
    )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    print()
    print("SCHEMA VALIDATION")
    print("-" * 70)

    print(f"Feature columns       : {len(FEATURE_COLUMNS)}")
    print(f"Diagnosis column      : diagnosis_code")
    print(f"Patient ID column     : patient_id")
    print(f"Derived target        : target_group")
    print(f"Final output columns  : {len(feature_df.columns)}")

    # Verify patient IDs.
    duplicate_ids = feature_df["patient_id"].duplicated().sum()

    print(f"Unique patient IDs    : {feature_df['patient_id'].nunique()}")
    print(f"Duplicate patient IDs : {duplicate_ids}")

    if duplicate_ids != 0:
        raise ValueError(
            "Duplicate patient IDs detected."
        )

    # Verify diagnosis values.
    invalid_diagnosis = (
        ~feature_df["diagnosis_code"].str.fullmatch(r"[A-Z-]+")
    ).sum()

    print(f"Invalid diagnosis codes: {invalid_diagnosis}")

    if invalid_diagnosis != 0:
        raise ValueError(
            "Invalid diagnosis codes detected."
        )

    # Verify target creation.
    missing_targets = feature_df["target_group"].isna().sum()

    print(f"Missing target groups : {missing_targets}")

    if missing_targets != 0:
        raise ValueError(
            "Some records could not be assigned a target group."
        )

    # ---------------------------------------------------------
    # Target distribution
    # ---------------------------------------------------------

    print()
    print("TARGET DISTRIBUTION")
    print("-" * 70)

    print(
        feature_df["target_group"]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Total assigned:",
        feature_df["target_group"].notna().sum()
    )

    # ---------------------------------------------------------
    # Missing-value profile
    # ---------------------------------------------------------

    print()
    print("MISSING VALUES IN ORIGINAL FEATURES")
    print("-" * 70)

    missing = feature_df[FEATURE_COLUMNS].isna().sum()

    print(
        missing[missing > 0].to_string()
        if (missing > 0).any()
        else "No missing values."
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 70)
    print("SCHEMA FILE CREATED SUCCESSFULLY")
    print("=" * 70)
    print(f"Output: {OUTPUT_FILE}")
    print()
    print("IMPORTANT:")
    print("- Original dataset was not modified.")
    print("- No imputation was performed.")
    print("- No encoding was performed.")
    print("- No SMOTE was performed.")
    print("- No model was trained.")
    print("- Patient ID is retained only for auditing/leakage checks.")
    print("- Patient ID must NOT be used as a model feature.")


if __name__ == "__main__":
    main()