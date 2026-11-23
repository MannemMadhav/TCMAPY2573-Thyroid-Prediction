from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# TCMAPY2573 - Enhanced Feature Audit
# Step 332
# ---------------------------------------------------------

INPUT_FILE = Path("data/thyroid_9172_enhanced_schema.csv")

FEATURES = [
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

CATEGORICAL_FEATURES = [
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
    "T3_measured",
    "TT4_measured",
    "T4U_measured",
    "FTI_measured",
    "TBG_measured",
    "referral_source",
]

MEASUREMENT_PAIRS = [
    ("TSH_measured", "TSH"),
    ("T3_measured", "T3"),
    ("TT4_measured", "TT4"),
    ("T4U_measured", "T4U"),
    ("FTI_measured", "FTI"),
    ("TBG_measured", "TBG"),
]


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print("=" * 80)
    print("TCMAPY2573 - ENHANCED FEATURE AUDIT")
    print("=" * 80)

    print(f"Rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Model input features: {len(FEATURES)}")

    print("\n" + "=" * 80)
    print("FEATURE INFORMATION")
    print("=" * 80)

    header = (
        f"{'FEATURE':28} "
        f"{'TYPE':12} "
        f"{'MISSING':8} "
        f"{'UNIQUE':8}"
    )

    print(header)
    print("-" * 80)

    for feature in FEATURES:
        dtype = str(df[feature].dtype)
        missing = int(df[feature].isna().sum())
        unique = int(df[feature].nunique(dropna=True))

        print(
            f"{feature:28} "
            f"{dtype:12} "
            f"{missing:<8} "
            f"{unique:<8}"
        )

    print("\n" + "=" * 80)
    print("CATEGORICAL FEATURE VALUES")
    print("=" * 80)

    for feature in CATEGORICAL_FEATURES:
        print(f"\n{feature}:")
        print(
            df[feature]
            .value_counts(dropna=False)
            .head(15)
            .to_string()
        )

    print("\n" + "=" * 80)
    print("NUMERICAL FEATURE RANGES")
    print("=" * 80)

    numerical_features = [
        "age",
        "TSH",
        "T3",
        "TT4",
        "T4U",
        "FTI",
        "TBG",
    ]

    for feature in numerical_features:
        series = pd.to_numeric(df[feature], errors="coerce")

        print(
            f"{feature:8} "
            f"min={series.min()} "
            f"max={series.max()} "
            f"median={series.median()}"
        )

    print("\n" + "=" * 80)
    print("MEASUREMENT-FLAG CONSISTENCY")
    print("=" * 80)

    for flag, value in MEASUREMENT_PAIRS:

        flag_values = (
            df[flag]
            .astype(str)
            .str.lower()
            .value_counts(dropna=False)
        )

        value_present = int(df[value].notna().sum())
        value_missing = int(df[value].isna().sum())

        print(f"\n{flag} -> {value}")
        print("Flag distribution:")
        print(flag_values.to_string())
        print(f"Value present : {value_present}")
        print(f"Value missing : {value_missing}")

    print("\n" + "=" * 80)
    print("TARGET DISTRIBUTION")
    print("=" * 80)

    print(
        df["target_group"]
        .value_counts()
        .to_string()
    )

    print("\n" + "=" * 80)
    print("PATIENT ID CHECK")
    print("=" * 80)

    print(
        f"Total IDs: {df['patient_id'].nunique()}"
    )

    print(
        f"Duplicate IDs: "
        f"{df['patient_id'].duplicated().sum()}"
    )

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()