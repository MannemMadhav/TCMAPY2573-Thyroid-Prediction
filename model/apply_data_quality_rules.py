from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# TCMAPY2573 — Enhanced Data Quality Rules
# Step 334
# ---------------------------------------------------------

INPUT_FILE = Path("data/thyroid_9172_enhanced_schema.csv")
OUTPUT_FILE = Path("data/thyroid_9172_quality_checked.csv")

AGE_MIN = 1
AGE_MAX = 120


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print("=" * 80)
    print("TCMAPY2573 - DATA QUALITY CHECK")
    print("=" * 80)

    print(f"Input rows: {len(df)}")

    # Make sure age is numeric.
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    # Identify invalid ages.
    invalid_age_mask = (
        (df["age"] < AGE_MIN)
        | (df["age"] > AGE_MAX)
    )

    invalid_count = int(invalid_age_mask.sum())

    print(f"Invalid age values found: {invalid_count}")

    if invalid_count > 0:
        print("\nINVALID AGE RECORDS")
        print("-" * 80)

        print(
            df.loc[
                invalid_age_mask,
                ["age", "sex", "diagnosis_code", "target_group", "patient_id"]
            ].to_string(index=False)
        )

    # Preserve the rows but convert invalid ages to missing.
    df.loc[invalid_age_mask, "age"] = pd.NA

    # Save the quality-checked copy.
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 80)
    print("DATA QUALITY RULES APPLIED")
    print("=" * 80)

    print(f"Age minimum allowed : {AGE_MIN}")
    print(f"Age maximum allowed : {AGE_MAX}")
    print(f"Invalid ages changed to NaN: {invalid_count}")
    print(f"Rows retained: {len(df)}")
    print(f"Output file: {OUTPUT_FILE}")

    print("\nIMPORTANT")
    print("- Original schema file was not modified.")
    print("- No rows were deleted.")
    print("- No imputation was performed.")
    print("- No encoding was performed.")
    print("- No SMOTE was performed.")
    print("- No feature selection was performed.")
    print("- No model was trained.")
    print("- Invalid ages will be imputed later inside training folds only.")


if __name__ == "__main__":
    main()