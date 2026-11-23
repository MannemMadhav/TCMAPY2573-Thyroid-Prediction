from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# TCMAPY2573
# Detailed Diagnostic Family Analysis
# Step 341A
# ---------------------------------------------------------

INPUT_FILE = Path(
    "data/thyroid_9172_quality_checked.csv"
)


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    diagnosis = (
        df["diagnosis_code"]
        .astype(str)
        .str.strip()
    )

    # -----------------------------------------------------
    # Diagnostic groups defined by thyroid0387.names
    # -----------------------------------------------------

    groups = {
        "Hyperthyroid": set("ABCD"),
        "Hypothyroid": set("EFGH"),
        "Binding Protein": set("IJ"),
        "General Health": set("K"),
        "Replacement Therapy": set("LMN"),
        "Antithyroid Treatment": set("OPQ"),
        "Miscellaneous": set("RST"),
    }

    print("=" * 80)
    print("TCMAPY2573 - DETAILED DIAGNOSTIC FAMILY DISTRIBUTION")
    print("=" * 80)

    print(f"Total records: {len(df)}")

    # -----------------------------------------------------
    # Count each family
    # -----------------------------------------------------

    print()
    print("DIAGNOSTIC FAMILY COUNTS")
    print("-" * 80)

    normal_count = int(
        (diagnosis == "-").sum()
    )

    print(
        f"{'Normal / No Diagnosis':30} {normal_count}"
    )

    family_counts = {}

    for family, codes in groups.items():

        mask = diagnosis.apply(
            lambda value: any(
                code in codes
                for code in value
            )
        )

        count = int(mask.sum())

        family_counts[family] = count

        print(
            f"{family:30} {count}"
        )

    # -----------------------------------------------------
    # Percentages
    # -----------------------------------------------------

    print()
    print("DIAGNOSTIC FAMILY PERCENTAGES")
    print("-" * 80)

    print(
        f"{'Normal / No Diagnosis':30} "
        f"{normal_count / len(df) * 100:.2f}%"
    )

    for family, count in family_counts.items():

        print(
            f"{family:30} "
            f"{count / len(df) * 100:.2f}%"
        )

    # -----------------------------------------------------
    # Exact diagnosis patterns
    # -----------------------------------------------------

    print()
    print("EXACT DIAGNOSIS PATTERNS")
    print("-" * 80)

    print(
        diagnosis
        .value_counts()
        .to_string()
    )

    # -----------------------------------------------------
    # Multiple-family records
    # -----------------------------------------------------

    print()
    print("RECORDS BELONGING TO MULTIPLE DIAGNOSTIC FAMILIES")
    print("-" * 80)

    multiple_family_records = 0

    for value in diagnosis.unique():

        if value == "-":
            continue

        matched_families = []

        for family, codes in groups.items():

            if any(
                code in codes
                for code in value
            ):
                matched_families.append(family)

        if len(matched_families) > 1:

            count = int(
                (diagnosis == value).sum()
            )

            multiple_family_records += count

            print(
                f"{value:8} -> "
                f"{', '.join(matched_families):45} "
                f"count={count}"
            )

    print()
    print(
        "TOTAL RECORDS WITH MULTIPLE FAMILIES:",
        multiple_family_records
    )

    # -----------------------------------------------------
    # Primary disease distribution
    # -----------------------------------------------------

    print()
    print("PRIMARY THYROID DISEASE COUNTS")
    print("-" * 80)

    hyper_mask = diagnosis.apply(
        lambda value: any(
            code in set("ABCD")
            for code in value
        )
    )

    hypo_mask = diagnosis.apply(
        lambda value: any(
            code in set("EFGH")
            for code in value
        )
    )

    print(
        "Hyperthyroid present:",
        int(hyper_mask.sum())
    )

    print(
        "Hypothyroid present:",
        int(hypo_mask.sum())
    )

    print(
        "Both Hyperthyroid + Hypothyroid:",
        int((hyper_mask & hypo_mask).sum())
    )

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()