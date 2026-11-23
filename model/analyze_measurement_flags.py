from pathlib import Path
import pandas as pd


INPUT_FILE = Path("data/thyroid_9172_quality_checked.csv")

FLAGS = [
    "TSH_measured",
    "T3_measured",
    "TT4_measured",
    "T4U_measured",
    "FTI_measured",
    "TBG_measured",
]


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"File not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print("=" * 80)
    print("TCMAPY2573 - MEASUREMENT FLAGS BY TARGET GROUP")
    print("=" * 80)

    target_groups = [
        "Normal",
        "Hyperthyroid",
        "Hypothyroid",
        "Other / Related Diagnosis",
    ]

    for target in target_groups:
        subset = df[df["target_group"] == target]

        print()
        print(target)
        print("-" * 80)
        print(f"Records: {len(subset)}")

        for flag in FLAGS:
            missing_count = (
                subset[flag]
                .astype(str)
                .str.lower()
                .eq("f")
                .sum()
            )

            percentage = (
                missing_count / len(subset) * 100
                if len(subset) > 0
                else 0
            )

            print(
                f"{flag:15} "
                f"not measured = {missing_count:4} "
                f"({percentage:6.2f}%)"
            )

    print()
    print("=" * 80)
    print("OVERALL TARGET COUNTS")
    print("=" * 80)

    print(
        df["target_group"]
        .value_counts()
        .to_string()
    )

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()