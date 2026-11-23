from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.utils.class_weight import compute_class_weight


# ---------------------------------------------------------
# TCMAPY2573
# Class Imbalance Analysis
# Step 344
# ---------------------------------------------------------

TRAIN_FILE = Path(
    "data/enhanced_train_with_targets.csv"
)


def analyze_target(
    y: pd.Series,
    stage_name: str,
    labels: list[str],
):
    print()
    print("=" * 80)
    print(stage_name)
    print("=" * 80)

    counts = (
        y.value_counts()
        .reindex(labels, fill_value=0)
    )

    percentages = (
        counts / len(y) * 100
    )

    print("\nCLASS COUNTS")
    print("-" * 80)

    for label in labels:
        print(
            f"{label:30} "
            f"{int(counts[label]):6} "
            f"({percentages[label]:6.2f}%)"
        )

    largest = counts.max()
    smallest = counts.min()

    print()
    print("Majority class count:", int(largest))
    print("Minority class count:", int(smallest))

    if smallest > 0:
        print(
            "Imbalance ratio:",
            f"{largest / smallest:.2f}:1"
        )

    # -----------------------------------------------------
    # Balanced class weights
    # -----------------------------------------------------

    label_to_number = {
        label: index
        for index, label in enumerate(labels)
    }

    y_encoded = (
        y.map(label_to_number)
        .astype(int)
        .to_numpy()
    )

    encoded_labels = np.array(
        list(range(len(labels))),
        dtype=int,
    )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=encoded_labels,
        y=y_encoded,
    )

    print()
    print("BALANCED CLASS WEIGHTS")
    print("-" * 80)

    for label, weight in zip(
        labels,
        weights,
    ):
        print(
            f"{label:30} {weight:.4f}"
        )

    # -----------------------------------------------------
    # Full balancing reference
    # -----------------------------------------------------

    target_count = int(largest)

    print()
    print("FULL BALANCING REFERENCE")
    print("-" * 80)

    total_after = 0

    for label in labels:

        current = int(counts[label])

        if current < target_count:
            added = target_count - current
        else:
            added = 0

        final = current + added
        total_after += final

        print(
            f"{label:30} "
            f"{current:6} -> {final:6} "
            f"(add {added:6})"
        )

    print(
        "Total after full balancing:",
        total_after
    )


def main():

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training file not found: {TRAIN_FILE}"
        )

    train = pd.read_csv(
        TRAIN_FILE
    )

    print("=" * 80)
    print("TCMAPY2573 - CLASS IMBALANCE ANALYSIS")
    print("=" * 80)

    print(
        f"Training records: {len(train)}"
    )

    stage1_labels = [
        "Normal",
        "Non-Normal",
    ]

    stage2_labels = [
        "Normal",
        "Hyperthyroid",
        "Hypothyroid",
        "Other / Related Diagnosis",
    ]

    analyze_target(
        train["stage1_label"],
        "STAGE 1",
        stage1_labels,
    )

    analyze_target(
        train["stage2_label"],
        "STAGE 2",
        stage2_labels,
    )

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    print()
    print("IMPORTANT:")
    print("- No resampling was performed.")
    print("- No SMOTE was performed.")
    print("- No class weights were applied to a model.")
    print("- Test data was not used.")
    print("- This step only analyzes imbalance.")


if __name__ == "__main__":
    main()