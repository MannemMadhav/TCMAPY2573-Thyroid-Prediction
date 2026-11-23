from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


# ---------------------------------------------------------
# TCMAPY2573
# Stage 1 Confidence Reliability Audit
# Step 350
# ---------------------------------------------------------

X_FILE = Path(
    "data/enhanced_X_train_processed.csv"
)

RAW_FILE = Path(
    "data/enhanced_train_with_targets.csv"
)

OUTPUT_FILE = Path(
    "data/enhanced_stage1_confidence_reliability.csv"
)

RANDOM_STATE = 42


MODEL_CONFIG = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.03,
    "min_child_weight": 2,
    "subsample": 0.90,
    "colsample_bytree": 0.90,
    "reg_lambda": 3.0,
}


def create_model():

    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=MODEL_CONFIG["n_estimators"],
        max_depth=MODEL_CONFIG["max_depth"],
        learning_rate=MODEL_CONFIG["learning_rate"],
        min_child_weight=MODEL_CONFIG["min_child_weight"],
        subsample=MODEL_CONFIG["subsample"],
        colsample_bytree=MODEL_CONFIG["colsample_bytree"],
        reg_lambda=MODEL_CONFIG["reg_lambda"],
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def expected_calibration_error(
    y_true,
    probabilities,
    number_of_bins=10,
):

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    bin_edges = np.linspace(
        0.0,
        1.0,
        number_of_bins + 1,
    )

    ece = 0.0
    rows = []

    for index in range(number_of_bins):

        lower = bin_edges[index]
        upper = bin_edges[index + 1]

        if index == number_of_bins - 1:
            mask = (
                (probabilities >= lower)
                & (probabilities <= upper)
            )
        else:
            mask = (
                (probabilities >= lower)
                & (probabilities < upper)
            )

        count = int(mask.sum())

        if count == 0:
            continue

        bin_probability = float(
            probabilities[mask].mean()
        )

        bin_accuracy = float(
            y_true[mask].mean()
        )

        bin_weight = (
            count / len(probabilities)
        )

        ece += (
            bin_weight
            * abs(
                bin_accuracy
                - bin_probability
            )
        )

        rows.append(
            {
                "bin": index + 1,
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_probability": bin_probability,
                "observed_positive_rate": bin_accuracy,
                "absolute_gap": abs(
                    bin_accuracy
                    - bin_probability
                ),
            }
        )

    return ece, pd.DataFrame(rows)


def main():

    print("=" * 80)
    print(
        "TCMAPY2573 - STAGE 1 CONFIDENCE RELIABILITY AUDIT"
    )
    print("=" * 80)

    X = pd.read_csv(X_FILE)

    raw = pd.read_csv(RAW_FILE)

    y = raw[
        "stage1_target"
    ].astype(int)

    # -----------------------------------------------------
    # Recreate the same internal partition structure
    # used during Step 349
    # -----------------------------------------------------

    (
        X_development,
        X_threshold,
        y_development,
        y_threshold,
    ) = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    (
        X_fit,
        X_calibration,
        y_fit,
        y_calibration,
    ) = train_test_split(
        X_development,
        y_development,
        test_size=0.25,
        stratify=y_development,
        random_state=RANDOM_STATE,
    )

    # -----------------------------------------------------
    # Train XGB_D
    # -----------------------------------------------------

    model = create_model()

    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_fit,
    )

    print()
    print("Training XGB_D...")

    model.fit(
        X_fit,
        y_fit,
        sample_weight=sample_weights,
    )

    print("Training complete.")

    # -----------------------------------------------------
    # Raw probabilities
    # -----------------------------------------------------

    probabilities = model.predict_proba(
        X_threshold
    )[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    accuracy = accuracy_score(
        y_threshold,
        predictions,
    )

    # -----------------------------------------------------
    # Confidence
    #
    # Confidence is the probability of the
    # predicted class.
    # -----------------------------------------------------

    confidence = np.where(
        predictions == 1,
        probabilities,
        1.0 - probabilities,
    )

    correct = (
        predictions
        == y_threshold.to_numpy()
    )

    # -----------------------------------------------------
    # ECE
    # -----------------------------------------------------

    ece, reliability = (
        expected_calibration_error(
            correct.astype(int),
            confidence,
            number_of_bins=10,
        )
    )

    # -----------------------------------------------------
    # Confidence bands
    # -----------------------------------------------------

    confidence_bands = [
        (0.50, 0.60),
        (0.60, 0.70),
        (0.70, 0.80),
        (0.80, 0.90),
        (0.90, 0.95),
        (0.95, 1.01),
    ]

    band_rows = []

    for lower, upper in confidence_bands:

        mask = (
            (confidence >= lower)
            & (confidence < upper)
        )

        count = int(mask.sum())

        if count == 0:
            continue

        observed_accuracy = float(
            correct[mask].mean()
        )

        mean_confidence = float(
            confidence[mask].mean()
        )

        band_rows.append(
            {
                "confidence_lower": lower,
                "confidence_upper": upper,
                "samples": count,
                "mean_confidence": mean_confidence,
                "observed_accuracy": observed_accuracy,
                "confidence_accuracy_gap": abs(
                    mean_confidence
                    - observed_accuracy
                ),
            }
        )

    band_df = pd.DataFrame(
        band_rows
    )

    # -----------------------------------------------------
    # High-confidence performance
    # -----------------------------------------------------

    high_confidence = (
        confidence >= 0.95
    )

    high_conf_count = int(
        high_confidence.sum()
    )

    if high_conf_count > 0:

        high_conf_accuracy = float(
            correct[high_confidence].mean()
        )

        high_conf_mean = float(
            confidence[high_confidence].mean()
        )

    else:

        high_conf_accuracy = np.nan
        high_conf_mean = np.nan

    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("OVERALL CONFIDENCE")
    print("=" * 80)

    print(
        f"Threshold-set accuracy: {accuracy:.4f}"
    )

    print(
        f"ECE: {ece:.6f}"
    )

    print()
    print(
        f"Samples with confidence >= 0.95: "
        f"{high_conf_count}"
    )

    print(
        f"Mean confidence >= 0.95: "
        f"{high_conf_mean:.6f}"
    )

    print(
        f"Observed accuracy >= 0.95: "
        f"{high_conf_accuracy:.6f}"
    )

    print()
    print("=" * 80)
    print("CONFIDENCE BANDS")
    print("=" * 80)

    if len(band_df) > 0:
        print(
            band_df.to_string(
                index=False
            )
        )

    print()
    print("=" * 80)
    print("RELIABILITY BINS")
    print("=" * 80)

    if len(reliability) > 0:
        print(
            reliability.to_string(
                index=False
            )
        )

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    band_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 80)
    print("STEP 350 COMPLETED")
    print("=" * 80)

    print(
        "Reliability report saved:",
        OUTPUT_FILE,
    )

    print()
    print("IMPORTANT:")
    print("- Only training-derived data was used.")
    print("- Final test data was NOT used.")
    print("- This is a confidence audit only.")
    print("- No threshold was changed.")
    print("- No Stage 2 model was changed.")
    print("- No Flask code was changed.")


if __name__ == "__main__":
    main()