from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# TCMAPY2573
# Calibrated Stage 1 Confidence Reliability Audit
# Step 351
# ---------------------------------------------------------

X_FILE = Path(
    "data/enhanced_X_train_processed.csv"
)

RAW_FILE = Path(
    "data/enhanced_train_with_targets.csv"
)

CALIBRATED_MODEL_FILE = Path(
    "model/enhanced_stage1_calibrated_model.pkl"
)

OUTPUT_FILE = Path(
    "data/enhanced_calibrated_confidence_reliability.csv"
)

RANDOM_STATE = 42


def calculate_ece(
    y_correct,
    confidence,
    number_of_bins=10,
):
    """
    Expected Calibration Error for confidence.

    y_correct:
        1 = prediction correct
        0 = prediction incorrect

    confidence:
        Probability assigned to the predicted class.
    """

    y_correct = np.asarray(
        y_correct,
        dtype=int,
    )

    confidence = np.asarray(
        confidence,
        dtype=float,
    )

    bins = np.linspace(
        0.0,
        1.0,
        number_of_bins + 1,
    )

    ece = 0.0
    rows = []

    for i in range(number_of_bins):

        lower = bins[i]
        upper = bins[i + 1]

        if i == number_of_bins - 1:
            mask = (
                (confidence >= lower)
                & (confidence <= upper)
            )
        else:
            mask = (
                (confidence >= lower)
                & (confidence < upper)
            )

        count = int(mask.sum())

        if count == 0:
            continue

        mean_confidence = float(
            confidence[mask].mean()
        )

        observed_accuracy = float(
            y_correct[mask].mean()
        )

        gap = abs(
            mean_confidence
            - observed_accuracy
        )

        ece += (
            count
            / len(confidence)
            * gap
        )

        rows.append(
            {
                "bin": i + 1,
                "lower": lower,
                "upper": upper,
                "samples": count,
                "mean_confidence": mean_confidence,
                "observed_accuracy": observed_accuracy,
                "absolute_gap": gap,
            }
        )

    return ece, pd.DataFrame(rows)


def main():

    print("=" * 80)
    print(
        "TCMAPY2573 - CALIBRATED STAGE 1 CONFIDENCE AUDIT"
    )
    print("=" * 80)

    # -----------------------------------------------------
    # Load data/model
    # -----------------------------------------------------

    X = pd.read_csv(
        X_FILE
    )

    raw = pd.read_csv(
        RAW_FILE
    )

    calibrated_model = joblib.load(
        CALIBRATED_MODEL_FILE
    )

    y = raw[
        "stage1_target"
    ].astype(int)

    print(
        "Training records:",
        len(X)
    )

    print(
        "Loaded model:",
        type(calibrated_model).__name__
    )

    # -----------------------------------------------------
    # Recreate Step 349 partition
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

    print()
    print("PARTITION USED FOR AUDIT")
    print("-" * 80)

    print(
        "Base-model fitting:",
        len(X_fit)
    )

    print(
        "Calibration:",
        len(X_calibration)
    )

    print(
        "Threshold evaluation:",
        len(X_threshold)
    )

    # -----------------------------------------------------
    # Verify model was created for Stage 1 calibration
    # -----------------------------------------------------

    print()
    print("CALIBRATED MODEL CHECK")
    print("-" * 80)

    print(
        "Model class:",
        type(calibrated_model).__name__
    )

    # -----------------------------------------------------
    # Predict with saved calibrated model
    # -----------------------------------------------------

    calibrated_probability = (
        calibrated_model.predict_proba(
            X_threshold
        )[:, 1]
    )

    predicted = (
        calibrated_probability >= 0.50
    ).astype(int)

    actual = (
        y_threshold.to_numpy()
    )

    correct = (
        predicted == actual
    )

    # -----------------------------------------------------
    # Confidence of predicted class
    # -----------------------------------------------------

    confidence = np.where(
        predicted == 1,
        calibrated_probability,
        1.0 - calibrated_probability,
    )

    # -----------------------------------------------------
    # ECE
    # -----------------------------------------------------

    ece, reliability = calculate_ece(
        correct.astype(int),
        confidence,
        number_of_bins=10,
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

        mean_confidence = float(
            confidence[mask].mean()
        )

        observed_accuracy = float(
            correct[mask].mean()
        )

        band_rows.append(
            {
                "confidence_lower": lower,
                "confidence_upper": upper,
                "samples": count,
                "mean_confidence": mean_confidence,
                "observed_accuracy": observed_accuracy,
                "absolute_gap": abs(
                    mean_confidence
                    - observed_accuracy
                ),
            }
        )

    bands_df = pd.DataFrame(
        band_rows
    )

    # -----------------------------------------------------
    # 0.95 confidence analysis
    # -----------------------------------------------------

    high_confidence = (
        confidence >= 0.95
    )

    high_count = int(
        high_confidence.sum()
    )

    if high_count > 0:

        high_mean_confidence = float(
            confidence[
                high_confidence
            ].mean()
        )

        high_accuracy = float(
            correct[
                high_confidence
            ].mean()
        )

    else:

        high_mean_confidence = np.nan
        high_accuracy = np.nan

    # -----------------------------------------------------
    # Candidate 0.95 routing statistics
    # -----------------------------------------------------

    low_confidence = (
        confidence < 0.95
    )

    confident_count = int(
        high_confidence.sum()
    )

    low_confidence_count = int(
        low_confidence.sum()
    )

    coverage = (
        confident_count
        / len(confidence)
    )

    uncertainty_rate = (
        low_confidence_count
        / len(confidence)
    )

    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("CALIBRATED CONFIDENCE RESULTS")
    print("=" * 80)

    print(
        f"ECE: {ece:.6f}"
    )

    print(
        f"Samples with confidence >= 0.95: "
        f"{high_count}"
    )

    print(
        f"Mean confidence >= 0.95: "
        f"{high_mean_confidence:.6f}"
    )

    print(
        f"Observed accuracy >= 0.95: "
        f"{high_accuracy:.6f}"
    )

    print(
        f"Confidence >= 0.95 coverage: "
        f"{coverage:.6f}"
    )

    print(
        f"Confidence < 0.95 rate: "
        f"{uncertainty_rate:.6f}"
    )

    print()
    print("=" * 80)
    print("CALIBRATED CONFIDENCE BANDS")
    print("=" * 80)

    if len(bands_df) > 0:
        print(
            bands_df.to_string(
                index=False
            )
        )

    print()
    print("=" * 80)
    print("CALIBRATED RELIABILITY BINS")
    print("=" * 80)

    if len(reliability) > 0:
        print(
            reliability.to_string(
                index=False
            )
        )

    # -----------------------------------------------------
    # Save combined report
    # -----------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    bands_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 80)
    print("STEP 351 COMPLETED")
    print("=" * 80)

    print(
        "Reliability report:",
        OUTPUT_FILE
    )

    print()
    print("IMPORTANT:")
    print("- Saved CALIBRATED model was used.")
    print("- Final 1835-record test set was NOT used.")
    print("- No threshold was changed.")
    print("- No Stage 2 model was changed.")
    print("- No Flask code was changed.")
    print("- This validates calibrated confidence only.")


if __name__ == "__main__":
    main()