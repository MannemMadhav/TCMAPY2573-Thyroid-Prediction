from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import (
    CalibratedClassifierCV,
    calibration_curve,
)
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


# ---------------------------------------------------------
# TCMAPY2573
# Stage 1 Probability Calibration + Confidence Threshold
# Step 349
# ---------------------------------------------------------

X_FILE = Path(
    "data/enhanced_X_train_processed.csv"
)

RAW_TRAIN_FILE = Path(
    "data/enhanced_train_with_targets.csv"
)

OUTPUT_METRICS = Path(
    "data/enhanced_stage1_calibration_metrics.csv"
)

OUTPUT_THRESHOLDS = Path(
    "data/enhanced_stage1_calibrated_threshold_analysis.csv"
)

OUTPUT_RELIABILITY = Path(
    "data/enhanced_stage1_reliability_data.csv"
)

OUTPUT_CALIBRATED_MODEL = Path(
    "model/enhanced_stage1_calibrated_model.pkl"
)

RANDOM_STATE = 42


# ---------------------------------------------------------
# XGB_D selected in Step 347
# ---------------------------------------------------------

MODEL_CONFIG = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.03,
    "min_child_weight": 2,
    "subsample": 0.90,
    "colsample_bytree": 0.90,
    "reg_lambda": 3.0,
}


def create_xgb():

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


def main():

    print("=" * 80)
    print("TCMAPY2573 - STAGE 1 CALIBRATION + CONFIDENCE ANALYSIS")
    print("=" * 80)

    # -----------------------------------------------------
    # Load training data
    # -----------------------------------------------------

    X = pd.read_csv(X_FILE)

    raw = pd.read_csv(
        RAW_TRAIN_FILE
    )

    y = raw[
        "stage1_target"
    ].astype(int)

    print(
        f"Total training records: {len(X)}"
    )

    # -----------------------------------------------------
    # First split:
    #
    # 80% development
    # 20% final threshold analysis
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

    # -----------------------------------------------------
    # Second split:
    #
    # 75% model fitting
    # 25% calibration
    #
    # Approximate final proportions:
    #
    # Model fitting  = 60%
    # Calibration     = 20%
    # Threshold       = 20%
    # -----------------------------------------------------

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
    print("=" * 80)
    print("THREE-WAY TRAINING PARTITION")
    print("=" * 80)

    print(
        "Base-model fitting records:",
        len(X_fit)
    )

    print(
        "Calibration records:",
        len(X_calibration)
    )

    print(
        "Threshold-tuning records:",
        len(X_threshold)
    )

    # -----------------------------------------------------
    # Train base XGB_D
    # -----------------------------------------------------

    print()
    print(
        "Training base XGB_D..."
    )

    base_model = create_xgb()

    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_fit,
    )

    base_model.fit(
        X_fit,
        y_fit,
        sample_weight=sample_weights,
    )

    print(
        "Base model trained."
    )

    # -----------------------------------------------------
    # Raw probabilities on calibration set
    # -----------------------------------------------------

    raw_calibration_probability = (
        base_model.predict_proba(
            X_calibration
        )[:, 1]
    )

    # -----------------------------------------------------
    # Calibrate using independent calibration data
    #
    # FrozenEstimator prevents retraining the XGB model.
    # -----------------------------------------------------

    print()
    print(
        "Calibrating Stage 1 probabilities..."
    )

    calibrated_model = CalibratedClassifierCV(
        estimator=FrozenEstimator(
            base_model
        ),
        method="sigmoid",
    )

    calibrated_model.fit(
        X_calibration,
        y_calibration,
    )

    print(
        "Probability calibration complete."
    )

    # -----------------------------------------------------
    # Evaluate raw and calibrated probabilities
    # on the independent threshold set
    # -----------------------------------------------------

    raw_probability = (
        base_model.predict_proba(
            X_threshold
        )[:, 1]
    )

    calibrated_probability = (
        calibrated_model.predict_proba(
            X_threshold
        )[:, 1]
    )

    # -----------------------------------------------------
    # Probability quality metrics
    # -----------------------------------------------------

    raw_brier = brier_score_loss(
        y_threshold,
        raw_probability,
    )

    calibrated_brier = brier_score_loss(
        y_threshold,
        calibrated_probability,
    )

    raw_logloss = log_loss(
        y_threshold,
        np.column_stack(
            [
                1 - raw_probability,
                raw_probability,
            ]
        ),
    )

    calibrated_logloss = log_loss(
        y_threshold,
        np.column_stack(
            [
                1 - calibrated_probability,
                calibrated_probability,
            ]
        ),
    )

    raw_auc = roc_auc_score(
        y_threshold,
        raw_probability,
    )

    calibrated_auc = roc_auc_score(
        y_threshold,
        calibrated_probability,
    )

    print()
    print("=" * 80)
    print("PROBABILITY QUALITY COMPARISON")
    print("=" * 80)

    print(
        f"Raw Brier score       : {raw_brier:.6f}"
    )

    print(
        f"Calibrated Brier score: {calibrated_brier:.6f}"
    )

    print(
        f"Raw Log Loss          : {raw_logloss:.6f}"
    )

    print(
        f"Calibrated Log Loss   : {calibrated_logloss:.6f}"
    )

    print(
        f"Raw ROC-AUC           : {raw_auc:.6f}"
    )

    print(
        f"Calibrated ROC-AUC    : {calibrated_auc:.6f}"
    )

    # -----------------------------------------------------
    # Reliability curve data
    # -----------------------------------------------------

    raw_fraction, raw_mean = calibration_curve(
        y_threshold,
        raw_probability,
        n_bins=10,
        strategy="uniform",
    )

    calibrated_fraction, calibrated_mean = (
        calibration_curve(
            y_threshold,
            calibrated_probability,
            n_bins=10,
            strategy="uniform",
        )
    )

    reliability_length = max(
        len(raw_fraction),
        len(calibrated_fraction),
    )

    reliability_rows = []

    for index in range(
        reliability_length
    ):

        raw_predicted = (
            raw_mean[index]
            if index < len(raw_mean)
            else np.nan
        )

        raw_observed = (
            raw_fraction[index]
            if index < len(raw_fraction)
            else np.nan
        )

        calibrated_predicted = (
            calibrated_mean[index]
            if index < len(calibrated_mean)
            else np.nan
        )

        calibrated_observed = (
            calibrated_fraction[index]
            if index < len(calibrated_fraction)
            else np.nan
        )

        reliability_rows.append(
            {
                "bin": index + 1,
                "raw_mean_probability": raw_predicted,
                "raw_fraction_positive": raw_observed,
                "calibrated_mean_probability": calibrated_predicted,
                "calibrated_fraction_positive": calibrated_observed,
            }
        )

    reliability_df = pd.DataFrame(
        reliability_rows
    )

    # -----------------------------------------------------
    # Confidence thresholds
    # -----------------------------------------------------

    thresholds = np.arange(
        0.50,
        0.951,
        0.025,
    )

    threshold_rows = []

    for threshold in thresholds:

        lower_normal_limit = (
            1.0 - threshold
        )

        confident_normal = (
            calibrated_probability
            <= lower_normal_limit
        )

        confident_non_normal = (
            calibrated_probability
            >= threshold
        )

        uncertain = (
            ~confident_normal
            & ~confident_non_normal
        )

        confident = (
            confident_normal
            | confident_non_normal
        )

        coverage = (
            confident.mean()
        )

        uncertainty_rate = (
            uncertain.mean()
        )

        if confident.any():

            actual_confident = (
                y_threshold
                .to_numpy()[confident]
            )

            predicted_confident = np.where(
                confident_non_normal[
                    confident
                ],
                1,
                0,
            )

            selective_accuracy = (
                accuracy_score(
                    actual_confident,
                    predicted_confident,
                )
            )

            selective_macro_f1 = (
                f1_score(
                    actual_confident,
                    predicted_confident,
                    average="macro",
                    zero_division=0,
                )
            )

        else:

            selective_accuracy = np.nan
            selective_macro_f1 = np.nan

        threshold_rows.append(
            {
                "threshold": round(
                    float(threshold),
                    3,
                ),
                "lower_normal_limit": round(
                    float(lower_normal_limit),
                    3,
                ),
                "coverage": round(
                    float(coverage),
                    4,
                ),
                "uncertainty_rate": round(
                    float(uncertainty_rate),
                    4,
                ),
                "selective_accuracy": round(
                    float(selective_accuracy),
                    4,
                ),
                "selective_macro_f1": round(
                    float(selective_macro_f1),
                    4,
                ),
                "confident_normal_count": int(
                    confident_normal.sum()
                ),
                "confident_non_normal_count": int(
                    confident_non_normal.sum()
                ),
                "uncertain_count": int(
                    uncertain.sum()
                ),
            }
        )

    threshold_df = pd.DataFrame(
        threshold_rows
    )

    # -----------------------------------------------------
    # Select candidate based on calibrated probabilities
    # -----------------------------------------------------

    selected = (
        threshold_df
        .sort_values(
            by=[
                "selective_macro_f1",
                "coverage",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .iloc[0]
    )

    candidate_threshold = float(
        selected["threshold"]
    )

    # -----------------------------------------------------
    # Save files
    # -----------------------------------------------------

    OUTPUT_METRICS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_df = pd.DataFrame(
        [
            {
                "raw_brier_score": raw_brier,
                "calibrated_brier_score": calibrated_brier,
                "raw_log_loss": raw_logloss,
                "calibrated_log_loss": calibrated_logloss,
                "raw_roc_auc": raw_auc,
                "calibrated_roc_auc": calibrated_auc,
                "candidate_threshold": candidate_threshold,
            }
        ]
    )

    metrics_df.to_csv(
        OUTPUT_METRICS,
        index=False,
    )

    threshold_df.to_csv(
        OUTPUT_THRESHOLDS,
        index=False,
    )

    reliability_df.to_csv(
        OUTPUT_RELIABILITY,
        index=False,
    )

    # -----------------------------------------------------
    # Save calibrated model
    # -----------------------------------------------------

    joblib.dump(
        calibrated_model,
        OUTPUT_CALIBRATED_MODEL,
    )

    # -----------------------------------------------------
    # Display threshold table
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("CALIBRATED CONFIDENCE THRESHOLD ANALYSIS")
    print("=" * 80)

    print(
        threshold_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("CALIBRATED THRESHOLD CANDIDATE")
    print("=" * 80)

    print(
        "Candidate threshold:",
        candidate_threshold,
    )

    print(
        "Selection criterion:",
        "highest calibrated selective Macro F1",
    )

    print()
    print("=" * 80)
    print("STEP 349 ANALYSIS COMPLETED")
    print("=" * 80)

    print(
        "Calibration metrics:",
        OUTPUT_METRICS,
    )

    print(
        "Threshold analysis:",
        OUTPUT_THRESHOLDS,
    )

    print(
        "Reliability data:",
        OUTPUT_RELIABILITY,
    )

    print(
        "Calibrated model:",
        OUTPUT_CALIBRATED_MODEL,
    )

    print()
    print("IMPORTANT:")
    print("- Only the 7337-record training set was used.")
    print("- Base model fitting, calibration and threshold tuning used separate subsets.")
    print("- The 1835-record final test set was NOT used.")
    print("- No Stage 2 model was trained.")
    print("- No Flask code was modified.")
    print("- The threshold is still a candidate until we inspect the result.")


if __name__ == "__main__":
    main()