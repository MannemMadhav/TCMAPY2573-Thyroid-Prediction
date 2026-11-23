from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier


# ---------------------------------------------------------
# TCMAPY2573
# Stage 1 Confidence Threshold Analysis
# Step 348
# ---------------------------------------------------------

X_FILE = Path(
    "data/enhanced_X_train_processed.csv"
)

RAW_TRAIN_FILE = Path(
    "data/enhanced_train_with_targets.csv"
)

OUTPUT_FILE = Path(
    "data/enhanced_stage1_threshold_analysis.csv"
)


RANDOM_STATE = 123


# ---------------------------------------------------------
# Selected XGB_D configuration from Step 347
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
        random_state=42,
        n_jobs=-1,
    )


def main():

    print("=" * 80)
    print("TCMAPY2573 - STAGE 1 CONFIDENCE THRESHOLD ANALYSIS")
    print("=" * 80)

    # -----------------------------------------------------
    # Load training data only
    # -----------------------------------------------------

    X = pd.read_csv(X_FILE)

    raw = pd.read_csv(
        RAW_TRAIN_FILE
    )

    y = raw[
        "stage1_target"
    ].astype(int)

    print(
        "Total training records:",
        len(X)
    )

    # -----------------------------------------------------
    # Separate threshold-tuning data
    # -----------------------------------------------------

    (
        X_fit,
        X_threshold,
        y_fit,
        y_threshold,
    ) = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(
        "Model-fitting records:",
        len(X_fit)
    )

    print(
        "Threshold-tuning records:",
        len(X_threshold)
    )

    # -----------------------------------------------------
    # Fit fixed XGB_D model
    # -----------------------------------------------------

    model = create_model()

    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_fit,
    )

    print()
    print(
        "Fitting XGB_D on threshold-training data..."
    )

    model.fit(
        X_fit,
        y_fit,
        sample_weight=sample_weights,
    )

    print(
        "Model fitting complete."
    )

    # -----------------------------------------------------
    # Get probability of Non-Normal
    # -----------------------------------------------------

    probability_non_normal = (
        model.predict_proba(
            X_threshold
        )[:, 1]
    )

    probability_normal = (
        1.0 - probability_non_normal
    )

    # -----------------------------------------------------
    # Candidate thresholds
    # -----------------------------------------------------

    thresholds = np.arange(
        0.50,
        0.951,
        0.025,
    )

    results = []

    print()
    print("=" * 80)
    print("THRESHOLD ANALYSIS")
    print("=" * 80)

    for threshold in thresholds:

        normal_limit = 1.0 - threshold

        confident_normal = (
            probability_non_normal
            <= normal_limit
        )

        confident_non_normal = (
            probability_non_normal
            >= threshold
        )

        uncertain = (
            ~confident_normal
            & ~confident_non_normal
        )

        confident_mask = (
            confident_normal
            | confident_non_normal
        )

        coverage = (
            confident_mask.mean()
        )

        uncertainty_rate = (
            uncertain.mean()
        )

        if confident_mask.any():

            confident_actual = (
                y_threshold
                .to_numpy()[confident_mask]
            )

            confident_predicted = np.where(
                confident_non_normal[
                    confident_mask
                ],
                1,
                0,
            )

            selective_accuracy = (
                accuracy_score(
                    confident_actual,
                    confident_predicted,
                )
            )

            selective_macro_f1 = (
                f1_score(
                    confident_actual,
                    confident_predicted,
                    average="macro",
                    zero_division=0,
                )
            )

        else:

            selective_accuracy = np.nan
            selective_macro_f1 = np.nan

        # Overall rule with uncertain cases sent to Stage 2.
        # Here we record Stage 1's confident decisions only.
        results.append(
            {
                "threshold": round(
                    float(threshold),
                    3,
                ),
                "lower_normal_limit": round(
                    float(normal_limit),
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

    results_df = pd.DataFrame(
        results
    )

    # -----------------------------------------------------
    # Display
    # -----------------------------------------------------

    print(
        results_df.to_string(
            index=False
        )
    )

    # -----------------------------------------------------
    # Select candidate threshold
    # -----------------------------------------------------

    # Primary criterion:
    # highest selective Macro F1.
    #
    # Secondary criterion:
    # higher coverage.
    selected = (
        results_df
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

    selected_threshold = float(
        selected["threshold"]
    )

    print()
    print("=" * 80)
    print("THRESHOLD CANDIDATE")
    print("=" * 80)

    print(
        "Candidate threshold:",
        selected_threshold
    )

    print(
        "Threshold selected using:",
        "threshold-tuning training subset"
    )

    print(
        "Selection criterion:",
        "highest selective Macro F1"
    )

    # -----------------------------------------------------
    # Save analysis
    # -----------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 80)
    print("STEP 348 ANALYSIS COMPLETED")
    print("=" * 80)

    print(
        "Analysis saved:",
        OUTPUT_FILE
    )

    print()
    print("IMPORTANT:")
    print("- Only training data was used.")
    print("- The final 1835-record test set was NOT used.")
    print("- XGB_D configuration was fixed from Step 347.")
    print("- Threshold was not tuned on the test set.")
    print("- No Stage 2 model was trained.")
    print("- No Flask code was changed.")
    print("- This is a threshold analysis only.")


if __name__ == "__main__":
    main()