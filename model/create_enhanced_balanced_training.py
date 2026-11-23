from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTENC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder


# ---------------------------------------------------------
# TCMAPY2573
# Enhanced Training Class Balancing
# Step 345
# ---------------------------------------------------------

TRAIN_FILE = Path(
    "data/enhanced_train_with_targets.csv"
)

PREPROCESSOR_FILE = Path(
    "model/enhanced_preprocessor.pkl"
)

OUTPUT_X_FILE = Path(
    "data/enhanced_stage2_smote_X_train_processed.csv"
)

OUTPUT_Y_FILE = Path(
    "data/enhanced_stage2_smote_y_train.csv"
)

OUTPUT_INFO_FILE = Path(
    "data/enhanced_stage2_smote_info.csv"
)

OUTPUT_WEIGHT_FILE = Path(
    "data/enhanced_stage1_class_weights.csv"
)


# ---------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------

NUMERICAL_FEATURES = [
    "age",
    "TSH",
    "T3",
    "TT4",
    "T4U",
    "FTI",
    "TBG",
]

BINARY_FEATURES = [
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
]

CATEGORICAL_FEATURES = [
    "sex",
    "referral_source",
]

ALL_FEATURES = (
    NUMERICAL_FEATURES
    + BINARY_FEATURES
    + CATEGORICAL_FEATURES
)


# ---------------------------------------------------------
# Stage 2 target mapping
# ---------------------------------------------------------

STAGE2_LABELS = {
    0: "Normal",
    1: "Hyperthyroid",
    2: "Hypothyroid",
    3: "Other / Related Diagnosis",
}


# ---------------------------------------------------------
# Controlled SMOTENC target
# ---------------------------------------------------------

SMOTE_TARGETS = {
    1: 1000,   # Hyperthyroid
    2: 1000,   # Hypothyroid
}


def main():

    print("=" * 80)
    print("TCMAPY2573 - ENHANCED TRAINING CLASS BALANCING")
    print("=" * 80)

    # -----------------------------------------------------
    # Load training data
    # -----------------------------------------------------

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training file not found: {TRAIN_FILE}"
        )

    if not PREPROCESSOR_FILE.exists():
        raise FileNotFoundError(
            f"Preprocessor not found: {PREPROCESSOR_FILE}"
        )

    train = pd.read_csv(TRAIN_FILE)

    print(
        f"Original training records: {len(train)}"
    )

    # -----------------------------------------------------
    # Load already fitted preprocessing
    # -----------------------------------------------------

    preprocessor = joblib.load(
        PREPROCESSOR_FILE
    )

    print(
        "Loaded preprocessor:",
        type(preprocessor).__name__
    )

    # -----------------------------------------------------
    # Create Stage 1 reference class weights
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("STAGE 1 - CLASS WEIGHT STRATEGY")
    print("=" * 80)

    stage1_counts = (
        train["stage1_label"]
        .value_counts()
    )

    stage1_total = len(train)

    stage1_weights = {}

    for label in [
        "Normal",
        "Non-Normal",
    ]:

        count = int(
            stage1_counts[label]
        )

        weight = (
            stage1_total /
            (
                2 * count
            )
        )

        stage1_weights[label] = weight

        print(
            f"{label:15} "
            f"count={count:5} "
            f"weight={weight:.4f}"
        )

    stage1_weight_df = pd.DataFrame(
        {
            "class_label": list(
                stage1_weights.keys()
            ),
            "class_weight": list(
                stage1_weights.values()
            ),
        }
    )

    stage1_weight_df.to_csv(
        OUTPUT_WEIGHT_FILE,
        index=False
    )

    # -----------------------------------------------------
    # Prepare Stage 2 raw training data
    # -----------------------------------------------------

    X_raw = train[
        ALL_FEATURES
    ].copy()

    y_stage2 = (
        train["stage2_target"]
        .astype(int)
        .to_numpy()
    )

    # -----------------------------------------------------
    # Impute numerical features
    # -----------------------------------------------------

    numerical_imputer = SimpleImputer(
        strategy="median"
    )

    X_num = numerical_imputer.fit_transform(
        X_raw[NUMERICAL_FEATURES]
    )

    # -----------------------------------------------------
    # Impute categorical/binary features
    # -----------------------------------------------------

    categorical_input = X_raw[
        BINARY_FEATURES + CATEGORICAL_FEATURES
    ].copy()

    categorical_imputer = SimpleImputer(
        strategy="most_frequent"
    )

    categorical_imputed = (
        categorical_imputer.fit_transform(
            categorical_input
        )
    )

    # -----------------------------------------------------
    # Convert categorical variables to integer codes
    # -----------------------------------------------------

    ordinal_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    X_cat = ordinal_encoder.fit_transform(
        categorical_imputed
    )

    # -----------------------------------------------------
    # Combine numerical + categorical
    # -----------------------------------------------------

    X_intermediate = np.column_stack(
        [
            X_num,
            X_cat,
        ]
    )

    categorical_start = len(
        NUMERICAL_FEATURES
    )

    categorical_indices = list(
        range(
            categorical_start,
            X_intermediate.shape[1]
        )
    )

    print()
    print("=" * 80)
    print("SMOTENC PREPARATION")
    print("=" * 80)

    print(
        "Numerical features:",
        len(NUMERICAL_FEATURES)
    )

    print(
        "Categorical/binary features:",
        len(BINARY_FEATURES)
        + len(CATEGORICAL_FEATURES)
    )

    print(
        "Categorical indices:",
        categorical_indices
    )

    # -----------------------------------------------------
    # Original Stage 2 distribution
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("STAGE 2 BEFORE SMOTENC")
    print("=" * 80)

    before_counts = (
        pd.Series(y_stage2)
        .value_counts()
        .sort_index()
    )

    for class_id, count in before_counts.items():

        print(
            f"{STAGE2_LABELS[int(class_id)]:30} "
            f"{int(count)}"
        )

    # -----------------------------------------------------
    # Controlled SMOTENC
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("APPLYING CONTROLLED SMOTENC")
    print("=" * 80)

    print(
        "Target synthetic counts:",
        SMOTE_TARGETS
    )

    smotenc = SMOTENC(
    categorical_features=categorical_indices,
    sampling_strategy=SMOTE_TARGETS,
    random_state=42,
    k_neighbors=5,
)
    X_balanced, y_balanced = smotenc.fit_resample(
        X_intermediate,
        y_stage2,
    )

    print(
        "SMOTENC completed."
    )

    # -----------------------------------------------------
    # Restore categorical values
    # -----------------------------------------------------

    X_balanced = np.asarray(
        X_balanced,
        dtype=float
    )

    X_balanced_cat = np.rint(
        X_balanced[:, categorical_indices]
    ).astype(int)

    # Protect against encoder-boundary rounding.
    max_codes = (
        ordinal_encoder
        .categories_
    )

    for column_index, categories in enumerate(
        max_codes
    ):

        minimum = 0
        maximum = len(categories) - 1

        X_balanced_cat[:, column_index] = np.clip(
            X_balanced_cat[:, column_index],
            minimum,
            maximum,
        )

    restored_categories = (
        ordinal_encoder.inverse_transform(
            X_balanced_cat
        )
    )

    # -----------------------------------------------------
    # Reconstruct raw-format training data
    # -----------------------------------------------------

    balanced_numerical = X_balanced[
        :,
        :len(NUMERICAL_FEATURES)
    ]

    balanced_raw = pd.DataFrame(
        balanced_numerical,
        columns=NUMERICAL_FEATURES,
    )

    balanced_categorical = pd.DataFrame(
        restored_categories,
        columns=(
            BINARY_FEATURES
            + CATEGORICAL_FEATURES
        ),
    )

    balanced_raw = pd.concat(
        [
            balanced_raw,
            balanced_categorical,
        ],
        axis=1,
    )

    # Ensure original feature order.
    balanced_raw = balanced_raw[
        ALL_FEATURES
    ]

    # -----------------------------------------------------
    # Transform synthetic/raw balanced data
    # using the preprocessor fitted on ORIGINAL
    # training data only.
    # -----------------------------------------------------

    X_balanced_processed = (
        preprocessor.transform(
            balanced_raw
        )
    )

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    X_balanced_processed = pd.DataFrame(
        X_balanced_processed,
        columns=feature_names,
    )

    # -----------------------------------------------------
    # Save balanced training data
    # -----------------------------------------------------

    OUTPUT_X_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    X_balanced_processed.to_csv(
        OUTPUT_X_FILE,
        index=False,
    )

    pd.DataFrame(
        {
            "stage2_target": y_balanced,
            "stage2_label": [
                STAGE2_LABELS[int(value)]
                for value in y_balanced
            ],
        }
    ).to_csv(
        OUTPUT_Y_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # Distribution after SMOTENC
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("STAGE 2 AFTER SMOTENC")
    print("=" * 80)

    after_counts = (
        pd.Series(y_balanced)
        .value_counts()
        .sort_index()
    )

    for class_id, count in after_counts.items():

        print(
            f"{STAGE2_LABELS[int(class_id)]:30} "
            f"{int(count)}"
        )

    # -----------------------------------------------------
    # Information file
    # -----------------------------------------------------

    info_rows = []

    for class_id in sorted(
        STAGE2_LABELS.keys()
    ):

        before = int(
            before_counts.get(
                class_id,
                0
            )
        )

        after = int(
            after_counts.get(
                class_id,
                0
            )
        )

        info_rows.append(
            {
                "class_id": class_id,
                "class_label": STAGE2_LABELS[
                    class_id
                ],
                "before_count": before,
                "after_count": after,
                "synthetic_added": (
                    after - before
                ),
            }
        )

    pd.DataFrame(
        info_rows
    ).to_csv(
        OUTPUT_INFO_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # Final verification
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("BALANCING COMPLETED SUCCESSFULLY")
    print("=" * 80)

    print(
        "Processed balanced shape:",
        X_balanced_processed.shape
    )

    print(
        "Saved X:",
        OUTPUT_X_FILE
    )

    print(
        "Saved y:",
        OUTPUT_Y_FILE
    )

    print(
        "Saved balancing information:",
        OUTPUT_INFO_FILE
    )

    print(
        "Saved Stage 1 weights:",
        OUTPUT_WEIGHT_FILE
    )

    print()
    print("IMPORTANT:")
    print("- Only TRAINING data was used.")
    print("- TEST data was never used.")
    print("- SMOTENC was applied before one-hot preprocessing.")
    print("- Existing fitted preprocessor was not refitted.")
    print("- Patient IDs were not used to generate synthetic samples.")
    print("- Original raw dataset was not modified.")
    print("- No model was trained.")
    print("- No performance claim is made at this stage.")


if __name__ == "__main__":
    main()