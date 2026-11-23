from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------
# TCMAPY2573
# Enhanced Leakage-Safe Preprocessing
# Step 338
# ---------------------------------------------------------

INPUT_FILE = Path("data/thyroid_9172_quality_checked.csv")


# ---------------------------------------------------------
# Feature groups
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

ALL_MODEL_FEATURES = (
    NUMERICAL_FEATURES
    + BINARY_FEATURES
    + CATEGORICAL_FEATURES
)

TARGET_COLUMN = "target_group"
ID_COLUMN = "patient_id"


# ---------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------

def create_preprocessor() -> ColumnTransformer:
    """
    Create the enhanced preprocessing pipeline.

    IMPORTANT:
    This function only CREATES the transformer.
    It does not fit it.

    The transformer must be fitted only on training data.
    """

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    binary_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    categories=[
                        [
                            "f",
                            "t",
                        ]
                    ]
                    * len(BINARY_FEATURES),
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Unknown",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES,
            ),
            (
                "binary",
                binary_pipeline,
                BINARY_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


# ---------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------

def load_quality_checked_data():
    """
    Load the quality-checked dataset.

    Patient ID is kept separately for auditing.
    It is NEVER supplied to the model.
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = ALL_MODEL_FEATURES + [
        TARGET_COLUMN,
        ID_COLUMN,
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    X = df[ALL_MODEL_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()
    patient_ids = df[ID_COLUMN].copy()

    return X, y, patient_ids


# ---------------------------------------------------------
# Direct execution verification
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 80)
    print("TCMAPY2573 - ENHANCED PREPROCESSING")
    print("=" * 80)

    X, y, patient_ids = load_quality_checked_data()

    print(f"Rows: {len(X)}")
    print(f"Input features: {X.shape[1]}")
    print(f"Target classes: {y.nunique()}")
    print(f"Unique patient IDs: {patient_ids.nunique()}")

    preprocessor = create_preprocessor()

    print()
    print("Preprocessor created successfully.")
    print()
    print("Preprocessing design:")
    print("- Numerical: median imputation + scaling")
    print("- Binary f/t: most-frequent imputation + one-hot encoding")
    print("- Categorical: Unknown imputation + one-hot encoding")
    print("- Patient ID: excluded from model features")
    print()
    print("IMPORTANT:")
    print("- Preprocessor is NOT fitted here.")
    print("- No test data is used.")
    print("- No SMOTE is applied.")
    print("- No model is trained.")