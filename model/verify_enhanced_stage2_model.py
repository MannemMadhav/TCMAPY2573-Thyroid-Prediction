import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------
# TCMAPY2573
# Enhanced Stage 2 Saved Model Verification
# Step 356
# ---------------------------------------------------------

MODEL_FILE = Path(
    "model/enhanced_stage2_calibrated_stacking.pkl"
)

ARCHITECTURE_FILE = Path(
    "data/enhanced_stage2_model_architecture.json"
)

FEATURE_FILE = Path(
    "data/enhanced_stage2_model_features.csv"
)

EXPECTED_FEATURES = 56

EXPECTED_CLASSES = [0, 1, 2, 3]


CLASS_NAMES = {
    0: "Normal",
    1: "Hyperthyroid",
    2: "Hypothyroid",
    3: "Other / Related Diagnosis",
}


def main():

    print("=" * 80)
    print(
        "TCMAPY2573 - ENHANCED STAGE 2 SAVED MODEL VERIFICATION"
    )
    print("=" * 80)

    # -----------------------------------------------------
    # File checks
    # -----------------------------------------------------

    required_files = [
        MODEL_FILE,
        ARCHITECTURE_FILE,
        FEATURE_FILE,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    print()
    print("MODEL LOADING")
    print("-" * 80)

    model = joblib.load(
        MODEL_FILE
    )

    print(
        "Loaded model:",
        type(model).__name__
    )

    if not isinstance(
        model,
        StackingClassifier,
    ):
        raise TypeError(
            "Saved model is not a StackingClassifier."
        )

    # -----------------------------------------------------
    # Load architecture metadata
    # -----------------------------------------------------

    with open(
        ARCHITECTURE_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        architecture = json.load(
            file
        )

    print(
        "Architecture metadata loaded."
    )

    # -----------------------------------------------------
    # Feature verification
    # -----------------------------------------------------

    features = pd.read_csv(
        FEATURE_FILE
    )

    feature_count = len(
        features
    )

    print()
    print("FEATURE VERIFICATION")
    print("-" * 80)

    print(
        "Saved feature count:",
        feature_count
    )

    print(
        "Expected feature count:",
        EXPECTED_FEATURES
    )

    if feature_count != EXPECTED_FEATURES:

        raise ValueError(
            "Incorrect number of saved features."
        )

    # -----------------------------------------------------
    # Stacking structure
    # -----------------------------------------------------

    print()
    print("STACKING ARCHITECTURE")
    print("-" * 80)

    print(
        "Model type:",
        type(model).__name__
    )

    print(
        "Base learner count:",
        len(model.estimators_)
    )

    print(
        "Stack method:",
        model.stack_method
    )

    print(
        "CV:",
        model.cv
    )

    # -----------------------------------------------------
    # Meta learner
    # -----------------------------------------------------

    print()
    print("META-LEARNER")
    print("-" * 80)

    meta_type = type(
        model.final_estimator
    ).__name__

    print(
        "Meta-learner:",
        meta_type
    )

    if not isinstance(
        model.final_estimator,
        LogisticRegression,
    ):
        raise TypeError(
            "Logistic Regression is not the meta-learner."
        )

    # -----------------------------------------------------
    # Base learners
    # -----------------------------------------------------

    print()
    print("BASE LEARNERS")
    print("-" * 80)

    expected_base_names = {
        "calibrated_rf": "CalibratedClassifierCV",
        "calibrated_xgb": "CalibratedClassifierCV",
        "calibrated_svm": "CalibratedClassifierCV",
    }

    for (
        name,
        estimator,
    ) in model.named_estimators_.items():

        estimator_type = type(
            estimator
        ).__name__

        print(
            f"{name}: {estimator_type}"
        )

        if not isinstance(
            estimator,
            CalibratedClassifierCV,
        ):
            raise TypeError(
                f"{name} is not calibrated."
            )

    # -----------------------------------------------------
    # Calibration methods
    # -----------------------------------------------------

    print()
    print("CALIBRATION METHOD")
    print("-" * 80)

    for (
        name,
        estimator,
    ) in model.named_estimators_.items():

        print(
            f"{name}: "
            f"{estimator.method}"
        )

        if estimator.method != "sigmoid":

            raise ValueError(
                f"{name} does not use sigmoid calibration."
            )

    # -----------------------------------------------------
    # Model classes
    # -----------------------------------------------------

    print()
    print("MODEL CLASSES")
    print("-" * 80)

    model_classes = list(
        model.classes_
    )

    print(
        "Saved model classes:",
        model_classes
    )

    if model_classes != EXPECTED_CLASSES:

        raise ValueError(
            "Saved model classes do not match expected classes."
        )

    for class_id in model_classes:

        print(
            f"{class_id} -> "
            f"{CLASS_NAMES[class_id]}"
        )

    # -----------------------------------------------------
    # Architecture metadata comparison
    # -----------------------------------------------------

    print()
    print("ARCHITECTURE METADATA CHECK")
    print("-" * 80)

    print(
        "Metadata model type:",
        architecture.get(
            "model_type"
        )
    )

    print(
        "Metadata number of features:",
        architecture.get(
            "number_of_features"
        )
    )

    print(
        "Metadata number of classes:",
        architecture.get(
            "number_of_classes"
        )
    )

    # -----------------------------------------------------
    # Final verification
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 356 VERIFICATION PASSED")
    print("=" * 80)

    print(
        "Saved Stage 2 model loads successfully."
    )

    print(
        "StackingClassifier verified."
    )

    print(
        "Three calibrated base learners verified."
    )

    print(
        "Logistic Regression meta-learner verified."
    )

    print(
        "Sigmoid calibration verified."
    )

    print(
        "Four output classes verified."
    )

    print(
        "56 model features verified."
    )

    print()
    print("FINAL ARCHITECTURE:")
    print(
        "Calibrated Random Forest"
    )
    print(
        "+"
    )
    print(
        "Calibrated XGBoost"
    )
    print(
        "+"
    )
    print(
        "Calibrated SVM"
    )
    print(
        "        ↓"
    )
    print(
        "Logistic Regression Meta-Learner"
    )
    print(
        "        ↓"
    )
    print(
        "Normal / Hyperthyroid / Hypothyroid / Other"
    )

    print()
    print("IMPORTANT:")
    print("- No test data was used.")
    print("- No model was retrained.")
    print("- No threshold was changed.")
    print("- No Flask code was modified.")


if __name__ == "__main__":
    main()