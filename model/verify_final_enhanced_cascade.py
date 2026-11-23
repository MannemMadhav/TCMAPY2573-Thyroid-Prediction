import sys
from pathlib import Path

import joblib

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------
# TCMAPY2573
# FINAL ENHANCED CASCADE VERIFICATION
# Step 361
# ---------------------------------------------------------

MODEL_FILE = Path(
    "model/final_enhanced_hierarchical_cascade.pkl"
)


def main():

    print("=" * 80)
    print(
        "TCMAPY2573 - FINAL ENHANCED CASCADE VERIFICATION"
    )
    print("=" * 80)

    # -----------------------------------------------------
    # Make model directory available for pickle imports
    # -----------------------------------------------------

    model_directory = str(
        Path("model").resolve()
    )

    if model_directory not in sys.path:

        sys.path.insert(
            0,
            model_directory
        )

    # -----------------------------------------------------
    # File check
    # -----------------------------------------------------

    print()
    print("FILE CHECK")
    print("-" * 80)

    print(
        "Model file:",
        MODEL_FILE
    )

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"Final cascade not found: {MODEL_FILE}"
        )

    print(
        "File exists: True"
    )

    # -----------------------------------------------------
    # Load final cascade
    # -----------------------------------------------------

    print()
    print("CASCADE LOADING")
    print("-" * 80)

    cascade = joblib.load(
        MODEL_FILE
    )

    print(
        "Loaded object:",
        type(cascade).__name__
    )

    # -----------------------------------------------------
    # Verify cascade structure
    # -----------------------------------------------------

    print()
    print("CASCADE STRUCTURE")
    print("-" * 80)

    print(
        "Stage 1 type:",
        type(
            cascade.stage1_model
        ).__name__
    )

    print(
        "Stage 2 type:",
        type(
            cascade.stage2_model
        ).__name__
    )

    print(
        "Confidence threshold:",
        cascade.confidence_threshold
    )

    if type(cascade.stage1_model).__name__ != (
        "CalibratedClassifierCV"
    ):

        raise TypeError(
            "Stage 1 is not a CalibratedClassifierCV."
        )

    if not isinstance(
        cascade.stage2_model,
        StackingClassifier,
    ):

        raise TypeError(
            "Stage 2 is not a StackingClassifier."
        )

    if (
        cascade.confidence_threshold
        != 0.95
    ):

        raise ValueError(
            "Confidence threshold is not 0.95."
        )

    # -----------------------------------------------------
    # Verify Stage 2 meta learner
    # -----------------------------------------------------

    print()
    print("STAGE 2 META-LEARNER")
    print("-" * 80)

    meta_learner = (
        cascade.stage2_model.final_estimator
    )

    print(
        "Meta-learner:",
        type(
            meta_learner
        ).__name__
    )

    if not isinstance(
        meta_learner,
        LogisticRegression,
    ):

        raise TypeError(
            "Stage 2 meta-learner is not LogisticRegression."
        )

    # -----------------------------------------------------
    # Verify base learners
    # -----------------------------------------------------

    print()
    print("STAGE 2 BASE LEARNERS")
    print("-" * 80)

    expected_names = [
        "calibrated_rf",
        "calibrated_xgb",
        "calibrated_svm",
    ]

    actual_names = list(
        cascade.stage2_model.named_estimators_.keys()
    )

    print(
        "Base learner names:",
        actual_names
    )

    if actual_names != expected_names:

        raise ValueError(
            "Base learner names do not match expected architecture."
        )

    for name, estimator in (
        cascade.stage2_model.named_estimators_.items()
    ):

        print(
            f"{name}: "
            f"{type(estimator).__name__}"
        )

        if not isinstance(
            estimator,
            CalibratedClassifierCV,
        ):

            raise TypeError(
                f"{name} is not calibrated."
            )

        print(
            f"  calibration method: "
            f"{estimator.method}"
        )

        if estimator.method != "sigmoid":

            raise ValueError(
                f"{name} does not use sigmoid calibration."
            )

    # -----------------------------------------------------
    # Verify stacking configuration
    # -----------------------------------------------------

    print()
    print("STACKING CONFIGURATION")
    print("-" * 80)

    print(
        "Stack method:",
        cascade.stage2_model.stack_method
    )

    print(
        "Cross-validation folds:",
        cascade.stage2_model.cv
    )

    if (
        cascade.stage2_model.stack_method
        != "predict_proba"
    ):

        raise ValueError(
            "Stacking does not use predict_proba."
        )

    if cascade.stage2_model.cv != 3:

        raise ValueError(
            "Expected 3-fold stacking CV."
        )

    # -----------------------------------------------------
    # Verify Stage 1 probability capability
    # -----------------------------------------------------

    print()
    print("STAGE 1 PROBABILITY CHECK")
    print("-" * 80)

    print(
        "Has predict_proba:",
        hasattr(
            cascade.stage1_model,
            "predict_proba",
        )
    )

    if not hasattr(
        cascade.stage1_model,
        "predict_proba",
    ):

        raise TypeError(
            "Stage 1 does not provide probability estimates."
        )

    # -----------------------------------------------------
    # Verify Stage 2 probability capability
    # -----------------------------------------------------

    print()
    print("STAGE 2 PROBABILITY CHECK")
    print("-" * 80)

    print(
        "Has predict_proba:",
        hasattr(
            cascade.stage2_model,
            "predict_proba",
        )
    )

    if not hasattr(
        cascade.stage2_model,
        "predict_proba",
    ):

        raise TypeError(
            "Stage 2 does not provide probability estimates."
        )

    # -----------------------------------------------------
    # Final verification
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 361 VERIFICATION PASSED")
    print("=" * 80)

    print(
        "Final enhanced cascade loads successfully."
    )

    print(
        "Calibrated Stage 1 verified."
    )

    print(
        "Confidence threshold 0.95 verified."
    )

    print(
        "Stage 2 StackingClassifier verified."
    )

    print(
        "Calibrated Random Forest verified."
    )

    print(
        "Calibrated XGBoost verified."
    )

    print(
        "Calibrated SVM verified."
    )

    print(
        "Logistic Regression meta-learner verified."
    )

    print(
        "predict_proba stacking verified."
    )

    print()
    print("FINAL LOCKED ARCHITECTURE:")
    print(
        "Calibrated Stage 1 XGBoost"
    )
    print(
        "        ↓"
    )
    print(
        "Confidence Router (0.95)"
    )
    print(
        "        ↓"
    )
    print(
        "Calibrated RF + Calibrated XGBoost + Calibrated SVM"
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
    print("- No training performed.")
    print("- No test data used.")
    print("- No threshold changed.")
    print("- No model modified.")
    print("- Flask not modified.")


if __name__ == "__main__":
    main()