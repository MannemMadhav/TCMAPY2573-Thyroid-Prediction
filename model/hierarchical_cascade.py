from pathlib import Path
import joblib
import pandas as pd


# ============================================================
# Project paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "model"


# ============================================================
# Model paths
# ============================================================

STAGE1_MODEL_PATH = (
    MODEL_DIR / "hierarchical_stage1_xgboost.pkl"
)

STAGE2_MODEL_PATH = (
    MODEL_DIR / "final_calibrated_stacking_model.pkl"
)


# ============================================================
# Load models
# ============================================================

stage1_model = joblib.load(
    STAGE1_MODEL_PATH
)

stage2_model = joblib.load(
    STAGE2_MODEL_PATH
)


# ============================================================
# Class mappings
# ============================================================

STAGE1_CLASSES = {
    0: "normal",
    1: "thyroid_dysfunction"
}


STAGE2_CLASSES = {
    1: "normal",
    2: "hyperfunction",
    3: "subnormal"
}


# ============================================================
# Hierarchical Cascade
# ============================================================

class HierarchicalThyroidCascade:

    def __init__(
        self,
        stage1_model,
        stage2_model
    ):
        self.stage1_model = stage1_model
        self.stage2_model = stage2_model

    # --------------------------------------------------------
    # Stage 1
    # --------------------------------------------------------

    def predict_stage1(self, X):

        prediction = int(
            self.stage1_model.predict(X)[0]
        )

        probability = self.stage1_model.predict_proba(X)[0]

        return prediction, probability

    # --------------------------------------------------------
    # Stage 2
    # --------------------------------------------------------

    def predict_stage2(self, X):

        prediction = int(
            self.stage2_model.predict(X)[0]
        )

        probability = self.stage2_model.predict_proba(X)[0]

        return prediction, probability

    # --------------------------------------------------------
    # Final hierarchical prediction
    # --------------------------------------------------------

    def predict(self, X):

        # Stage 1 screening
        stage1_prediction, stage1_probability = (
            self.predict_stage1(X)
        )

        # ----------------------------------------------------
        # Stage 1 says normal
        # ----------------------------------------------------

        if stage1_prediction == 0:

            return {
                "stage1_prediction": 0,
                "stage1_class": "normal",
                "stage1_probabilities": stage1_probability,

                "stage2_prediction": None,
                "stage2_class": None,
                "stage2_probabilities": None,

                "final_class": "normal"
            }

        # ----------------------------------------------------
        # Stage 1 says thyroid dysfunction
        # Run Stage 2
        # ----------------------------------------------------

        stage2_prediction, stage2_probability = (
            self.predict_stage2(X)
        )

        stage2_class = STAGE2_CLASSES[
            stage2_prediction
        ]

        return {
            "stage1_prediction": 1,
            "stage1_class": "thyroid_dysfunction",
            "stage1_probabilities": stage1_probability,

            "stage2_prediction": stage2_prediction,
            "stage2_class": stage2_class,
            "stage2_probabilities": stage2_probability,

            "final_class": stage2_class
        }


# ============================================================
# Basic test
# ============================================================

if __name__ == "__main__":

    # Load one test sample from the ANN dataset
    X_test = pd.read_csv(
        BASE_DIR / "data" / "ann_X_test.csv"
    )

    sample = X_test.iloc[[0]]

    cascade = HierarchicalThyroidCascade(
        stage1_model,
        stage2_model
    )

    result = cascade.predict(
        sample
    )

    print("=" * 60)
    print("HIERARCHICAL THYROID CASCADE")
    print("=" * 60)

    print("\nStage 1 prediction:")
    print(
        result["stage1_prediction"]
    )

    print("\nStage 1 class:")
    print(
        result["stage1_class"]
    )

    print("\nFinal class:")
    print(
        result["final_class"]
    )

    if result["stage2_prediction"] is not None:

        print("\nStage 2 prediction:")
        print(
            result["stage2_prediction"]
        )

        print("\nStage 2 class:")
        print(
            result["stage2_class"]
        )

    else:

        print(
            "\nStage 2 was not required."
        )

    print(
        "\nHierarchical cascade test completed successfully!"
    )