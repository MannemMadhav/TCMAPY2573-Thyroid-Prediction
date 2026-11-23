from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# TCMAPY2573
# ENHANCED CONFIDENCE-GUIDED HIERARCHICAL CASCADE
# Step 358
# ---------------------------------------------------------

STAGE1_MODEL_FILE = Path(
    "model/enhanced_stage1_calibrated_model.pkl"
)

STAGE2_MODEL_FILE = Path(
    "model/enhanced_stage2_calibrated_stacking.pkl"
)

CONFIDENCE_THRESHOLD = 0.95


CLASS_NAMES = {
    0: "Normal",
    1: "Hyperthyroid",
    2: "Hypothyroid",
    3: "Other / Related Diagnosis",
}


@dataclass
class EnhancedCascadeResult:

    stage1_probability_normal: float

    stage1_probability_non_normal: float

    stage1_confidence: float

    stage1_label: str

    routing_decision: str

    final_class: int

    final_label: str

    stage2_used: bool

    stage2_probabilities: np.ndarray | None


class EnhancedHierarchicalCascade:

    def __init__(
        self,
        stage1_model=None,
        stage2_model=None,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    ):

        if stage1_model is None:

            if not STAGE1_MODEL_FILE.exists():

                raise FileNotFoundError(
                    f"Stage 1 model not found: "
                    f"{STAGE1_MODEL_FILE}"
                )

            stage1_model = joblib.load(
                STAGE1_MODEL_FILE
            )

        if stage2_model is None:

            if not STAGE2_MODEL_FILE.exists():

                raise FileNotFoundError(
                    f"Stage 2 model not found: "
                    f"{STAGE2_MODEL_FILE}"
                )

            stage2_model = joblib.load(
                STAGE2_MODEL_FILE
            )

        self.stage1_model = stage1_model

        self.stage2_model = stage2_model

        self.confidence_threshold = float(
            confidence_threshold
        )

        self.lower_normal_limit = (
            1.0 - self.confidence_threshold
        )

    def _normalize_input(self, X):
        """
        Normalize cascade inference input while preserving the
        56 trained feature names used by the Stage 2 learners.
        """
        if isinstance(X, pd.DataFrame):
            normalized = X.copy()
        else:
            normalized = np.asarray(X)

            if normalized.ndim == 1:
                normalized = normalized.reshape(1, -1)

            if normalized.ndim != 2:
                raise ValueError(
                    "X must be a 1-D or 2-D numeric feature array."
                )

        feature_names = getattr(
            self.stage2_model,
            "feature_names_in_",
            None,
        )

        if feature_names is None:
            feature_names = getattr(
                self.stage1_model,
                "feature_names_in_",
                None,
            )

        if feature_names is not None:
            feature_names = list(feature_names)

            if normalized.shape[1] == len(feature_names):
                if isinstance(normalized, pd.DataFrame):
                    if (
                        list(normalized.columns) != feature_names
                        and set(normalized.columns) == set(feature_names)
                    ):
                        normalized = normalized.loc[:, feature_names]
                else:
                    normalized = pd.DataFrame(
                        normalized,
                        columns=feature_names,
                    )

        return normalized

    def predict_one(
        self,
        X,
    ):

        X = self._normalize_input(X)


        # -------------------------------------------------
        # Stage 1 calibrated probability
        # -------------------------------------------------

        stage1_probabilities = (
            self.stage1_model.predict_proba(
                X
            )
        )

        probability_non_normal = float(
            stage1_probabilities[
                0,
                1
            ]
        )

        probability_normal = (
            1.0
            - probability_non_normal
        )

        # -------------------------------------------------
        # Confidence-guided routing
        # -------------------------------------------------

        if (
            probability_non_normal
            <= self.lower_normal_limit
        ):

            confidence = (
                probability_normal
            )

            stage1_label = "Normal"

            routing_decision = (
                "FINAL_NORMAL"
            )

            final_class = 0

            final_label = (
                CLASS_NAMES[
                    final_class
                ]
            )

            return EnhancedCascadeResult(
                stage1_probability_normal=(
                    probability_normal
                ),
                stage1_probability_non_normal=(
                    probability_non_normal
                ),
                stage1_confidence=confidence,
                stage1_label=stage1_label,
                routing_decision=routing_decision,
                final_class=final_class,
                final_label=final_label,
                stage2_used=False,
                stage2_probabilities=None,
            )

        # -------------------------------------------------
        # Non-normal or uncertain case
        # -------------------------------------------------

        confidence = max(
            probability_normal,
            probability_non_normal,
        )

        if (
            probability_non_normal
            >= self.confidence_threshold
        ):

            routing_decision = (
                "CONFIDENT_NON_NORMAL_TO_STAGE_2"
            )

        else:

            routing_decision = (
                "UNCERTAIN_TO_STAGE_2"
            )

        # -------------------------------------------------
        # Stage 2 detailed diagnosis
        # -------------------------------------------------

        stage2_probabilities = (
            self.stage2_model.predict_proba(
                X
            )[0]
        )

        final_class = int(
            self.stage2_model.predict(
                X
            )[0]
        )

        final_label = (
            CLASS_NAMES[
                final_class
            ]
        )

        return EnhancedCascadeResult(
            stage1_probability_normal=(
                probability_normal
            ),
            stage1_probability_non_normal=(
                probability_non_normal
            ),
            stage1_confidence=confidence,
            stage1_label=stage1_label_from_probability(
                probability_non_normal
            ),
            routing_decision=routing_decision,
            final_class=final_class,
            final_label=final_label,
            stage2_used=True,
            stage2_probabilities=(
                stage2_probabilities
            ),
        )

    def predict(
        self,
        X,
    ):

        X = self._normalize_input(X)


        stage1_probabilities = (
            self.stage1_model.predict_proba(
                X
            )[:, 1]
        )

        normal_probability = (
            1.0
            - stage1_probabilities
        )

        confident_normal = (
            stage1_probabilities
            <= self.lower_normal_limit
        )

        final_predictions = np.zeros(
            len(X),
            dtype=int,
        )

        stage2_mask = (
            ~confident_normal
        )

        if stage2_mask.any():

            final_predictions[
                stage2_mask
            ] = self.stage2_model.predict(
                X[
                    stage2_mask
                ]
            )

        return final_predictions

    def predict_proba(
        self,
        X,
    ):

        X = self._normalize_input(X)


        final_predictions = (
            self.predict(X)
        )

        probabilities = np.zeros(
            (
                len(X),
                4,
            ),
            dtype=float,
        )

        stage1_probabilities = (
            self.stage1_model.predict_proba(
                X
            )[:, 1]
        )

        stage1_normal = (
            1.0
            - stage1_probabilities
        )

        normal_mask = (
            stage1_probabilities
            <= self.lower_normal_limit
        )

        probabilities[
            normal_mask,
            0
        ] = 1.0

        stage2_mask = (
            ~normal_mask
        )

        if stage2_mask.any():

            stage2_probabilities = (
                self.stage2_model.predict_proba(
                    X[
                        stage2_mask
                    ]
                )
            )

            probabilities[
                stage2_mask
            ] = stage2_probabilities

        return probabilities


def stage1_label_from_probability(
    probability_non_normal,
):

    if (
        probability_non_normal
        >= 0.50
    ):

        return "Non-Normal"

    return "Normal"


def main():

    print("=" * 80)
    print(
        "TCMAPY2573 - ENHANCED HIERARCHICAL CASCADE"
    )
    print("=" * 80)

    cascade = EnhancedHierarchicalCascade()

    print()
    print("MODELS LOADED")
    print("-" * 80)

    print(
        "Stage 1:",
        type(
            cascade.stage1_model
        ).__name__,
    )

    print(
        "Stage 2:",
        type(
            cascade.stage2_model
        ).__name__,
    )

    print(
        "Confidence threshold:",
        cascade.confidence_threshold,
    )

    print(
        "Normal upper probability:",
        cascade.lower_normal_limit,
    )

    print()
    print("=" * 80)
    print("STEP 358 COMPLETED")
    print("=" * 80)

    print(
        "Enhanced cascade created successfully."
    )

    print()
    print("ROUTING:")
    print(
        "P(Non-Normal) <= 0.05"
    )
    print(
        "    -> Final Normal"
    )

    print(
        "P(Non-Normal) > 0.05"
    )
    print(
        "    -> Stage 2"
    )

    print()
    print("IMPORTANT:")
    print("- No test data was used.")
    print("- No models were retrained.")
    print("- Confidence threshold remains 0.95.")
    print("- Existing baseline cascade was not modified.")
    print("- Flask code was not modified.")


if __name__ == "__main__":
    main()
