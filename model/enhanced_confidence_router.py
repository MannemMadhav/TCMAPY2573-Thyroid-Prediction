from dataclasses import dataclass


# ---------------------------------------------------------
# TCMAPY2573
# Confidence-Guided Stage 1 Router
# Step 352
# ---------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.95


@dataclass
class RoutingResult:
    """
    Result produced by the confidence-guided router.
    """

    probability_non_normal: float
    probability_normal: float
    confidence: float
    stage1_prediction: int
    stage1_label: str
    routing_decision: str


class EnhancedConfidenceRouter:
    """
    Confidence-guided routing logic.

    Stage 1 classes:

        0 -> Normal
        1 -> Non-Normal

    Routing rule:

        P(Non-Normal) <= 0.05
            -> confident Normal

        0.05 < P(Non-Normal) < 0.95
            -> uncertain, send to Stage 2

        P(Non-Normal) >= 0.95
            -> confident Non-Normal, send to Stage 2
    """

    def __init__(
        self,
        threshold: float = CONFIDENCE_THRESHOLD,
    ):
        if not 0.50 <= threshold < 1.0:
            raise ValueError(
                "Threshold must be >= 0.50 and < 1.00."
            )

        self.threshold = float(threshold)

        self.lower_normal_limit = (
            1.0 - self.threshold
        )

    def route(
        self,
        probability_non_normal: float,
    ) -> RoutingResult:
        """
        Convert Stage 1 calibrated probability
        into a confidence-guided routing decision.
        """

        probability_non_normal = float(
            probability_non_normal
        )

        if not 0.0 <= probability_non_normal <= 1.0:
            raise ValueError(
                "Probability must be between 0 and 1."
            )

        probability_normal = (
            1.0 - probability_non_normal
        )

        # -------------------------------------------------
        # Confident Normal
        # -------------------------------------------------

        if (
            probability_non_normal
            <= self.lower_normal_limit
        ):

            stage1_prediction = 0
            stage1_label = "Normal"
            confidence = probability_normal
            routing_decision = (
                "FINAL_NORMAL"
            )

        # -------------------------------------------------
        # Confident Non-Normal
        # -------------------------------------------------

        elif (
            probability_non_normal
            >= self.threshold
        ):

            stage1_prediction = 1
            stage1_label = "Non-Normal"
            confidence = probability_non_normal
            routing_decision = (
                "SEND_TO_STAGE_2"
            )

        # -------------------------------------------------
        # Uncertain
        # -------------------------------------------------

        else:

            stage1_prediction = int(
                probability_non_normal >= 0.50
            )

            stage1_label = (
                "Non-Normal"
                if stage1_prediction == 1
                else "Normal"
            )

            confidence = max(
                probability_normal,
                probability_non_normal,
            )

            routing_decision = (
                "UNCERTAIN_SEND_TO_STAGE_2"
            )

        return RoutingResult(
            probability_non_normal=(
                probability_non_normal
            ),
            probability_normal=(
                probability_normal
            ),
            confidence=confidence,
            stage1_prediction=(
                stage1_prediction
            ),
            stage1_label=stage1_label,
            routing_decision=(
                routing_decision
            ),
        )


if __name__ == "__main__":

    router = EnhancedConfidenceRouter()

    examples = [
        0.01,
        0.04,
        0.05,
        0.30,
        0.50,
        0.80,
        0.95,
        0.98,
    ]

    print("=" * 80)
    print(
        "TCMAPY2573 - CONFIDENCE ROUTER TEST"
    )
    print("=" * 80)

    print(
        f"Confidence threshold: "
        f"{router.threshold}"
    )

    print(
        f"Normal upper probability: "
        f"{router.lower_normal_limit}"
    )

    print()

    for probability in examples:

        result = router.route(
            probability
        )

        print(
            f"P(Non-Normal) = "
            f"{probability:.2f} | "
            f"Confidence = "
            f"{result.confidence:.2f} | "
            f"{result.routing_decision}"
        )

    print()
    print("=" * 80)
    print("ROUTER TEST COMPLETED")
    print("=" * 80)