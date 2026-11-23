from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# TCMAPY2573
# ROUTER / STAGE-2 PREDICTION CONSISTENCY VERIFICATION
# Step 365
# ---------------------------------------------------------

X_TEST_FILE = Path(
    "data/enhanced_X_test_processed.csv"
)

ROUTING_FILE = Path(
    "data/enhanced_cascade_routing_analysis.csv"
)

STAGE2_MODEL_FILE = Path(
    "model/enhanced_stage2_calibrated_stacking.pkl"
)

CASCADE_MODEL_FILE = Path(
    "model/final_enhanced_hierarchical_cascade.pkl"
)

EXPECTED_ROWS = 1835
EXPECTED_FEATURES = 56


def main():

    print("=" * 80)
    print(
        "TCMAPY2573 - ROUTER PREDICTION CONSISTENCY CHECK"
    )
    print("=" * 80)

    # -----------------------------------------------------
    # File checks
    # -----------------------------------------------------

    print()
    print("FILE CHECK")
    print("-" * 80)

    required_files = [
        X_TEST_FILE,
        ROUTING_FILE,
        STAGE2_MODEL_FILE,
        CASCADE_MODEL_FILE,
    ]

    for path in required_files:

        print(
            f"{path}: {path.exists()}"
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    # -----------------------------------------------------
    # Load data and models
    # -----------------------------------------------------

    X_test = pd.read_csv(
        X_TEST_FILE
    )

    routing = pd.read_csv(
        ROUTING_FILE
    )

    stage2 = joblib.load(
        STAGE2_MODEL_FILE
    )

    cascade = joblib.load(
        CASCADE_MODEL_FILE
    )

    if X_test.shape != (
        EXPECTED_ROWS,
        EXPECTED_FEATURES,
    ):

        raise ValueError(
            f"Unexpected test shape: {X_test.shape}"
        )

    if len(routing) != EXPECTED_ROWS:

        raise ValueError(
            f"Unexpected routing rows: {len(routing)}"
        )

    print()
    print("DATA VALIDATION")
    print("-" * 80)

    print(
        "Test shape:",
        X_test.shape
    )

    print(
        "Routing rows:",
        len(routing)
    )

    # -----------------------------------------------------
    # Compute predictions
    # -----------------------------------------------------

    print()
    print("COMPUTING PREDICTIONS")
    print("-" * 80)

    stage2_predictions = (
        stage2.predict(
            X_test
        )
        .astype(int)
    )

    cascade_predictions = (
        cascade.predict(
            X_test
        )
        .astype(int)
    )

    print(
        "Stage 2 predictions completed."
    )

    print(
        "Cascade predictions completed."
    )

    # -----------------------------------------------------
    # Identify routes
    # -----------------------------------------------------

    if "routing_decision" not in routing.columns:

        raise ValueError(
            "routing_decision column not found."
        )

    direct_normal_mask = (
        routing[
            "routing_decision"
        ]
        .astype(str)
        .eq("FINAL_NORMAL")
        .to_numpy()
    )

    stage2_route_mask = (
        ~direct_normal_mask
    )

    direct_normal_count = int(
        direct_normal_mask.sum()
    )

    stage2_count = int(
        stage2_route_mask.sum()
    )

    print()
    print("ROUTING COUNTS")
    print("-" * 80)

    print(
        "Direct Final Normal:",
        direct_normal_count
    )

    print(
        "Stage 2 routed:",
        stage2_count
    )

    # -----------------------------------------------------
    # Direct normal verification
    # -----------------------------------------------------

    direct_predictions = (
        cascade_predictions[
            direct_normal_mask
        ]
    )

    direct_normal_errors = int(
        (
            direct_predictions
            != 0
        ).sum()
    )

    print()
    print("DIRECT NORMAL ROUTE CHECK")
    print("-" * 80)

    print(
        "Direct Normal cases:",
        direct_normal_count
    )

    print(
        "Cases not finalized as Normal:",
        direct_normal_errors
    )

    if direct_normal_errors != 0:

        raise ValueError(
            "Some FINAL_NORMAL routes did not "
            "produce class 0."
        )

    # -----------------------------------------------------
    # Stage 2 route consistency
    # -----------------------------------------------------

    routed_stage2_predictions = (
        stage2_predictions[
            stage2_route_mask
        ]
    )

    routed_cascade_predictions = (
        cascade_predictions[
            stage2_route_mask
        ]
    )

    stage2_mismatch_mask = (
        routed_stage2_predictions
        != routed_cascade_predictions
    )

    stage2_mismatch_count = int(
        stage2_mismatch_mask.sum()
    )

    stage2_match_count = (
        stage2_count
        - stage2_mismatch_count
    )

    stage2_consistency = (
        stage2_match_count
        / stage2_count
        if stage2_count > 0
        else 1.0
    )

    print()
    print("STAGE 2 ROUTE CONSISTENCY")
    print("-" * 80)

    print(
        "Stage 2 routed cases:",
        stage2_count
    )

    print(
        "Matching Stage 2 predictions:",
        stage2_match_count
    )

    print(
        "Mismatching predictions:",
        stage2_mismatch_count
    )

    print(
        f"Stage 2 route consistency: "
        f"{stage2_consistency:.6f}"
    )

    if stage2_mismatch_count > 0:

        print()
        print("FIRST MISMATCHES")
        print("-" * 80)

        mismatch_indices = np.where(
            stage2_route_mask
            & (
                stage2_predictions
                != cascade_predictions
            )
        )[0]

        for index in mismatch_indices[:20]:

            print(
                f"Row {index}: "
                f"Stage2={stage2_predictions[index]} "
                f"Cascade={cascade_predictions[index]}"
            )

        raise ValueError(
            "Stage 2 routed predictions do not "
            "match cascade predictions."
        )

    # -----------------------------------------------------
    # Global consistency
    # -----------------------------------------------------

    global_mismatch_count = int(
        (
            stage2_predictions
            != cascade_predictions
        ).sum()
    )

    global_match_count = (
        EXPECTED_ROWS
        - global_mismatch_count
    )

    global_consistency = (
        global_match_count
        / EXPECTED_ROWS
    )

    print()
    print("GLOBAL PREDICTION CONSISTENCY")
    print("-" * 80)

    print(
        "Total predictions:",
        EXPECTED_ROWS
    )

    print(
        "Matching:",
        global_match_count
    )

    print(
        "Mismatching:",
        global_mismatch_count
    )

    print(
        f"Global consistency: "
        f"{global_consistency:.6f}"
    )

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print(
        "STEP 365 VERIFICATION PASSED"
    )
    print("=" * 80)

    print(
        "All direct-normal routes produced Normal."
    )

    print(
        "All Stage 2 routed predictions match "
        "the final cascade."
    )

    print(
        "Router prediction consistency verified."
    )

    print()
    print("IMPORTANT:")
    print("- No model was retrained.")
    print("- No model was modified.")
    print("- No threshold was changed.")
    print("- No test-set tuning was performed.")
    print("- This step verifies routing correctness only.")


if __name__ == "__main__":
    main()