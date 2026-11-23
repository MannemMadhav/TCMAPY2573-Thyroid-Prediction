from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# TCMAPY2573
# ENHANCED CASCADE COMPUTATIONAL EFFICIENCY ANALYSIS
# Step 363
# ---------------------------------------------------------

STAGE1_MODEL_FILE = Path(
    "model/enhanced_stage1_calibrated_model.pkl"
)

STAGE2_MODEL_FILE = Path(
    "model/enhanced_stage2_calibrated_stacking.pkl"
)

FINAL_CASCADE_FILE = Path(
    "model/final_enhanced_hierarchical_cascade.pkl"
)

X_TEST_FILE = Path(
    "data/enhanced_X_test_processed.csv"
)

ROUTING_FILE = Path(
    "data/enhanced_cascade_routing_analysis.csv"
)

OUTPUT_FILE = Path(
    "data/enhanced_cascade_efficiency_results.csv"
)

EXPECTED_ROWS = 1835
EXPECTED_FEATURES = 56

WARMUP_RUNS = 2
TIMING_RUNS = 5


def file_size_mb(path):

    return path.stat().st_size / (
        1024 * 1024
    )


def measure_predict_time(
    model,
    X,
    runs=TIMING_RUNS,
):

    # Warm-up
    for _ in range(WARMUP_RUNS):

        model.predict(X)

    timings = []

    for _ in range(runs):

        start = time.perf_counter()

        model.predict(X)

        end = time.perf_counter()

        timings.append(
            end - start
        )

    return {
        "mean_seconds": float(
            np.mean(timings)
        ),
        "minimum_seconds": float(
            np.min(timings)
        ),
        "maximum_seconds": float(
            np.max(timings)
        ),
    }


def main():

    print("=" * 80)
    print(
        "TCMAPY2573 - ENHANCED CASCADE EFFICIENCY ANALYSIS"
    )
    print("=" * 80)

    # -----------------------------------------------------
    # File checks
    # -----------------------------------------------------

    print()
    print("FILE CHECK")
    print("-" * 80)

    required_files = [
        STAGE1_MODEL_FILE,
        STAGE2_MODEL_FILE,
        FINAL_CASCADE_FILE,
        X_TEST_FILE,
        ROUTING_FILE,
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
    # Model loading
    # -----------------------------------------------------

    print()
    print("MODEL LOADING")
    print("-" * 80)

    stage1 = joblib.load(
        STAGE1_MODEL_FILE
    )

    stage2 = joblib.load(
        STAGE2_MODEL_FILE
    )

    cascade = joblib.load(
        FINAL_CASCADE_FILE
    )

    print(
        "Stage 1:",
        type(stage1).__name__
    )

    print(
        "Stage 2:",
        type(stage2).__name__
    )

    print(
        "Final cascade:",
        type(cascade).__name__
    )

    # -----------------------------------------------------
    # Test data
    # -----------------------------------------------------

    X_test = pd.read_csv(
        X_TEST_FILE
    )

    if X_test.shape != (
        EXPECTED_ROWS,
        EXPECTED_FEATURES,
    ):

        raise ValueError(
            f"Unexpected test shape: {X_test.shape}"
        )

    print()
    print("DATASET")
    print("-" * 80)

    print(
        "Test rows:",
        len(X_test)
    )

    print(
        "Features:",
        X_test.shape[1]
    )

    # -----------------------------------------------------
    # Routing analysis
    # -----------------------------------------------------

    routing = pd.read_csv(
        ROUTING_FILE
    )

    routing_count = int(
        (
            routing[
                "routing_decision"
            ]
            == "FINAL_NORMAL"
        ).sum()
    )

    stage2_count = int(
        (
            routing[
                "routing_decision"
            ]
            != "FINAL_NORMAL"
        ).sum()
    )

    stage2_rate = (
        stage2_count
        / EXPECTED_ROWS
        * 100
    )

    direct_normal_rate = (
        routing_count
        / EXPECTED_ROWS
        * 100
    )

    print()
    print("CONFIDENCE ROUTING")
    print("-" * 80)

    print(
        "Total cases:",
        EXPECTED_ROWS
    )

    print(
        "Direct Final Normal:",
        routing_count
    )

    print(
        "Stage 2 used:",
        stage2_count
    )

    print(
        f"Direct Final Normal rate: "
        f"{direct_normal_rate:.2f}%"
    )

    print(
        f"Stage 2 routing rate: "
        f"{stage2_rate:.2f}%"
    )

    # -----------------------------------------------------
    # Model sizes
    # -----------------------------------------------------

    stage1_size = file_size_mb(
        STAGE1_MODEL_FILE
    )

    stage2_size = file_size_mb(
        STAGE2_MODEL_FILE
    )

    cascade_size = file_size_mb(
        FINAL_CASCADE_FILE
    )

    print()
    print("MODEL STORAGE")
    print("-" * 80)

    print(
        f"Stage 1 model size: "
        f"{stage1_size:.4f} MB"
    )

    print(
        f"Stage 2 model size: "
        f"{stage2_size:.4f} MB"
    )

    print(
        f"Final cascade size: "
        f"{cascade_size:.4f} MB"
    )

    # -----------------------------------------------------
    # Stage 1 timing
    # -----------------------------------------------------

    print()
    print("STAGE 1 INFERENCE TIMING")
    print("-" * 80)

    stage1_timing = measure_predict_time(
        stage1,
        X_test,
    )

    print(
        f"Mean: "
        f"{stage1_timing['mean_seconds']:.6f} seconds"
    )

    print(
        f"Minimum: "
        f"{stage1_timing['minimum_seconds']:.6f} seconds"
    )

    print(
        f"Maximum: "
        f"{stage1_timing['maximum_seconds']:.6f} seconds"
    )

    # -----------------------------------------------------
    # Stage 2 timing
    # -----------------------------------------------------

    print()
    print("STAGE 2 INFERENCE TIMING")
    print("-" * 80)

    stage2_input = X_test.iloc[
        routing[
            "routing_decision"
        ].to_numpy()
        != "FINAL_NORMAL"
    ].reset_index(drop=True)

    if len(stage2_input) != stage2_count:

        raise ValueError(
            "Stage 2 routing count mismatch."
        )

    stage2_timing = measure_predict_time(
        stage2,
        stage2_input,
    )

    print(
        "Stage 2 timing batch:",
        len(stage2_input)
    )

    print(
        f"Mean: "
        f"{stage2_timing['mean_seconds']:.6f} seconds"
    )

    print(
        f"Minimum: "
        f"{stage2_timing['minimum_seconds']:.6f} seconds"
    )

    print(
        f"Maximum: "
        f"{stage2_timing['maximum_seconds']:.6f} seconds"
    )

    # -----------------------------------------------------
    # Complete cascade timing
    # -----------------------------------------------------

    print()
    print("COMPLETE CASCADE INFERENCE TIMING")
    print("-" * 80)

    cascade_timing = measure_predict_time(
        cascade,
        X_test,
    )

    print(
        f"Mean: "
        f"{cascade_timing['mean_seconds']:.6f} seconds"
    )

    print(
        f"Minimum: "
        f"{cascade_timing['minimum_seconds']:.6f} seconds"
    )

    print(
        f"Maximum: "
        f"{cascade_timing['maximum_seconds']:.6f} seconds"
    )

    # -----------------------------------------------------
    # Per-patient timing
    # -----------------------------------------------------

    cascade_mean_per_patient = (
        cascade_timing[
            "mean_seconds"
        ]
        / EXPECTED_ROWS
    )

    stage1_mean_per_patient = (
        stage1_timing[
            "mean_seconds"
        ]
        / EXPECTED_ROWS
    )

    stage2_mean_per_routed_patient = (
        stage2_timing[
            "mean_seconds"
        ]
        / stage2_count
        if stage2_count > 0
        else 0.0
    )

    print()
    print("AVERAGE PER-PATIENT TIMING")
    print("-" * 80)

    print(
        f"Stage 1: "
        f"{stage1_mean_per_patient:.9f} seconds"
    )

    print(
        f"Stage 2 routed case: "
        f"{stage2_mean_per_routed_patient:.9f} seconds"
    )

    print(
        f"Complete cascade: "
        f"{cascade_mean_per_patient:.9f} seconds"
    )

    # -----------------------------------------------------
    # Summary table
    # -----------------------------------------------------

    results = pd.DataFrame(
        [
            {
                "stage1_model_size_mb":
                    stage1_size,

                "stage2_model_size_mb":
                    stage2_size,

                "final_cascade_size_mb":
                    cascade_size,

                "total_test_cases":
                    EXPECTED_ROWS,

                "direct_final_normal_cases":
                    routing_count,

                "stage2_cases":
                    stage2_count,

                "direct_final_normal_rate_percent":
                    direct_normal_rate,

                "stage2_routing_rate_percent":
                    stage2_rate,

                "stage1_mean_seconds":
                    stage1_timing[
                        "mean_seconds"
                    ],

                "stage2_mean_seconds":
                    stage2_timing[
                        "mean_seconds"
                    ],

                "cascade_mean_seconds":
                    cascade_timing[
                        "mean_seconds"
                    ],

                "stage1_seconds_per_patient":
                    stage1_mean_per_patient,

                "stage2_seconds_per_routed_patient":
                    stage2_mean_per_routed_patient,

                "cascade_seconds_per_patient":
                    cascade_mean_per_patient,
            }
        ]
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # Completion
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 363 COMPLETED")
    print("=" * 80)

    print(
        "Efficiency analysis saved:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print("IMPORTANT:")
    print("- No training was performed.")
    print("- No model was modified.")
    print("- No threshold was changed.")
    print("- No tuning was performed.")
    print("- Timing was measured on the current machine.")
    print("- Final test predictions were not changed.")
    print("- Efficiency measurements do not alter model performance.")


if __name__ == "__main__":
    main()