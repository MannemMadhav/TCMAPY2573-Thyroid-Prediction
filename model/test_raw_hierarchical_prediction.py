import sys
from pathlib import Path

import pandas as pd

# Make the model folder importable
MODEL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODEL_DIR))

from ann_feature_conversion import convert_to_ann_features
from hierarchical_cascade import (
    HierarchicalThyroidCascade,
    stage1_model,
    stage2_model
)


# ============================================================
# Example RAW clinical values
# ============================================================

raw_data = {
    "age": 50,
    "sex": 0,
    "on_thyroxine": 0,
    "query_on_thyroxine": 0,
    "on_antithyroid_medication": 0,
    "sick": 0,
    "pregnant": 0,
    "thyroid_surgery": 0,
    "I131_treatment": 0,
    "query_hypothyroid": 0,
    "query_hyperthyroid": 0,
    "lithium": 0,
    "goitre": 0,
    "tumor": 0,
    "hypopituitary": 0,
    "psych": 0,
    "TSH": 2.0,
    "T3": 1.5,
    "TT4": 100.0,
    "T4U": 80.0,
    "FTI": 125.0
}


# ============================================================
# Convert RAW values to ANN representation
# ============================================================

sample = convert_to_ann_features(
    raw_data
)


# ============================================================
# Create hierarchical cascade
# ============================================================

cascade = HierarchicalThyroidCascade(
    stage1_model,
    stage2_model
)


# ============================================================
# Predict
# ============================================================

result = cascade.predict(
    sample
)


# ============================================================
# Display results
# ============================================================

print("=" * 60)
print("RAW → ANN → HIERARCHICAL CASCADE TEST")
print("=" * 60)

print("\nConverted feature shape:")
print(sample.shape)

print("\nConverted features:")
print(sample.to_string(index=False))

print("\nStage 1 class:")
print(
    result["stage1_class"]
)

print("\nFinal class:")
print(
    result["final_class"]
)

print("\nStage 2 class:")
print(
    result["stage2_class"]
)

print("\nStage 1 probabilities:")

print(
    result["stage1_probabilities"]
)

if result["stage2_probabilities"] is not None:

    print("\nStage 2 probabilities:")

    print(
        result["stage2_probabilities"]
    )

else:

    print(
        "\nStage 2 probabilities: None"
    )

print(
    "\nRaw-to-hierarchical prediction test completed successfully!"
)