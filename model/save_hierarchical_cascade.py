import joblib

from hierarchical_cascade import (
    HierarchicalThyroidCascade,
    stage1_model,
    stage2_model
)


# ============================================================
# Create complete hierarchical cascade
# ============================================================

cascade = HierarchicalThyroidCascade(
    stage1_model,
    stage2_model
)


# ============================================================
# Save complete cascade
# ============================================================

MODEL_PATH = (
    r".\model\final_hierarchical_cascade.pkl"
)

joblib.dump(
    cascade,
    MODEL_PATH
)


print("=" * 60)
print("HIERARCHICAL CASCADE SAVED")
print("=" * 60)

print("\nSaved to:")
print(MODEL_PATH)

print(
    "\nComplete hierarchical cascade saved successfully!"
)