import pandas as pd
import joblib


# ============================================================
# Load models
# ============================================================

stage1_model = joblib.load(
    r".\model\final_xgboost_model.pkl"
)

stage2_model = joblib.load(
    r".\model\final_calibrated_stacking_model.pkl"
)


# ============================================================
# Load one Stage 2 test sample
# ============================================================

X_test = pd.read_csv(
    r".\data\ann_X_test.csv"
)

sample = X_test.iloc[[0]]


# ============================================================
# Stage 2 prediction
# ============================================================

stage2_prediction = stage2_model.predict(
    sample
)[0]

stage2_probabilities = stage2_model.predict_proba(
    sample
)[0]


# ============================================================
# Display results
# ============================================================

print("=" * 60)
print("HIERARCHICAL CASCADE TEST")
print("=" * 60)

print("\nSample shape:")
print(sample.shape)

print("\nStage 2 prediction:")
print(stage2_prediction)

print("\nStage 2 class:")
if stage2_prediction == 1:
    print("normal")
elif stage2_prediction == 2:
    print("hyperfunction")
elif stage2_prediction == 3:
    print("subnormal")
else:
    print("Unknown")

print("\nStage 2 probabilities:")

print(
    "normal:",
    round(float(stage2_probabilities[0]) * 100, 2),
    "%"
)

print(
    "hyperfunction:",
    round(float(stage2_probabilities[1]) * 100, 2),
    "%"
)

print(
    "subnormal:",
    round(float(stage2_probabilities[2]) * 100, 2),
    "%"
)

print("\nHierarchical Stage 2 test completed successfully!")