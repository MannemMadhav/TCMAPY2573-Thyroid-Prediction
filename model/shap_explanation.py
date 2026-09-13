import joblib
import pandas as pd
import shap

# ============================================================
# Load final XGBoost model
# ============================================================

model = joblib.load(
    r".\model\final_xgboost_model.pkl"
)

print("Final XGBoost model loaded.")

# ============================================================
# Load test data
# ============================================================

X_test = pd.read_csv(
    r".\data\X_test.csv"
)

# Use one test patient for explanation
sample = X_test.iloc[[0]]

print("\nSample selected for explanation.")
print(sample)

# ============================================================
# Create SHAP explainer
# ============================================================

explainer = shap.TreeExplainer(model)

# Calculate SHAP values
shap_values = explainer.shap_values(sample)

print("\nSHAP explanation generated successfully!")

print("\nNumber of features:", len(sample.columns))

print("\nSHAP values:")
for feature, value in zip(sample.columns, shap_values[0]):
    print(f"{feature}: {value:.6f}")

    # ============================================================
# Save SHAP values
# ============================================================

shap_result = pd.DataFrame({
    "Feature": sample.columns,
    "SHAP_Value": shap_values[0]
})

shap_result["Absolute_SHAP"] = shap_result["SHAP_Value"].abs()

shap_result = shap_result.sort_values(
    by="Absolute_SHAP",
    ascending=False
)

shap_result.to_csv(
    r".\data\shap_explanation.csv",
    index=False
)

print("\nSHAP explanation saved successfully!")
print("Saved to: data/shap_explanation.csv")

print("\nTop 10 features by absolute SHAP value:")
print(
    shap_result.head(10).to_string(index=False)
)