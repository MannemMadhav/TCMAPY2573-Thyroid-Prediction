import joblib
import pandas as pd

# ============================================================
# Load final XGBoost model
# ============================================================

model = joblib.load(
    r".\model\final_xgboost_model.pkl"
)

# ============================================================
# Create one sample patient
# ============================================================

sample = pd.DataFrame([{
    "age": 50,
    "sex": 0,
    "on_thyroxine": 0,
    "query_on_thyroxine": 0,
    "on_antithyroid_medication": 0,
    "thyroid_surgery": 0,
    "query_hypothyroid": 1,
    "query_hyperthyroid": 0,
    "pregnant": 0,
    "sick": 0,
    "tumor": 0,
    "lithium": 0,
    "goitre": 0,
    "TSH_measured": 1,
    "TSH": 8.0,
    "T3_measured": 1,
    "T3": 1.5,
    "TT4_measured": 1,
    "TT4": 80.0,
    "T4U_measured": 1,
    "T4U": 1.0,
    "FTI_measured": 1,
    "FTI": 80.0
}])

# Make sure the feature order exactly matches the model
sample = sample[model.feature_names_in_]

# ============================================================
# Prediction
# ============================================================

prediction = model.predict(sample)[0]
probability = model.predict_proba(sample)[0]

print("Prediction:", prediction)

print("\nPrediction probabilities:")
print("Negative:", round(probability[0], 4))
print("Hypothyroid:", round(probability[1], 4))

if prediction == 1:
    print("\nPredicted class: HYPOTHYROID")
else:
    print("\nPredicted class: NEGATIVE")