import pandas as pd


# ============================================================
# Example raw clinical values
# ============================================================

raw_data = {
    "age": 73,
    "sex": 0,
    "on_thyroxine": 1,
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
    "TSH": 0.6,
    "T3": 15,
    "TT4": 120,
    "T4U": 82,
    "FTI": 146
}


# ============================================================
# Convert raw values to ANN representation
# ============================================================

ann_data = {
    "age": raw_data["age"] / 100,
    "sex": raw_data["sex"],
    "on_thyroxine": raw_data["on_thyroxine"],
    "query_on_thyroxine": raw_data["query_on_thyroxine"],
    "on_antithyroid_medication":
        raw_data["on_antithyroid_medication"],
    "sick": raw_data["sick"],
    "pregnant": raw_data["pregnant"],
    "thyroid_surgery": raw_data["thyroid_surgery"],
    "I131_treatment": raw_data["I131_treatment"],
    "query_hypothyroid":
        raw_data["query_hypothyroid"],
    "query_hyperthyroid":
        raw_data["query_hyperthyroid"],
    "lithium": raw_data["lithium"],
    "goitre": raw_data["goitre"],
    "tumor": raw_data["tumor"],
    "hypopituitary":
        raw_data["hypopituitary"],
    "psych": raw_data["psych"],
    "TSH": raw_data["TSH"] / 1000,
    "T3": raw_data["T3"] / 1000,
    "TT4": raw_data["TT4"] / 1000,
    "T4U": raw_data["T4U"] / 1000,
    "FTI": raw_data["FTI"] / 1000
}


# ============================================================
# Create DataFrame
# ============================================================

sample = pd.DataFrame(
    [ann_data]
)


# ============================================================
# Display conversion
# ============================================================

print("=" * 60)
print("ANN FEATURE CONVERSION TEST")
print("=" * 60)

print("\nRaw values:")
print(
    pd.Series(raw_data)
)

print("\nANN-scaled values:")
print(
    sample.T
)

print("\nConversion completed successfully!")