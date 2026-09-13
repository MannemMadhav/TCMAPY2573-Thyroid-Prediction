import pandas as pd

from ann_feature_conversion import (
    convert_to_ann_features,
    ANN_FEATURES
)


# ============================================================
# Load real ANN training data
# ============================================================

X_train = pd.read_csv(
    r".\data\ann_X_train.csv"
)

real_sample = X_train.iloc[0]


# ============================================================
# Convert ANN values back to raw-style values
#
# This is only for verification.
# ============================================================

raw_data = {
    "age": real_sample["age"] * 100,

    "sex": int(real_sample["sex"]),

    "on_thyroxine":
        int(real_sample["on_thyroxine"]),

    "query_on_thyroxine":
        int(real_sample["query_on_thyroxine"]),

    "on_antithyroid_medication":
        int(real_sample["on_antithyroid_medication"]),

    "sick":
        int(real_sample["sick"]),

    "pregnant":
        int(real_sample["pregnant"]),

    "thyroid_surgery":
        int(real_sample["thyroid_surgery"]),

    "I131_treatment":
        int(real_sample["I131_treatment"]),

    "query_hypothyroid":
        int(real_sample["query_hypothyroid"]),

    "query_hyperthyroid":
        int(real_sample["query_hyperthyroid"]),

    "lithium":
        int(real_sample["lithium"]),

    "goitre":
        int(real_sample["goitre"]),

    "tumor":
        int(real_sample["tumor"]),

    "hypopituitary":
        int(real_sample["hypopituitary"]),

    "psych":
        int(real_sample["psych"]),

    "TSH":
        real_sample["TSH"] * 1000,

    "T3":
        real_sample["T3"] * 1000,

    "TT4":
        real_sample["TT4"] * 1000,

    "T4U":
        real_sample["T4U"] * 1000,

    "FTI":
        real_sample["FTI"] * 1000
}


# ============================================================
# Convert back to ANN representation
# ============================================================

converted = convert_to_ann_features(
    raw_data
)


# ============================================================
# Compare values
# ============================================================

original = real_sample[
    ANN_FEATURES
].astype(float)

recreated = converted.iloc[0].astype(float)

difference = (
    original - recreated
).abs()


# ============================================================
# Display results
# ============================================================

print("=" * 60)
print("ANN CONVERSION VERIFICATION")
print("=" * 60)

print("\nMaximum absolute difference:")
print(
    difference.max()
)

print("\nOriginal ANN sample:")
print(
    original.to_string()
)

print("\nRecreated ANN sample:")
print(
    recreated.to_string()
)

if difference.max() < 1e-12:
    print(
        "\nSUCCESS: Conversion is numerically identical."
    )
else:
    print(
        "\nWARNING: Conversion differences detected."
    )