import pandas as pd


# ============================================================
# ANN feature order
# ============================================================

ANN_FEATURES = [
    "age",
    "sex",
    "on_thyroxine",
    "query_on_thyroxine",
    "on_antithyroid_medication",
    "sick",
    "pregnant",
    "thyroid_surgery",
    "I131_treatment",
    "query_hypothyroid",
    "query_hyperthyroid",
    "lithium",
    "goitre",
    "tumor",
    "hypopituitary",
    "psych",
    "TSH",
    "T3",
    "TT4",
    "T4U",
    "FTI"
]


# ============================================================
# Convert raw clinical values to ANN model format
# ============================================================

def convert_to_ann_features(data):
    """
    Convert raw clinical values into the
    21-feature ANN representation.
    """

    ann_data = {
        "age": float(data["age"]) / 100.0,

        "sex": int(data["sex"]),

        "on_thyroxine":
            int(data["on_thyroxine"]),

        "query_on_thyroxine":
            int(data["query_on_thyroxine"]),

        "on_antithyroid_medication":
            int(data["on_antithyroid_medication"]),

        "sick":
            int(data["sick"]),

        "pregnant":
            int(data["pregnant"]),

        "thyroid_surgery":
            int(data["thyroid_surgery"]),

        "I131_treatment":
            int(data["I131_treatment"]),

        "query_hypothyroid":
            int(data["query_hypothyroid"]),

        "query_hyperthyroid":
            int(data["query_hyperthyroid"]),

        "lithium":
            int(data["lithium"]),

        "goitre":
            int(data["goitre"]),

        "tumor":
            int(data["tumor"]),

        "hypopituitary":
            int(data["hypopituitary"]),

        "psych":
            int(data["psych"]),

        "TSH":
            float(data["TSH"]) / 1000.0,

        "T3":
            float(data["T3"]) / 1000.0,

        "TT4":
            float(data["TT4"]) / 1000.0,

        "T4U":
            float(data["T4U"]) / 1000.0,

        "FTI":
            float(data["FTI"]) / 1000.0
    }

    return pd.DataFrame(
        [ann_data],
        columns=ANN_FEATURES
    )


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

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

    result = convert_to_ann_features(
        raw_data
    )

    print("=" * 60)
    print("ANN FEATURE CONVERSION UTILITY")
    print("=" * 60)

    print("\nConverted feature DataFrame:")
    print(result)

    print("\nFeature count:")
    print(result.shape[1])

    print("\nConversion utility working successfully!")