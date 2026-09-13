import sys
from pathlib import Path

import pandas as pd

# ------------------------------------------------------------
# Make model folder importable
# ------------------------------------------------------------

MODEL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODEL_DIR))

from hierarchical_cascade import (
    HierarchicalThyroidCascade,
    stage1_model,
    stage2_model,
)


# ------------------------------------------------------------
# Load test data
# ------------------------------------------------------------

X = pd.read_csv(
    Path(__file__).resolve().parent.parent
    / "data"
    / "ann_X_test.csv"
)

y = pd.read_csv(
    Path(__file__).resolve().parent.parent
    / "data"
    / "ann_y_test.csv"
).iloc[:, 0]


# ------------------------------------------------------------
# Create cascade
# ------------------------------------------------------------

cascade = HierarchicalThyroidCascade(
    stage1_model,
    stage2_model
)


print("=" * 60)
print("SEARCHING FOR CORRECT HYPERFUNCTION SAMPLE")
print("=" * 60)


# ------------------------------------------------------------
# Search class-2 samples
# ------------------------------------------------------------

found = False

for i in range(len(X)):

    # Class 2 = hyperfunction
    if int(y.iloc[i]) != 2:
        continue

    sample = X.iloc[[i]]

    result = cascade.predict(
        sample
    )

    if result["final_class"] == "hyperfunction":

        row = X.iloc[i]

        print("\nCorrect hyperfunction sample found!")
        print("Test index:", i)

        print("\nANN-scaled values:")
        print(row.to_string())

        print("\nRaw website values:")

        print("age:", round(float(row["age"]) * 100, 2))
        print("sex:", int(row["sex"]))
        print("on_thyroxine:", int(row["on_thyroxine"]))
        print(
            "query_on_thyroxine:",
            int(row["query_on_thyroxine"])
        )
        print(
            "on_antithyroid_medication:",
            int(row["on_antithyroid_medication"])
        )
        print("sick:", int(row["sick"]))
        print("pregnant:", int(row["pregnant"]))
        print(
            "thyroid_surgery:",
            int(row["thyroid_surgery"])
        )
        print(
            "I131_treatment:",
            int(row["I131_treatment"])
        )
        print(
            "query_hypothyroid:",
            int(row["query_hypothyroid"])
        )
        print(
            "query_hyperthyroid:",
            int(row["query_hyperthyroid"])
        )
        print("lithium:", int(row["lithium"]))
        print("goitre:", int(row["goitre"]))
        print("tumor:", int(row["tumor"]))
        print(
            "hypopituitary:",
            int(row["hypopituitary"])
        )
        print("psych:", int(row["psych"]))

        print("TSH:", float(row["TSH"]) * 1000)
        print("T3:", float(row["T3"]) * 1000)
        print("TT4:", float(row["TT4"]) * 1000)
        print("T4U:", float(row["T4U"]) * 1000)
        print("FTI:", float(row["FTI"]) * 1000)

        print("\nModel results:")
        print(
            "Stage 1:",
            result["stage1_class"]
        )
        print(
            "Stage 2:",
            result["stage2_class"]
        )
        print(
            "Final:",
            result["final_class"]
        )

        print(
            "Stage 2 probabilities:",
            result["stage2_probabilities"]
        )

        found = True
        break


if not found:
    print(
        "\nNo correctly classified hyperfunction "
        "sample was found."
    )