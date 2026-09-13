import pandas as pd
from pathlib import Path


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = (
    BASE_DIR
    / "data"
    / "thyroid+disease"
    / "ann-train.data"
)


# ============================================================
# Class mapping
# ============================================================

CLASS_NAMES = {
    1: "normal",
    2: "hyperfunction",
    3: "subnormal"
}


# ============================================================
# Load ANN thyroid training dataset
# ============================================================

def load_ann_thyroid_data():

    # Read whitespace-separated data
    df = pd.read_csv(
        TRAIN_FILE,
        sep=r"\s+",
        header=None
    )

    # Last column is the target class
    X = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].copy()

    # Create feature names
    X.columns = [
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

    # Convert target to integer
    y = y.astype(int)

    return X, y


# ============================================================
# Main test
# ============================================================

if __name__ == "__main__":

    X, y = load_ann_thyroid_data()

    print("=" * 60)
    print("ANN THYROID MULTICLASS DATASET")
    print("=" * 60)

    print("\nDataset shape:")
    print(X.shape)

    print("\nNumber of features:")
    print(X.shape[1])

    print("\nClass distribution:")

    class_counts = y.value_counts().sort_index()

    for class_id, count in class_counts.items():

        print(
            f"{class_id} -> "
            f"{CLASS_NAMES[class_id]}: "
            f"{count}"
        )

    print("\nFirst 5 feature rows:")
    print(X.head())

    print("\nFirst 5 target values:")
    print(y.head().tolist())

    print("\nDataset loaded successfully!")