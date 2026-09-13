import pandas as pd

# ============================================================
# TCMAPY2573 - Thyroid Disease Prediction
# Load the official UCI Hypothyroid Dataset
# ============================================================

# Dataset path
file_path = r".\data\thyroid+disease\hypothyroid.data"

# Column names from hypothyroid.names
columns = [
    "target",
    "age",
    "sex",
    "on_thyroxine",
    "query_on_thyroxine",
    "on_antithyroid_medication",
    "thyroid_surgery",
    "query_hypothyroid",
    "query_hyperthyroid",
    "pregnant",
    "sick",
    "tumor",
    "lithium",
    "goitre",
    "TSH_measured",
    "TSH",
    "T3_measured",
    "T3",
    "TT4_measured",
    "TT4",
    "T4U_measured",
    "T4U",
    "FTI_measured",
    "FTI",
    "TBG_measured",
    "TBG"
]

# Load dataset
df = pd.read_csv(
    file_path,
    header=None,
    names=columns
)

# Display basic information
print("=" * 60)
print("TCMAPY2573 - THYROID DATASET")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nTarget class distribution:")
print(df["target"].value_counts())

print("\nMissing values represented by '?':")
print(df.isin(["?"]).sum())

print("\nDataset loaded successfully!")