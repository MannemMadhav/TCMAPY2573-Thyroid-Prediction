import pandas as pd

# ============================================================
# TCMAPY2573 - Thyroid Disease Prediction
# Data Preprocessing
# ============================================================

# Dataset path
file_path = r".\data\thyroid+disease\hypothyroid.data"

# Column names
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

# Load the original dataset
df = pd.read_csv(
    file_path,
    header=None,
    names=columns
)

print("=" * 60)
print("ORIGINAL DATASET")
print("=" * 60)

print("Shape:", df.shape)

# Replace '?' with missing values
df = df.replace("?", pd.NA)

# TBG has too many missing values, so remove it
df = df.drop(columns=["TBG_measured", "TBG"])

print("\nRemoved TBG because it has more than 90% missing values.")
print("New dataset shape:", df.shape)
print("\nMissing values after replacing '?':")
print(df.isna().sum())

# Convert numeric columns to numeric data types
numeric_columns = [
    "age",
    "TSH",
    "T3",
    "TT4",
    "T4U",
    "FTI"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

print("\nNumeric columns converted successfully.")

# Fill missing numeric values with the median
for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

print("\nMissing numeric values filled using median.")

# Fill missing sex values with the most frequent value
df["sex"] = df["sex"].fillna(df["sex"].mode()[0])

print("\nMissing sex values filled using the most frequent value.")
# Display data types
print("\nData types:")
print(df.dtypes)

# Convert binary categorical values to numeric values
binary_columns = [
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
    "T3_measured",
    "TT4_measured",
    "T4U_measured",
    "FTI_measured"
]

for column in binary_columns:
    df[column] = df[column].map({"f": 0, "t": 1, "n": 0, "y": 1})

# Encode sex
df["sex"] = df["sex"].map({"F": 0, "M": 1})

print("\nCategorical values converted to numeric values.")
print("\nPreprocessing inspection completed!")
print("\nFinal data types:")
print(df.dtypes)

print("\nRemaining missing values:")
print(df.isna().sum().sum())

# Separate features and target
X = df.drop(columns=["target"])
y = df["target"]

print("\nFeatures (X) shape:")
print(X.shape)

print("\nTarget (y) shape:")
print(y.shape)

print("\nFeature columns:")
print(X.columns.tolist())

print("\nTarget classes:")
print(y.value_counts())

# Save the cleaned dataset
processed_df = pd.concat([X, y], axis=1)

processed_df.to_csv(
    r".\data\thyroid_cleaned.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")
print("Saved to: data/thyroid_cleaned.csv")

# ============================================================
# Train-Test Split
# ============================================================

from sklearn.model_selection import train_test_split

# Split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain-Test Split:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

# ============================================================
# Encode Target Labels
# ============================================================

y_train_encoded = y_train.map({
    "negative": 0,
    "hypothyroid": 1
})

y_test_encoded = y_test.map({
    "negative": 0,
    "hypothyroid": 1
})

print("\nEncoded target classes:")
print("negative -> 0")
print("hypothyroid -> 1")

print("\nEncoded training target:")
print(y_train_encoded.value_counts().sort_index())

print("\nEncoded testing target:")
print(y_test_encoded.value_counts().sort_index())

# ============================================================
# Save Prepared Train/Test Data
# ============================================================

X_train.to_csv(r".\data\X_train.csv", index=False)
X_test.to_csv(r".\data\X_test.csv", index=False)

y_train_encoded.to_csv(
    r".\data\y_train.csv",
    index=False,
    header=["target"]
)

y_test_encoded.to_csv(
    r".\data\y_test.csv",
    index=False,
    header=["target"]
)

print("\nPrepared train/test datasets saved successfully!")
print("Saved: data/X_train.csv")
print("Saved: data/X_test.csv")
print("Saved: data/y_train.csv")
print("Saved: data/y_test.csv")