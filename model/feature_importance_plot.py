import pandas as pd
import matplotlib.pyplot as plt

# Load feature importance results
file_path = r".\data\random_forest_feature_importance.csv"

df = pd.read_csv(file_path)

# Select top 10 features
top_features = df.head(10)

# Create the graph
plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Random Forest Feature Importance")

plt.tight_layout()

# Save the graph
plt.savefig(
    r".\data\random_forest_feature_importance.png",
    dpi=300
)

print("Feature importance graph saved successfully!")
print("Saved to: data/random_forest_feature_importance.png")