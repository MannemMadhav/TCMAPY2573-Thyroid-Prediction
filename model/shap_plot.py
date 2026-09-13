import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Load SHAP explanation
# ============================================================

df = pd.read_csv(
    r".\data\shap_explanation.csv"
)

# Select top 10 features
top_features = df.head(10).copy()

# Sort for plotting
top_features = top_features.sort_values(
    by="SHAP_Value"
)

# ============================================================
# Create SHAP explanation graph
# ============================================================

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"],
    top_features["SHAP_Value"]
)

plt.xlabel("SHAP Value")
plt.ylabel("Feature")
plt.title("Top 10 Features Influencing the Sample Prediction")

plt.axvline(
    x=0,
    linewidth=1
)

plt.tight_layout()

# ============================================================
# Save graph
# ============================================================

plt.savefig(
    r".\data\shap_explanation.png",
    dpi=300
)

print("SHAP explanation graph saved successfully!")
print("Saved to: data/shap_explanation.png")