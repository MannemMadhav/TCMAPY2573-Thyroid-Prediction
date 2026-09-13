import pandas as pd
import matplotlib.pyplot as plt

# Load model comparison results
file_path = r".\data\model_comparison.csv"
df = pd.read_csv(file_path)

# Create accuracy comparison graph
plt.figure(figsize=(9, 6))

plt.bar(
    df["Model"],
    df["Accuracy"]
)

plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.ylim(0.90, 1.00)

plt.tight_layout()

# Save graph
plt.savefig(
    r".\data\model_accuracy_comparison.png",
    dpi=300
)

print("Model comparison graph saved successfully!")
print("Saved to: data/model_accuracy_comparison.png")