import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ============================================================
# Load prepared training data
# ============================================================

X_train = pd.read_csv(r".\data\X_train.csv")
y_train = pd.read_csv(r".\data\y_train.csv")["target"]

print("Training data loaded.")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# ============================================================
# Create Random Forest model
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# ============================================================
# Train the model
# ============================================================

model.fit(X_train, y_train)

print("\nRandom Forest model trained successfully!")

# ============================================================
# Evaluate Random Forest
# ============================================================

X_test = pd.read_csv(r".\data\X_test.csv")
y_test = pd.read_csv(r".\data\y_test.csv")["target"]

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = model.score(X_test, y_test)

print("\nRandom Forest Evaluation:")
print("Accuracy:", round(accuracy, 4))

from sklearn.metrics import confusion_matrix

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

from sklearn.metrics import precision_score, recall_score, f1_score

# Calculate evaluation metrics for hypothyroid class
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nHypothyroid Class Metrics:")
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-Score:", round(f1, 4))

from sklearn.metrics import roc_auc_score

# ROC-AUC Score
y_probability = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_probability)

print("\nROC-AUC Score:")
print(round(roc_auc, 4))

from sklearn.metrics import classification_report

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["negative", "hypothyroid"]
))

# ============================================================
# Random Forest Feature Importance
# ============================================================

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features:")
print(feature_importance.head(10).to_string(index=False))

# Save feature importance results
feature_importance.to_csv(
    r".\data\random_forest_feature_importance.csv",
    index=False
)

print("\nFeature importance saved successfully!")
print("Saved to: data/random_forest_feature_importance.csv")