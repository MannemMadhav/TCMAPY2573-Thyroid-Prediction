import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report, roc_auc_score
# ============================================================
# Load prepared training data
# ============================================================

X_train = pd.read_csv(r".\data\X_train.csv")
y_train = pd.read_csv(r".\data\y_train.csv")["target"]

print("Training data loaded.")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# ============================================================
# Create Logistic Regression model
# ============================================================

model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

# ============================================================
# Train the model
# ============================================================

model.fit(X_train, y_train)

print("\nLogistic Regression model trained successfully!")
# ============================================================
# Evaluate Logistic Regression
# ============================================================

X_test = pd.read_csv(r".\data\X_test.csv")
y_test = pd.read_csv(r".\data\y_test.csv")["target"]

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = model.score(X_test, y_test)

print("\nLogistic Regression Evaluation:")
print("Accuracy:", round(accuracy, 4))


# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)



# Calculate evaluation metrics for hypothyroid class
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nHypothyroid Class Metrics:")
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-Score:", round(f1, 4))



print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["negative", "hypothyroid"]
))

# ROC-AUC Score
y_probability = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_probability)

print("\nROC-AUC Score:")
print(round(roc_auc, 4))