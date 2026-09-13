import pandas as pd
from xgboost import XGBClassifier

# ============================================================
# Load prepared training data
# ============================================================

X_train = pd.read_csv(r".\data\X_train.csv")
y_train = pd.read_csv(r".\data\y_train.csv")["target"]

print("Training data loaded.")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# ============================================================
# Calculate class imbalance ratio
# ============================================================

negative_count = (y_train == 0).sum()
hypothyroid_count = (y_train == 1).sum()

scale_ratio = negative_count / hypothyroid_count

print("\nClass imbalance ratio:", round(scale_ratio, 2))

# ============================================================
# Create XGBoost model
# ============================================================

model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_ratio,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

# ============================================================
# Train the model
# ============================================================

model.fit(X_train, y_train)

print("\nXGBoost model trained successfully!")

# ============================================================
# Evaluate XGBoost
# ============================================================

X_test = pd.read_csv(r".\data\X_test.csv")
y_test = pd.read_csv(r".\data\y_test.csv")["target"]

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = model.score(X_test, y_test)

print("\nXGBoost Evaluation:")
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
# Save Final XGBoost Model
# ============================================================

import joblib

model_path = r".\model\final_xgboost_model.pkl"

joblib.dump(model, model_path)

print("\nFinal XGBoost model saved successfully!")
print("Saved to:", model_path)