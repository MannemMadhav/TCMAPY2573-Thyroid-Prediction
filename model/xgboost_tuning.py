import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

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
# Create base XGBoost model
# ============================================================

xgb_model = XGBClassifier(
    scale_pos_weight=scale_ratio,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

# ============================================================
# Define hyperparameter grid
# ============================================================

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 4],
    "learning_rate": [0.05, 0.1]
}

# ============================================================
# Grid Search
# ============================================================

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

print("\nStarting GridSearchCV...")
grid_search.fit(X_train, y_train)

# ============================================================
# Display best parameters
# ============================================================

print("\nBest parameters:")
print(grid_search.best_params_)

print("\nBest cross-validation F1-score:")
print(round(grid_search.best_score_, 4))

# ============================================================
# Evaluate the tuned model on the test set
# ============================================================

X_test = pd.read_csv(r".\data\X_test.csv")
y_test = pd.read_csv(r".\data\y_test.csv")["target"]

# Get the best model found by GridSearchCV
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)

# Calculate test accuracy
accuracy = best_model.score(X_test, y_test)

print("\nTuned XGBoost Test Evaluation:")
print("Accuracy:", round(accuracy, 4))

from sklearn.metrics import confusion_matrix

# Confusion matrix for tuned XGBoost
cm = confusion_matrix(y_test, y_pred)

print("\nTuned XGBoost Confusion Matrix:")
print(cm)

from sklearn.metrics import precision_score, recall_score, f1_score

# Tuned XGBoost metrics for hypothyroid class
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nTuned XGBoost Hypothyroid Metrics:")
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-Score:", round(f1, 4))

from sklearn.metrics import roc_auc_score

# ROC-AUC for tuned XGBoost
y_probability = best_model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_probability)

print("\nTuned XGBoost ROC-AUC:")
print(round(roc_auc, 4))