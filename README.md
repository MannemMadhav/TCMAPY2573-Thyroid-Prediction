# TCMAPY2573 - Enhancing Thyroid Dysfunction Prediction with Ensemble Learning and Hierarchical Classification

## Problem Statement

The objective of this project is to develop an accurate and reliable machine learning framework for predicting thyroid dysfunction using a comprehensive public dataset.

The project introduces two novel models:

1. Calibrated Stacking Ensemble Model
2. Two-Stage Hierarchical Cascade Classifier

The system uses leakage-aware preprocessing, clinically inspired feature engineering, and SMOTE-based class balancing to improve prediction accuracy, minority-class detection, and clinical decision support.

---

## Novel Model 1 - Calibrated Stacking Ensemble

### Base Learners

- Random Forest
- XGBoost
- Support Vector Machine (SVM)

### Meta-Learner

- Logistic Regression

### Architecture

Random Forest + XGBoost + SVM
                |
                v
      Logistic Regression
         Meta-Learner
                |
                v
      Calibrated Probabilities

The three base learners use calibrated probabilities, which are combined by Logistic Regression as the meta-learner.

---

## Novel Model 2 - Two-Stage Hierarchical Cascade Classifier

### Stage 1 - Binary Screening

XGBoost classifies the input as:

- Normal
- Thyroid Dysfunction

### Stage 2 - Multi-Class Diagnosis

Samples classified as thyroid dysfunction are passed to the calibrated stacking ensemble.

Stage 2 predicts:

- Normal
- Hyperfunction
- Subnormal

### Overall Flow

Patient Input
    |
    v
21 ANN-Compatible Features
    |
    v
Stage 1 XGBoost
    |
    +---- Normal ------------> Final Normal
    |
    +---- Thyroid Dysfunction
                |
                v
     Calibrated Stacking Ensemble
        RF + XGBoost + SVM
                |
                v
      Logistic Regression
         Meta-Learner
                |
                v
      Normal / Hyperfunction / Subnormal

---

## Dataset

**UCI ANN Thyroid Disease Dataset**

- Total samples: 3772
- Features: 21
- Training samples: 3017
- Test samples: 755

The test set is kept completely untouched during SMOTE processing.

---

## Preprocessing

The project performs:

- Data cleaning
- Leakage-aware preprocessing
- Train/test separation
- Clinically inspired feature engineering
- ANN-compatible feature conversion
- SMOTE-based class balancing

### SMOTE

SMOTE is applied only to training data.

- Original training: 3017 samples
- SMOTE training: 8370 samples
- Untouched test: 755 samples

---

## Algorithms

### Stage 1

- XGBoost

### Stage 2 Base Learners

- Random Forest
- XGBoost
- Support Vector Machine

### Stage 2 Meta-Learner

- Logistic Regression

### Calibration

- CalibratedClassifierCV
- predict_proba
- 3-fold cross-validation

---

## Model Performance

### Stage 1

- Accuracy: 99.87%
- Macro F1: 98.61%

### Stage 2 Calibrated Stacking

- Accuracy: 99.07%
- Macro F1: 93.95%

### Complete Hierarchical Cascade

- Accuracy: 99.34%
- Macro F1: 96.35%

### Hierarchical Class Performance

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Normal | 0.95 | 0.95 | 0.95 |
| Hyperfunction | 0.97 | 0.92 | 0.95 |
| Subnormal | 1.00 | 1.00 | 1.00 |

---

## Explainable AI

The project uses SHAP (SHapley Additive exPlanations) to explain individual predictions.

The application displays important contributing features such as:

- FTI
- TSH
- TT4
- T3
- T4U
- Age

---

## Web Application

The Flask application provides:

- User registration
- User login
- Logout
- Protected routes
- Patient/test input
- Thyroid prediction
- Stage 1 probabilities
- Stage 2 probabilities
- SHAP explanations
- Prediction history
- SQLite database

---

## Prediction Flow

Raw Clinical Values
    |
    v
Input Validation
    |
    v
21 ANN Feature Conversion
    |
    v
Stage 1 XGBoost
    |
    v
Stage 2 Calibrated Stacking
    |
    v
Final Prediction
    |
    +---- SHAP Explanation
    |
    +---- SQLite Prediction History

---

## Important Model Files

```text
model/final_calibrated_stacking_model.pkl
model/final_hierarchical_cascade.pkl
model/hierarchical_stage1_xgboost.pkl