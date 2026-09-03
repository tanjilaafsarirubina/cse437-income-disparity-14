"""
CSE437 Final Project: Disparity and Error Analysis in Linear Income Classification
Script 05: Hyperparameter Tuning, Multi-Model Comparison, and Baseline Evaluation

Requirements Satisfied:
  - Hyperparameter Tuning: GridSearchCV across C on LinearSVC.
  - Multi-Model Validation: Compares LinearSVC against LogisticRegression.
"""

import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(BASE_DIR, "X_train.csv")):
    BASE_DIR = os.getcwd()

# 1. Load data partitions
X_train = pd.read_csv(os.path.join(BASE_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(BASE_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(BASE_DIR, "y_train.csv")).squeeze()
y_test = pd.read_csv(os.path.join(BASE_DIR, "y_test.csv")).squeeze()
test_meta = pd.read_csv(os.path.join(BASE_DIR, "test_metadata.csv"))

# ==============================================================================
# Hyperparameter Tuning (Grid Search with 3-Fold Cross-Validation)
# ==============================================================================
print("=" * 60)
print("HYPERPARAMETER TUNING: LinearSVC (Search Space: C in [0.01, 0.1, 1.0, 10.0])")
print("=" * 60)

param_grid = {"C": [0.01, 0.1, 1.0, 10.0]}
base_svm = LinearSVC(dual=False, max_iter=5000, random_state=42)

grid_search = GridSearchCV(
    estimator=base_svm,
    param_grid=param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"Optimal Hyperparameter: C = {grid_search.best_params_['C']}")
print(f"Best Validation F1-Score: {grid_search.best_score_:.4f}")
print("\nFull Grid Search Evaluation:")
for mean_score, params in zip(grid_search.cv_results_["mean_test_score"], grid_search.cv_results_["params"]):
    print(f"  C = {params['C']:<5} -> Mean Validation F1 = {mean_score:.4f}")

# Best tuned LinearSVC model
best_svm = grid_search.best_estimator_
y_pred_svm = best_svm.predict(X_test)
decision_scores = best_svm.decision_function(X_test)

# ==============================================================================
# Multi-Model Validation: Second Model Family (Logistic Regression)
# ==============================================================================
print("\n" + "=" * 60)
print("MODEL COMPARISON: LinearSVC vs. LogisticRegression")
print("=" * 60)

log_reg = LogisticRegression(max_iter=5000, random_state=42)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)

comparison_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision (1)", "Recall (1)", "F1-Score (1)"],
    "LinearSVC (Tuned)": [
        accuracy_score(y_test, y_pred_svm),
        precision_score(y_test, y_pred_svm),
        recall_score(y_test, y_pred_svm),
        f1_score(y_test, y_pred_svm),
    ],
    "LogisticRegression": [
        accuracy_score(y_test, y_pred_lr),
        precision_score(y_test, y_pred_lr),
        recall_score(y_test, y_pred_lr),
        f1_score(y_test, y_pred_lr),
    ],
})
print(comparison_df.to_string(index=False))

# ==============================================================================
# Baseline Test Metrics & Error Tagging (Primary Model: LinearSVC)
# ==============================================================================
test_meta["PRED_HIGH_EARNER"] = y_pred_svm
test_meta["DECISION_SCORE"] = decision_scores
test_meta["CORRECT"] = (test_meta["HIGH_EARNER_ACTUAL"] == y_pred_svm).astype(int)
test_meta["ERROR_TYPE"] = "Correct"
test_meta.loc[
    (test_meta["HIGH_EARNER_ACTUAL"] == 1) & (test_meta["PRED_HIGH_EARNER"] == 0),
    "ERROR_TYPE",
] = "False_Negative"
test_meta.loc[
    (test_meta["HIGH_EARNER_ACTUAL"] == 0) & (test_meta["PRED_HIGH_EARNER"] == 1),
    "ERROR_TYPE",
] = "False_Positive"

eval_output_path = os.path.join(BASE_DIR, "test_predictions_evaluated.csv")
test_meta.to_csv(eval_output_path, index=False)
comparison_df.to_csv(os.path.join(BASE_DIR, "model_family_comparison.csv"), index=False)

print(f"\n[SUCCESS] Exported evaluated test data to: {eval_output_path}")