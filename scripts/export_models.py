"""
CSE437 Final Project: Disparity and Error Analysis in Linear Income Classification
Script: export_models.py

Purpose:
    Retrains both linear classifiers on data/processed/X_train.csv (written by script 04)
    and serializes them to models/ with joblib. LinearSVC is fixed at C=1.0, the setting
    used in the final report.
"""

import os
from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Resolve paths from the repository root so the script runs from any directory
REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load training matrices
print("[STATUS] Loading training data from data/processed/...")
X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze("columns")

# 2. Train Tuned LinearSVC (C=1.0, dual=False)
print("[STATUS] Training LinearSVC (C=1.0)...")
linear_svc = LinearSVC(C=1.0, dual=False, max_iter=5000, random_state=42)
linear_svc.fit(X_train, y_train)

# 3. Train Baseline LogisticRegression
print("[STATUS] Training LogisticRegression baseline...")
log_reg = LogisticRegression(max_iter=5000, random_state=42)
log_reg.fit(X_train, y_train)

# 4. Serialize and export models
svc_path = MODELS_DIR / "linear_svc_tuned.joblib"
lr_path = MODELS_DIR / "logistic_regression_baseline.joblib"

joblib.dump(linear_svc, svc_path)
joblib.dump(log_reg, lr_path)

print("-" * 55)
print(f"[SUCCESS] Exported: {svc_path.resolve()} ({svc_path.stat().st_size / 1024:.1f} KB)")
print(f"[SUCCESS] Exported: {lr_path.resolve()} ({lr_path.stat().st_size / 1024:.1f} KB)")
print("-" * 55)