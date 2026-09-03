"""
CSE437 Final Project: Disparity and Error Analysis in Linear Income Classification
Script 04: Feature Engineering, Stratified Partitioning, and Linear Encoding Pipeline

Purpose:
    1. Buckets raw Census educational attainment codes (SCHL) into 4 standard tiers.
    2. Maps demographic sex codes to descriptive labels.
    3. Partitions the 30,000-sample dataset into an 80/20 stratified train/test split.
    4. Applies scikit-learn ColumnTransformer:
         - Standardizes numeric features (StandardScaler) fit strictly on train to prevent data leakage.
         - Dummy-encodes categorical variables (OneHotEncoder, drop='first') to prevent multicollinearity.
    5. Exports transformed feature matrices, target vectors, and raw slice metadata for RQ auditing.
"""

import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ==============================================================================
# 1. Portable File Path Resolution
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "texas_cleaned_30k.csv")

if not os.path.exists(file_path):
    BASE_DIR = os.getcwd()
    file_path = os.path.join(BASE_DIR, "texas_cleaned_30k.csv")

if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"Could not locate 'texas_cleaned_30k.csv' in '{BASE_DIR}'. "
        "Please run '03_data_preprocessing_and_subsampling.py' first."
    )

print(f"[STATUS] Loading cleaned sample from: {os.path.basename(file_path)}")
df = pd.read_csv(file_path)

# ==============================================================================
# 2. Education Tier Bucketing & Demographic Labeling
# ==============================================================================
def bucket_education(code: float) -> str:
    """
    Collapses 24 granular Census SCHL attainment levels into 4 distinct tiers:
      - Less_than_HS:       Codes 1-15  (No high school diploma / GED)
      - HS_or_Some_College: Codes 16-20 (Regular diploma, GED, some college, Associate's)
      - Bachelors:          Code 21     (4-year Bachelor's degree)
      - Graduate_Plus:      Codes 22-24 (Master's, Professional degree, Doctorate)
    """
    c = int(code)
    if c < 16:
        return "Less_than_HS"
    elif 16 <= c <= 20:
        return "HS_or_Some_College"
    elif c == 21:
        return "Bachelors"
    else:
        return "Graduate_Plus"


df["SCHL_TIER"] = df["SCHL"].apply(bucket_education)

# Map numeric Census SEX (1: Male, 2: Female) to explicit strings
df["SEX_LABEL"] = df["SEX"].map({1: "Male", 2: "Female"})

# ==============================================================================
# 3. Feature Selection & Metadata Isolation
# ==============================================================================
# Core predictive features:
#   - Categorical (5): SEX_LABEL (RQ1), COW_GROUP (RQ2), OCCP_GROUP, SCHL_TIER (RQ1), MAR
#   - Numeric (2):     AGEP (RQ1 trajectory), WKHP
categorical_features = ["SEX_LABEL", "COW_GROUP", "OCCP_GROUP", "SCHL_TIER", "MAR"]
numeric_features = ["AGEP", "WKHP"]
feature_cols = categorical_features + numeric_features

X = df[feature_cols].copy()
y = df["HIGH_EARNER"].copy()

# ==============================================================================
# 4. Stratified Train/Test Partitioning (80/20)
# ==============================================================================
# Stratification ensures identical top-quartile class distributions across splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# ==============================================================================
# 5. Leakage-Free Preprocessing Pipeline
# ==============================================================================
# Methodological safeguard:
#   1. StandardScaler is fit strictly on X_train to prevent test data leakage.
#   2. drop='first' avoids dummy variable trap / exact collinearity in linear SVM.
#   3. handle_unknown='ignore' ensures robustness against unseen test categories.
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        (
            "cat",
            OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
            categorical_features,
        ),
    ]
)

# Fit exclusively on train set, transform both partitions
X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

# Extract post-transformation feature column names for inspectability
encoded_cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(
    categorical_features
)
all_feature_names = numeric_features + list(encoded_cat_names)

# Reassemble structured DataFrames with original split indices preserved
X_train_df = pd.DataFrame(
    X_train_encoded, columns=all_feature_names, index=X_train.index
)
X_test_df = pd.DataFrame(
    X_test_encoded, columns=all_feature_names, index=X_test.index
)

# ==============================================================================
# 6. Metadata Retention & Dataset Export
# ==============================================================================
# Preserve unencoded demographic and sectoral identifiers for slice-level evaluation
test_metadata = X_test[["SEX_LABEL", "COW_GROUP", "SCHL_TIER", "AGEP"]].copy()
test_metadata["HIGH_EARNER_ACTUAL"] = y_test

# Export processed artifacts
X_train_df.to_csv(os.path.join(BASE_DIR, "X_train.csv"), index=False)
X_test_df.to_csv(os.path.join(BASE_DIR, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(BASE_DIR, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(BASE_DIR, "y_test.csv"), index=False)
test_metadata.to_csv(os.path.join(BASE_DIR, "test_metadata.csv"), index=False)

# ==============================================================================
# 7. Preprocessing Integrity Verification
# ==============================================================================
print("\n" + "=" * 50)
print("PREPROCESSING PIPELINE SUMMARY")
print("=" * 50)
print(f"X_train Shape: {X_train_df.shape} | y_train Instances: {len(y_train):,}")
print(f"X_test Shape:  {X_test_df.shape}  | y_test Instances:  {len(y_test):,}")
print(f"Total Dimensionality (Features): {len(all_feature_names)}")

print("-" * 50)
print(f"Train High-Earner Class Balance: {y_train.mean()*100:.2f}%")
print(f"Test High-Earner Class Balance:  {y_test.mean()*100:.2f}%")
print("-" * 50)

print("Encoded Feature Columns (28 Total):")
for idx, name in enumerate(all_feature_names, 1):
    print(f"  {idx:02d}. {name}")
print("=" * 50)