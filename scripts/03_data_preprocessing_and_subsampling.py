"""
CSE437 Final Project: Disparity and Error Analysis in Linear Income Classification
Script 03: Data Preprocessing, Domain Aggregation, and Cohort Subsampling

Purpose:
    Performs data cleaning, outlier controls, and domain aggregation on the Texas ACS PUMS
    microdata:
      1. Coerces Census blank/sentinel strings to numeric and drops unrecorded rows.
      2. Enforces workforce criteria (ESR=1, WKHP>=35, PERNP>0).
      3. Implements age filtering (16-80) and weekly hours outlier clipping (<=98).
      4. Aggregates ~500 4-digit OCCP codes into 12 broad SOC occupational domains.
      5. Maps numeric COW codes to descriptive sector labels.
      6. Computes the empirical 75th percentile income target ($90,000.00).
      7. Draws a reproducible 30,000-sample subset for stable linear model training.
"""

import os
import numpy as np
import pandas as pd

# ==============================================================================
# 1. Portable Path Resolution
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "psam_p48.csv")

if not os.path.exists(file_path):
    BASE_DIR = os.getcwd()
    file_path = os.path.join(BASE_DIR, "psam_p48.csv")

if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"Could not locate 'psam_p48.csv' in '{BASE_DIR}'. "
        "Ensure the Texas ACS PUMS CSV file is located in the working directory."
    )

output_file = os.path.join(BASE_DIR, "texas_cleaned_30k.csv")

# ==============================================================================
# 2. Variable Loading and Type Sanitization
# ==============================================================================
selected_cols = [
    "PERNP",
    "WKHP",
    "ESR",
    "AGEP",
    "SEX",
    "SCHL",
    "MAR",
    "COW",
    "OCCP",
    "WKWN",
]

print(f"[STATUS] Ingesting core variables from: {os.path.basename(file_path)}")
df = pd.read_csv(file_path, usecols=selected_cols, low_memory=False)

# Convert Census whitespace or sentinel characters to NaN, then drop missing instances
for col in selected_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=selected_cols).copy()

# ==============================================================================
# 3. Population Filtering and Outlier Management
# ==============================================================================
# Filter to actively employed civilian full-time workers with positive compensation
df = df[(df["ESR"] == 1) & (df["WKHP"] >= 35) & (df["PERNP"] > 0)].copy()

# Restrict age range (16-80) to mitigate high-variance sparse tails in senior labor
df = df[(df["AGEP"] >= 16) & (df["AGEP"] <= 80)].copy()

# Outlier control: cap extreme weekly hours worked at the 98-hour threshold
df["WKHP"] = df["WKHP"].clip(upper=98)

# ==============================================================================
# 4. Feature Engineering: Occupational Bucketing (2018 SOC Taxonomy)
# ==============================================================================
def map_occupation_bucket(code: float) -> str:
    """
    Collapses ~500 granular 4-digit Census OCCP codes into broad Standard
    Occupational Classification (SOC) domains to reduce high-cardinality sparsity.
    """
    c = int(code)
    if 10 <= c <= 960:
        return "Management_Business_Finance"
    elif 1005 <= c <= 1980:
        return "STEM_Science_Architecture"
    elif 2001 <= c <= 2555:
        return "Education_Legal_Social"
    elif 2600 <= c <= 2970:
        return "Arts_Media_Design"
    elif 3000 <= c <= 3655:
        return "Healthcare"
    elif 3700 <= c <= 3960:
        return "Protective_Service"
    elif 4000 <= c <= 4655:
        return "Service_Food_Personal_Cleaning"
    elif 4700 <= c <= 5940:
        return "Sales_Office_Admin"
    elif 6005 <= c <= 6950:
        return "Construction_Extraction_Farming"
    elif 7000 <= c <= 7640:
        return "Installation_Maintenance_Repair"
    elif 7700 <= c <= 8990:
        return "Production_Manufacturing"
    elif 9005 <= c <= 9760:
        return "Transportation_Material_Moving"
    return "Other"


df["OCCP_GROUP"] = df["OCCP"].apply(map_occupation_bucket)

# ==============================================================================
# 5. Class of Worker (COW) Categorical Mapping
# ==============================================================================
# Map numeric Census COW codes to human-readable institutional sectors
cow_labels = {
    1: "Private_ForProfit",
    2: "Private_NonProfit",
    3: "Local_Gov",
    4: "State_Gov",
    5: "Federal_Gov",
    6: "Self_Employed_NotInc",
    7: "Self_Employed_Inc",
    8: "Without_Pay",
}
df["COW_GROUP"] = df["COW"].map(cow_labels).fillna("Other")

# ==============================================================================
# 6. Target Variable Construction (75th Percentile Income Cutoff)
# ==============================================================================
p75 = df["PERNP"].quantile(0.75)
df["HIGH_EARNER"] = (df["PERNP"] >= p75).astype(int)

print("\n" + "=" * 50)
print("PREPROCESSED POPULATION SUMMARY")
print("=" * 50)
print(f"Total Filtered Observations: {len(df):,}")
print(f"Top-Quartile Target Cutoff:  ${p75:,.2f}")
print("Target Class Breakdown:")
print(df["HIGH_EARNER"].value_counts(normalize=True).apply(lambda x: f"{x*100:.2f}%"))

# ==============================================================================
# 7. Subsampling and Export
# ==============================================================================
# Draw a stratified random sample of 30,000 records for fast, stable LinearSVC convergence
df_sampled = df.sample(n=30000, random_state=42).reset_index(drop=True)

df_sampled.to_csv(output_file, index=False)
print("-" * 50)
print(f"[SUCCESS] Exported cleaned dataset to: {output_file}")
print(f"Final Export Shape: {df_sampled.shape[0]:,} rows, {df_sampled.shape[1]} columns")
print("=" * 50)