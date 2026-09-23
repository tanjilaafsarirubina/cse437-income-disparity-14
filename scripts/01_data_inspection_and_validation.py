"""
CSE437 Final Project: Disparity and Error Analysis in Linear Income Classification
Script 01: Raw Data Ingestion, Memory Audit, and Schema Validation

Purpose:
    Validates the presence, dimensions, memory footprint, and column schema
    of the raw 2023 U.S. Census Bureau ACS 1-Year PUMS person-record file for Texas
    (psam_p48.csv) prior to filtering and downstream modeling.
"""

import os
import pandas as pd

# ==============================================================================
# 1. Directory and File Path Configuration
# ==============================================================================
# Resolve the raw file from the repository root so the script runs from any directory
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(REPO_ROOT, "data", "raw", "psam_p48.csv")

if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"Could not locate '{file_path}'. "
        "Download it with 'python src/download_data.py' (see data/README.md)."
    )

print(f"[STATUS] Ingesting raw microdata file: {os.path.basename(file_path)}")

# ==============================================================================
# 2. Raw Dataset Ingestion
# ==============================================================================
# Ingest raw person-level records directly from Census CSV
df_raw = pd.read_csv(file_path)

# ==============================================================================
# 3. Structural Dimensions & Memory Audit
# ==============================================================================
print("\n" + "=" * 50)
print("RAW DATASET INTEGRITY CHECK")
print("=" * 50)

# Check survey volume to verify unaggregated raw Census microdata
n_rows, n_cols = df_raw.shape
print(f"Total Observations (Person Records): {n_rows:,}")
print(f"Total Recorded Attributes (Variables): {n_cols:,}")

# Audit memory consumption to monitor system scalability
ram_usage_mb = df_raw.memory_usage(deep=True).sum() / (1024**2)
print(f"Memory Footprint in RAM: {ram_usage_mb:.2f} MB")

# ==============================================================================
# 4. Schema & Variable Verification
# ==============================================================================
# Verify all project-critical demographic, labor, and target variables exist in the schema:
#   - PERNP: Total person's earnings (used to compute the 75th percentile target cutoff)
#   - WKHP:  Usual hours worked per week past 12 months (full-time filter: >= 35)
#   - ESR:   Employment status recode (civilian employed filter: ESR == 1)
#   - SEX:   Demographic sex identifier (focal variable for RQ1)
#   - AGEP:  Reported person age (experience control; life cohort analysis for RQ2)
#   - SCHL:  Educational attainment level (credential tier analysis for RQ1)
#   - COW:   Class of worker / employment sector (focal variable for RQ3)
#   - OCCP:  4-digit Standard Occupational Classification code (domain grouping)
target_and_core_vars = ["PERNP", "WKHP", "ESR", "SEX", "AGEP", "SCHL", "COW", "OCCP"]
missing_vars = [var for var in target_and_core_vars if var not in df_raw.columns]

print("-" * 50)
if missing_vars:
    print(f"[WARNING] Missing required variables: {missing_vars}")
else:
    print("[SUCCESS] All 8 required target and predictor variables verified.")
print("-" * 50)

# Preview top five records across key fields
print("\nSample Preview (First 5 records across core variables):")
print(df_raw[target_and_core_vars].head())