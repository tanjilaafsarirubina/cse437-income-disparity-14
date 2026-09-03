"""
CSE437 Final Project: Disparity and Error Analysis in Linear Income Classification
Script 02: Population Cohort Filtering and Target Variable Construction

Purpose:
    Isolates the target study population (civilian full-time workers actively at work)
    from the Texas ACS 1-Year PUMS dataset and derives the binary classification 
    target (HIGH_EARNER) using the 75th percentile empirical earnings threshold ($90,000.00).
"""

import os
import pandas as pd

# ==============================================================================
# 1. Portable File Path Configuration
# ==============================================================================
# Resolve paths dynamically relative to script location for grader reproducibility
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "psam_p48.csv")

# Fallback to current working directory if script executed from root
if not os.path.exists(file_path):
    BASE_DIR = os.getcwd()
    file_path = os.path.join(BASE_DIR, "psam_p48.csv")

if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"Could not locate 'psam_p48.csv' in '{BASE_DIR}'. "
        "Please place the raw Census Texas CSV file in the working directory."
    )

# ==============================================================================
# 2. Memory-Optimized Ingestion of Candidate Variables
# ==============================================================================
# Load only core demographic, labor, and economic variables to minimize RAM usage:
#   - PERNP: Total person's annual earnings (target foundation)
#   - WKHP:  Usual weekly hours worked past 12 months (full-time filter)
#   - ESR:   Employment status recode (civilian labor filter)
#   - AGEP:  Age (experience control)
#   - SEX:   Demographic sex identifier (RQ1)
#   - SCHL:  Educational attainment tier (RQ1)
#   - MAR:   Marital status control
#   - COW:   Class of worker / employment sector (RQ2)
#   - OCCP:  4-digit SOC occupation code
#   - WKWN:  Weeks worked (inspected for labor attachment, omitted downstream for parsimony)
cols_to_use = [
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

print(f"[STATUS] Loading selected attributes from: {os.path.basename(file_path)}")
df = pd.read_csv(file_path, usecols=cols_to_use)
print(f"Initial raw subset shape: {df.shape[0]:,} rows, {df.shape[1]} columns")

# ==============================================================================
# 3. Methodological Cohort Filtering
# ==============================================================================
# Restrict population to active, full-time civilian wage and salary / self-employed workers:
#   1. ESR == 1:  Civilian employed, at work (excludes unemployed, military, and absent)
#   2. WKHP >= 35: Standard U.S. Bureau of Labor Statistics full-time employment benchmark
#   3. PERNP > 0: Positive annual earnings (excludes unpaid or zero-earning labor)
df_filtered = df[(df["ESR"] == 1) & (df["WKHP"] >= 35) & (df["PERNP"] > 0)].copy()

print(f"Filtered full-time workforce shape: {df_filtered.shape[0]:,} rows, {df_filtered.shape[1]} columns")

# ==============================================================================
# 4. Target Variable Construction (75th Percentile Cutoff)
# ==============================================================================
# Define top-quartile earnings threshold on the filtered full-time working population
threshold_75 = df_filtered["PERNP"].quantile(0.75)
median_earnings = df_filtered["PERNP"].median()

print("\n" + "=" * 55)
print("LABOR EARNINGS DISTRIBUTION (FULL-TIME CIVILIAN WORKERS)")
print("=" * 55)
print(f"Minimum Earnings:           ${df_filtered['PERNP'].min():>12,.2f}")
print(f"Median Earnings (P50):      ${median_earnings:>12,.2f}")
print(f"Top Quartile Cutoff (P75):  ${threshold_75:>12,.2f}")
print(f"Maximum Earnings:           ${df_filtered['PERNP'].max():>12,.2f}")
print("-" * 55)

# Binarize target: 1 = Top Quartile (High Earner), 0 = Standard Earner
df_filtered["HIGH_EARNER"] = (df_filtered["PERNP"] >= threshold_75).astype(int)

# Verify empirical class distribution for classification setup
balance = df_filtered["HIGH_EARNER"].value_counts(normalize=True) * 100
print("\nTarget Class Distribution (HIGH_EARNER):")
print(f"  Class 0 (Standard Earners < $90,000): {balance[0]:.2f}%")
print(f"  Class 1 (Top Quartile Earners >= $90,000): {balance[1]:.2f}%")
print("=" * 55)