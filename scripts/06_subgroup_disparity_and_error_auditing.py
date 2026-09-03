"""
CSE437 Final Project: Disparity and Error Analysis in Linear Income Classification
Script 06: Subgroup Disparity Auditing and Sectoral Error Evaluation (RQ1 & RQ2)

Purpose:
    Executes post-hoc disaggregated model auditing on held-out test predictions:
      1. Evaluates Research Question 1 (RQ1):
         - Measures predicted high-earner rates across Education Tiers partitioned by Gender.
         - Compares predicted rates against empirical test set ground truth to quantify 
           decision threshold amplification and subgroup erasure.
         - Computes predicted high-earner rates across five working-age cohorts (16-80).
      2. Evaluates Research Question 2 (RQ2):
         - Computes subgroup-sliced classification error metrics across Class of Work (COW):
           Total Error Rate (%), False Positive Rate (FPR %), and False Negative Rate (FNR %).
         - Exports structured evaluation tables for tabular presentation in the final report.
"""

import os
import numpy as np
import pandas as pd

# ==============================================================================
# 1. Portable File Path Resolution
# ==============================================================================
# Resolve paths relative to script location for reproducible grading
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
eval_file = os.path.join(BASE_DIR, "test_predictions_evaluated.csv")

if not os.path.exists(eval_file):
    BASE_DIR = os.getcwd()
    eval_file = os.path.join(BASE_DIR, "test_predictions_evaluated.csv")

if not os.path.exists(eval_file):
    raise FileNotFoundError(
        f"Could not locate 'test_predictions_evaluated.csv' in '{BASE_DIR}'. "
        "Please run '05_model_training_and_baseline_evaluation.py' first."
    )

print(f"[STATUS] Ingesting evaluated test set from: {os.path.basename(eval_file)}")
df_eval = pd.read_csv(eval_file)

# ==============================================================================
# 2. Research Question 1: Gender Disparities Across Education & Age Cohorts
# ==============================================================================
# Discretize continuous age into standard demographic labor force brackets
age_bins = [15, 29, 39, 49, 59, 81]
age_labels = ["16-29", "30-39", "40-49", "50-59", "60-80"]
df_eval["AGE_BRACKET"] = pd.cut(
    df_eval["AGEP"], bins=age_bins, labels=age_labels, right=True
)

# ------------------------------------------------------------------------------
# RQ1 (Part A): Predicted vs. Empirical High-Earner Rates Across Education
# ------------------------------------------------------------------------------
edu_order = ["Less_than_HS", "HS_or_Some_College", "Bachelors", "Graduate_Plus"]

# Compute predicted classification rates (P(Y_pred = 1 | Tier, Sex))
pred_by_edu = (
    df_eval.groupby(["SCHL_TIER", "SEX_LABEL"])["PRED_HIGH_EARNER"]
    .mean()
    .unstack()
    .reindex(edu_order)
)
pred_by_edu["Gender_Gap (M - F)"] = pred_by_edu["Male"] - pred_by_edu["Female"]

# Compute empirical ground truth rates (P(Y_actual = 1 | Tier, Sex)) for calibration auditing
actual_by_edu = (
    df_eval.groupby(["SCHL_TIER", "SEX_LABEL"])["HIGH_EARNER_ACTUAL"]
    .mean()
    .unstack()
    .reindex(edu_order)
)
actual_by_edu["Actual_Gap (M - F)"] = (
    actual_by_edu["Male"] - actual_by_edu["Female"]
)

# ------------------------------------------------------------------------------
# RQ1 (Part B): Predicted High-Earner Trajectories Across Life Cohorts
# ------------------------------------------------------------------------------
pred_by_age = (
    df_eval.groupby(["AGE_BRACKET", "SEX_LABEL"])["PRED_HIGH_EARNER"]
    .mean()
    .unstack()
)
pred_by_age["Gender_Gap (M - F)"] = pred_by_age["Male"] - pred_by_age["Female"]

print("\n" + "=" * 65)
print("RQ1: PREDICTED HIGH-EARNER PROPORTIONS BY EDUCATION & GENDER (%)")
print("=" * 65)
print((pred_by_edu * 100).round(2).to_string())

print("\n" + "-" * 65)
print("RQ1: EMPIRICAL GROUND TRUTH PROPORTIONS IN TEST SAMPLE (%)")
print("-" * 65)
print((actual_by_edu * 100).round(2).to_string())

print("\n" + "=" * 65)
print("RQ1: PREDICTED HIGH-EARNER PROPORTIONS ACROSS AGE COHORTS (%)")
print("=" * 65)
print((pred_by_age * 100).round(2).to_string())
print("=" * 65)

# ==============================================================================
# 3. Research Question 2: Error Disparities Across Class of Work (COW)
# ==============================================================================
def compute_cow_metrics(group: pd.DataFrame) -> pd.Series:
    """
    Computes class-specific and subgroup error distributions for an employment sector:
      - Total Error Rate: (FP + FN) / Total Instances
      - False Positive Rate (FPR): FP / Actual Standard Earners (False Alarm rate)
      - False Negative Rate (FNR): FN / Actual High Earners (Omission rate)
    """
    total = len(group)
    actual_high = (group["HIGH_EARNER_ACTUAL"] == 1).sum()
    actual_std = (group["HIGH_EARNER_ACTUAL"] == 0).sum()

    fp = (
        (group["HIGH_EARNER_ACTUAL"] == 0)
        & (group["PRED_HIGH_EARNER"] == 1)
    ).sum()
    fn = (
        (group["HIGH_EARNER_ACTUAL"] == 1)
        & (group["PRED_HIGH_EARNER"] == 0)
    ).sum()
    errors = fp + fn

    fpr = (fp / actual_std * 100) if actual_std > 0 else np.nan
    fnr = (fn / actual_high * 100) if actual_high > 0 else np.nan
    error_rate = (errors / total) * 100

    return pd.Series(
        {
            "Test_N": int(total),
            "Actual_High_N": int(actual_high),
            "Error_Rate_%": round(error_rate, 2),
            "FPR_%": round(fpr, 2),
            "FNR_%": round(fnr, 2),
        }
    )


# Apply metric calculation across institutional sectors, ranked by overall error rate
cow_table = (
    df_eval.groupby("COW_GROUP")
    .apply(compute_cow_metrics, include_groups=False)
    .sort_values(by="Error_Rate_%", ascending=False)
)

print("\n" + "=" * 65)
print("RQ2: CLASSIFICATION ERROR PROFILES BY CLASS OF WORK (COW)")
print("=" * 65)
print(cow_table.to_string())
print("=" * 65)

# ==============================================================================
# 4. Structured Evaluation Export
# ==============================================================================
# Save structured artifacts for reporting and viva verification
pred_by_edu.to_csv(os.path.join(BASE_DIR, "rq1_education_gap.csv"))
actual_by_edu.to_csv(os.path.join(BASE_DIR, "rq1_actual_ground_truth.csv"))
pred_by_age.to_csv(os.path.join(BASE_DIR, "rq1_age_gap.csv"))
cow_table.to_csv(os.path.join(BASE_DIR, "rq2_cow_errors.csv"))

print(f"\n[SUCCESS] All evaluation tables successfully exported to: {BASE_DIR}")