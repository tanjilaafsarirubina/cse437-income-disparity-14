"""
CSE437 Final Project: Disparity and Error Analysis in Linear Income Classification
Script 07: Statistical Hypothesis Testing on Subgroup Predictive Disparities

Purpose:
    Formalizes the audit of the model's 0.00% high-earner prediction rate for women
    within the "High School or Some College" educational tier using standard hypothesis
    testing procedures aligned with course lecture notes:
      1. Test 1 (1-Sample z-test for Proportion):
         Evaluates whether the model's predicted rate for women (p_hat = 0.00%) is 
         significantly lower than their empirical test-set baseline (p_0 = 8.29%).
         Includes an exact binomial test companion check for the boundary case (0 successes).
      2. Test 2 (2-Sample z-test for Proportions):
         Evaluates whether the model's predicted high-earner rate for men (6.30%) 
         differs significantly from that of women (0.00%) holding education fixed.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

# ==============================================================================
# 1. Portable File Path Resolution
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
eval_file = os.path.join(BASE_DIR, "test_predictions_evaluated.csv")

if not os.path.exists(eval_file):
    BASE_DIR = os.getcwd()
    eval_file = os.path.join(BASE_DIR, "test_predictions_evaluated.csv")

if not os.path.exists(eval_file):
    raise FileNotFoundError(
        f"Could not locate 'test_predictions_evaluated.csv' in '{BASE_DIR}'. "
        "Please ensure '05_model_training_and_baseline_evaluation.py' has been executed."
    )

print(f"[STATUS] Loading evaluated test predictions from: {os.path.basename(eval_file)}")
df_eval = pd.read_csv(eval_file)

# ==============================================================================
# 2. Subgroup Isolation: High School or Some College Tier
# ==============================================================================
hs_df = df_eval[df_eval["SCHL_TIER"] == "HS_or_Some_College"].copy()

females = hs_df[hs_df["SEX_LABEL"] == "Female"]
males = hs_df[hs_df["SEX_LABEL"] == "Male"]

n_f = len(females)
n_m = len(males)

# Empirical ground-truth high earners in test partition
actual_high_f = females["HIGH_EARNER_ACTUAL"].sum()
p_actual_f = actual_high_f / n_f

# Model-predicted high earners
pred_high_f = females["PRED_HIGH_EARNER"].sum()
pred_high_m = males["PRED_HIGH_EARNER"].sum()
p_pred_f = pred_high_f / n_f
p_pred_m = pred_high_m / n_m

print("\n" + "=" * 65)
print("SAMPLE COUNTS & OBSERVED PROPORTIONS (HS OR SOME COLLEGE)")
print("=" * 65)
print(f"Females (n_f): {n_f:,}")
print(f"  - Actual High Earners:    {actual_high_f:>4} ({p_actual_f * 100:>5.2f}%)")
print(f"  - Predicted High Earners: {pred_high_f:>4} ({p_pred_f * 100:>5.2f}%)")
print("-" * 65)
print(f"Males   (n_m): {n_m:,}")
print(f"  - Predicted High Earners: {pred_high_m:>4} ({p_pred_m * 100:>5.2f}%)")
print("=" * 65)

# ==============================================================================
# 3. Test 1: 1-Sample z-Test for Proportion (Model vs. Empirical Baseline)
# ==============================================================================
# Framework (Course Lecture Notes, Section 5):
#   H0: p >= p_actual_f  (Model predicts at or above empirical baseline)
#   Ha: p <  p_actual_f  (Model systematically underpredicts high earners)
#   Direction: Left-tailed test
#   Formula: z = (p_hat - p_0) / sqrt(p_0 * (1 - p_0) / n)
print("\n" + "=" * 65)
print("TEST 1: 1-SAMPLE z-TEST (PREDICTED VS. EMPIRICAL BASELINE FOR WOMEN)")
print("=" * 65)

se_1 = np.sqrt(p_actual_f * (1 - p_actual_f) / n_f)
z_1 = (p_pred_f - p_actual_f) / se_1
p_val_1 = stats.norm.cdf(z_1)

# Exact binomial companion test (addressing zero-success boundary condition)
binom_res = stats.binomtest(k=int(pred_high_f), n=n_f, p=p_actual_f, alternative="less")

print(f"Null Hypothesis (H0) Benchmark (p_0):       {p_actual_f:.4f} (8.29%)")
print(f"Observed Sample Proportion (p_hat):        {p_pred_f:.4f} (0.00%)")
print(f"Standard Error (SE):                       {se_1:.4f}")
print(f"Test Statistic (z):                        {z_1:.4f}")
print(f"Asymptotic p-value (Normal Approximation): {p_val_1:.4e} (p < 0.001)")
print(f"Exact Binomial Companion p-value:          {binom_res.pvalue:.4e}")
print(
    f"Decision at alpha = 0.05:                  "
    f"{'Reject H0 (Statistically Significant Underprediction)' if p_val_1 < 0.05 else 'Fail to Reject H0'}"
)

# ==============================================================================
# 4. Test 2: 2-Sample z-Test for Proportions (Predicted Men vs. Women)
# ==============================================================================
# Framework (Course Lecture Notes, Section 8):
#   H0: p_m - p_f == 0  (Predicted high-earner rates do not differ by gender)
#   Ha: p_m - p_f != 0  (Predicted high-earner rates differ significantly)
#   Direction: Two-tailed test
#   Formula: z = (p_hat_1 - p_hat_2) / sqrt(p_c * (1 - p_c) * (1/n_1 + 1/n_2))
#   where p_c = (x_1 + x_2) / (n_1 + n_2)
print("\n" + "=" * 65)
print("TEST 2: 2-SAMPLE z-TEST (PREDICTED MEN VS. PREDICTED WOMEN)")
print("=" * 65)

p_c = (pred_high_m + pred_high_f) / (n_m + n_f)
se_2 = np.sqrt(p_c * (1 - p_c) * (1 / n_m + 1 / n_f))
z_2 = (p_pred_m - p_pred_f) / se_2

# Compute survival function directly to avoid floating point underflow (literal 0.0)
p_val_2 = 2 * stats.norm.sf(abs(z_2))

print(f"Pooled Proportion (p_c):                   {p_c:.4f}")
print(f"Predicted Proportion Men (p_hat_m):        {p_pred_m:.4f} (6.30%)")
print(f"Predicted Proportion Women (p_hat_f):      {p_pred_f:.4f} (0.00%)")
print(f"Standard Error (SE):                       {se_2:.4f}")
print(f"Test Statistic (z):                        {z_2:.4f}")
print(f"Two-Tailed p-value:                        {p_val_2:.4e} (p < 0.001)")
print(
    f"Decision at alpha = 0.05:                  "
    f"{'Reject H0 (Statistically Significant Gender Disparity)' if p_val_2 < 0.05 else 'Fail to Reject H0'}"
)
print("=" * 65)