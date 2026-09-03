"""
CSE437 Final Project: Disparity and Error Analysis in Linear Income Classification
Script 08: Visualizations for Model Results, Subgroup Disparities, and Errors
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
eval_file = os.path.join(BASE_DIR, "test_predictions_evaluated.csv")
cow_file = os.path.join(BASE_DIR, "rq2_cow_errors.csv")

df_eval = pd.read_csv(eval_file)
cow_df = pd.read_csv(cow_file)

# -------------------------------------------------------------
# Figure 1: RQ1 Predicted vs Actual High-Earner Gap by Education
# -------------------------------------------------------------
edu_order = ["Less_than_HS", "HS_or_Some_College", "Bachelors", "Graduate_Plus"]
pred_rates = df_eval.groupby(["SCHL_TIER", "SEX_LABEL"])["PRED_HIGH_EARNER"].mean().unstack().reindex(edu_order) * 100
actual_rates = df_eval.groupby(["SCHL_TIER", "SEX_LABEL"])["HIGH_EARNER_ACTUAL"].mean().unstack().reindex(edu_order) * 100

x = np.arange(len(edu_order))
width = 0.20

fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
ax.bar(x - 1.5*width, actual_rates["Male"], width, label="Actual Men", color="#4a7bb7", alpha=0.7)
ax.bar(x - 0.5*width, actual_rates["Female"], width, label="Actual Women", color="#e8765c", alpha=0.7)
ax.bar(x + 0.5*width, pred_rates["Male"], width, label="Predicted Men", color="#1f4e79")
ax.bar(x + 1.5*width, pred_rates["Female"], width, label="Predicted Women", color="#c00000")

ax.set_ylabel("High-Earner Proportion (%)", fontsize=11, fontweight="bold")
ax.set_title("RQ1: Threshold Amplification & Female Subgroup Erasure Across Education", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(["Less than HS", "HS / Some College", "Bachelor's", "Graduate+"], fontsize=10)
ax.legend(frameon=True)
ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
fig_1_path = os.path.join(BASE_DIR, "fig1_rq1_education_disparity.png")
plt.savefig(fig_1_path)
print(f"[SUCCESS] Saved: {fig_1_path}")

# -------------------------------------------------------------
# Figure 2: RQ2 False Negative Rate by Class of Work
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
sorted_cow = cow_df.sort_values(by="FNR_%", ascending=True)

bars = ax.barh(sorted_cow["COW_GROUP"], sorted_cow["FNR_%"], color="#2f5597")
ax.set_xlabel("False Negative Rate (FNR %)", fontsize=11, fontweight="bold")
ax.set_title("RQ2: False Negative Rate by Class of Work (High-Earner Omission)", fontsize=12, fontweight="bold")
ax.grid(axis="x", linestyle="--", alpha=0.5)

for bar in bars:
    w = bar.get_width()
    ax.text(w + 1, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va="center", fontsize=9)

plt.tight_layout()
fig_2_path = os.path.join(BASE_DIR, "fig2_rq2_sector_fnr.png")
plt.savefig(fig_2_path)
print(f"[SUCCESS] Saved: {fig_2_path}")