"""
Pipeline Integrity Runner
Executes scripts 01 through 07 in order and checks that all artifacts are produced.
"""

import os
import subprocess
import sys
import time

SCRIPTS = [
    ("01_data_inspection_and_validation.py", []),
    ("02_cohort_filtering_and_target_definition.py", []),
    ("03_data_preprocessing_and_subsampling.py", ["texas_cleaned_30k.csv"]),
    (
        "04_feature_engineering_and_data_splitting.py",
        [
            "X_train.csv",
            "X_test.csv",
            "y_train.csv",
            "y_test.csv",
            "test_metadata.csv",
        ],
    ),
    (
        "05_model_training_and_baseline_evaluation.py",
        ["test_predictions_evaluated.csv"],
    ),
    (
        "06_subgroup_disparity_and_error_auditing.py",
        [
            "rq1_education_gap.csv",
            "rq1_actual_ground_truth.csv",
            "rq1_age_gap.csv",
            "rq2_cow_errors.csv",
        ],
    ),
    ("07_statistical_hypothesis_testing.py", []),
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
python_bin = sys.executable

print("=" * 60)
print("STARTING FULL REPRODUCIBILITY AUDIT")
print("=" * 60)

for script_name, expected_outputs in SCRIPTS:
    script_path = os.path.join(BASE_DIR, script_name)
    if not os.path.exists(script_path):
        print(f"[FAIL] Script not found: {script_name}")
        sys.exit(1)

    print(f"\n[RUNNING] {script_name}...")
    start_time = time.time()
    result = subprocess.run([python_bin, script_path], capture_output=True, text=True)
    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"[ERROR] {script_name} crashed (Exit code {result.returncode}):")
        print(result.stderr)
        sys.exit(1)

    print(f"[SUCCESS] {script_name} finished in {elapsed:.2f}s.")

    for out_file in expected_outputs:
        out_path = os.path.join(BASE_DIR, out_file)
        if os.path.exists(out_path):
            size_kb = os.path.getsize(out_path) / 1024
            print(f"   -> Verified artifact: {out_file} ({size_kb:.1f} KB)")
        else:
            print(f"   -> [MISSING] Expected output not found: {out_file}")
            sys.exit(1)

print("\n" + "=" * 60)
print("PIPELINE AUDIT COMPLETE: ALL 7 SCRIPTS EXECUTED FLAWLESSLY!")
print("=" * 60)