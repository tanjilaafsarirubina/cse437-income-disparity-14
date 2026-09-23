"""
Pipeline Integrity Runner
Executes scripts 01 through 08 in order and checks that all artifacts are produced.

Usage:
    python scripts/run_all_check.py                   # full run; needs data/raw/psam_p48.csv
    python scripts/run_all_check.py --from-processed  # starts at script 04 from the committed
                                                      # data/processed/texas_cleaned_30k.csv
"""

import argparse
import os
import subprocess
import sys
import time

# (script, artifacts it must produce, relative to the repository root)
SCRIPTS = [
    ("01_data_inspection_and_validation.py", []),
    ("02_cohort_filtering_and_target_definition.py", []),
    ("03_data_preprocessing_and_subsampling.py", ["data/processed/texas_cleaned_30k.csv"]),
    (
        "04_feature_engineering_and_data_splitting.py",
        [
            "data/processed/X_train.csv",
            "data/processed/X_test.csv",
            "data/processed/y_train.csv",
            "data/processed/y_test.csv",
            "data/processed/test_metadata.csv",
        ],
    ),
    (
        "05_model_training_and_baseline_evaluation.py",
        [
            "data/processed/test_predictions_evaluated.csv",
            "data/processed/model_family_comparison.csv",
        ],
    ),
    (
        "06_subgroup_disparity_and_error_auditing.py",
        [
            "data/processed/rq1_education_gap.csv",
            "data/processed/rq1_actual_ground_truth.csv",
            "data/processed/rq2_age_gap.csv",
            "data/processed/rq3_cow_errors.csv",
        ],
    ),
    ("07_statistical_hypothesis_testing.py", []),
    (
        "08_visualizations.py",
        [
            "figures/fig1_rq1_education_disparity.png",
            "figures/fig2_rq3_sector_fnr.png",
        ],
    ),
]

parser = argparse.ArgumentParser(description=__doc__.splitlines()[2])
parser.add_argument(
    "--from-processed",
    action="store_true",
    help="skip scripts 01-03, which need the raw Census file in data/raw/",
)
args = parser.parse_args()

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPTS_DIR)
python_bin = sys.executable
to_run = SCRIPTS[3:] if args.from_processed else SCRIPTS

print("=" * 60)
print("STARTING FULL REPRODUCIBILITY AUDIT")
print("=" * 60)

for script_name, expected_outputs in to_run:
    script_path = os.path.join(SCRIPTS_DIR, script_name)
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
        out_path = os.path.join(REPO_ROOT, out_file)
        if os.path.exists(out_path):
            size_kb = os.path.getsize(out_path) / 1024
            print(f"   -> Verified artifact: {out_file} ({size_kb:.1f} KB)")
        else:
            print(f"   -> [MISSING] Expected output not found: {out_file}")
            sys.exit(1)

print("\n" + "=" * 60)
print(f"PIPELINE AUDIT COMPLETE: {len(to_run)} SCRIPTS RAN AND ALL ARTIFACTS WERE FOUND")
print("=" * 60)
