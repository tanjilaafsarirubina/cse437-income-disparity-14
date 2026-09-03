"""
CSE437 Final Project: Disparity and Error Analysis in Linear Income Classification
Script: download_data.py (Located in src/)

Purpose:
    Automates the acquisition of the large Census raw microdata file (psam_p48.csv)
    from the public Google Drive folder into the data/raw/ directory.
"""

import os
import sys
from pathlib import Path

try:
    import gdown
except ImportError:
    print("[ERROR] 'gdown' package is missing.")
    print("Install dependencies using: pip install -r requirements.txt")
    sys.exit(1)

# Resolve paths relative to project root
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
TARGET_RAW_DIR = PROJECT_ROOT / "data" / "raw"
TARGET_FILE = TARGET_RAW_DIR / "psam_p48.csv"

# Public Google Drive Folder Link for Group 14
GDRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1E6GYPV0siUHCq2ohG6EdfZky0AJJOXd3?usp=sharing"
)

TARGET_RAW_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("ACQUIRING RAW CENSUS MICRODATA FROM PUBLIC GOOGLE DRIVE")
print("=" * 65)
print(f"Target Destination: {TARGET_RAW_DIR}")
print(f"Source Folder:      {GDRIVE_FOLDER_URL}")
print("-" * 65)

if TARGET_FILE.exists() and TARGET_FILE.stat().st_size > 1024 * 1024 * 100:
    print(f"[STATUS] '{TARGET_FILE.name}' is already present ({TARGET_FILE.stat().st_size / (1024*1024):.1f} MB).")
    print("Download skipped.")
else:
    try:
        # download_folder handles Google Drive virus warnings on large files automatically
        gdown.download_folder(
            url=GDRIVE_FOLDER_URL,
            output=str(TARGET_RAW_DIR),
            quiet=False,
            use_cookies=False,
        )
        print("\n" + "=" * 65)
        print("[SUCCESS] Raw dataset successfully placed in data/raw/!")
        print("=" * 65)
    except Exception as exc:
        print(f"\n[ERROR] Failed to download via gdown: {exc}")
        print("Please verify your internet connection or download 'psam_p48.csv' manually")
        print(f"from: {GDRIVE_FOLDER_URL}")
        sys.exit(1)