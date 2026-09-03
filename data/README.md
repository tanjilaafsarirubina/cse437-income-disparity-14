# Data Sources & Provenance

## 1. Raw Dataset (Exceeds 50 MB Limit)

- **Dataset Name:** U.S. Census Bureau American Community Survey (ACS) 2023 1-Year PUMS (Texas Person Records)
- **Direct Portal Link:** https://www.census.gov/programs-surveys/acs/microdata.html
- **File Name:** `psam_p48.csv`
- **File Size:** ~1.1 GB uncompressed (286,878 rows, 288 columns)
- **Licence:** Public domain U.S. Government data (13 U.S.C. § 9)

### How to Obtain the Raw Data

Per project guidelines, datasets exceeding 50 MB are hosted externally:

- **Public Google Drive Folder:** https://drive.google.com/drive/folders/1E6GYPV0siUHCq2ohG6EdfZky0AJJOXd3?usp=sharing

To run the raw preprocessing pipeline from scratch:

1. Download `psam_p48.csv` from the Google Drive link above.
2. Place `psam_p48.csv` directly inside the `data/raw/` directory.
3. Alternatively, run the automated downloader:
```bash
   python src/download_data.py
```

## 2. Processed Dataset (Committed < 50 MB)

- **File Name:** `data/processed/texas_cleaned_30k.csv`
- **File Size:** ~2.6 MB (30,000 rows, 12 columns)
- **Generation:** Produced by `notebooks/02_preprocessing.ipynb`
- **Description:** A stratified subsample (`random_state=42`) of actively employed civilian full-time workers (`ESR == 1`, `WKHP >= 35`, `PERNP > 0`, ages 16–80) with mapped 12 SOC occupational domains, decoded sectors, and the binary top-quartile target `HIGH_EARNER` (threshold: $90,000.00).
- **Usage:** Committed directly to GitHub to allow reproduction of notebooks `03` through `05` without requiring the 1.1 GB raw Census download.
