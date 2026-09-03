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