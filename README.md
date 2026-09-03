# CSE437 Final Project: Disparity and Error Analysis in Linear Income Classification
**An Audit of 2023 Texas ACS PUMS Microdata**

- **Course:** CSE437 Data Science, Summer 2026
- **Repository:** `cse437-income-disparity-14`
- **Group 14 Members:**
  - Tanjila Afsari Rubina (Student ID: 24241310)[cite: 10]
  - Sandip Kumar Paul (Student ID: 24241311)[cite: 10]

---

## 1. Problem Statement
Algorithmic classification systems deployed in labor analytics, hiring screening, and credit underwriting commonly utilize linear decision boundaries for their interpretability and operational simplicity[cite: 10]. However, when applied to skewed socioeconomic microdata, additive linear weights can interact with historical wage distributions to create severe localized disparities[cite: 10]. This study audits top-quartile income classification to identify where and why linear models fail across demographic intersections (gender, educational attainment, age) and institutional employment sectors[cite: 10].

## 2. Dataset Description
- **Source:** U.S. Census Bureau, 2023 American Community Survey (ACS) 1-Year Public Use Microdata Sample (PUMS) for Texas (`psam_p48.csv`)[cite: 10].
- **Working Link:** [U.S. Census Bureau ACS PUMS Microdata Portal](https://www.census.gov/programs-surveys/acs/microdata.html)[cite: 10]
- **Cohort Filter:** Restricted to actively employed civilian workers (`ESR = 1`) with full-time status (`WKHP >= 35`) and positive earnings (`PERNP > 0`) aged 16–80[cite: 10].
- **Target Variable (`HIGH_EARNER`):** Binary classification target representing the empirical 75th percentile earnings threshold ($90,000.00)[cite: 10].
  - **Class 0 (Standard Earner, < $90,000):** 73.95% of filtered population[cite: 10].
  - **Class 1 (Top-Quartile Earner, >= $90,000):** 26.05% of filtered population[cite: 10].

## 3. The Three Research Questions
1. **RQ1 (Educational Attainment & Gender Disparity):** How do model-predicted high-earner rates diverge between men and women across educational attainment tiers, and does the linear classifier produce subgroup erasure for lower-credentialed women[cite: 10]?
2. **RQ2 (Age Cohort Trajectories):** How does the predicted high-earner gender gap evolve across career stages from early-career (ages 16–29) to late-career (ages 60–80)[cite: 10]?
3. **RQ3 (Institutional Sector Error Profiles):** How do classification errors (False Negative Rate and False Positive Rate) distribute across Class of Work (COW) sectors, and what data artifacts drive high-error sectors[cite: 10]?

---

## 4. Repository Structure
```text
cse437-income-disparity-14/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                 # Contains psam_p48.csv (downloaded from Google Drive)
│   ├── processed/           # Contains texas_cleaned_30k.csv (committed <50 MB)
│   └── README.md            # Data provenance, download links, and descriptions
│
├── notebooks/
│   ├── 01_data_audit_and_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling_and_tuning.ipynb
│   └── 05_evaluation_and_error_analysis.ipynb
│
├── src/
│   └── download_data.py     # Automated Google Drive downloader utility
│
├── models/
│   ├── linear_svc_tuned.joblib
│   └── logistic_regression_baseline.joblib
│
├── figures/
│   ├── fig1_rq1_education_disparity.png
│   └── fig2_rq2_sector_fnr.png
│
└── report/
    ├── report.pdf           # Formal 10-page project report
    └── report.md            # Markdown version of report

## 5. Setup & Execution Instructions
### 1. Environment Setup

Clone the repository and install required packages:
git clone [https://github.com/](https://github.com/)<your_username>/cse437-income-disparity-14.git
cd cse437-income-disparity-14
pip install -r requirements.txt


## Dataset Acquisition
The preprocessed 30,000-record dataset (data/processed/texas_cleaned_30k.csv, ~2.6 MB) is included directly in this repository so notebooks 03 through 05 run out-of-the-box.  The raw Census microdata (psam_p48.csv, ~1.1 GB) exceeds the 50 MB GitHub commit threshold and is hosted on Google Drive:Dataset Google Drive Link: https://drive.google.com/drive/folders/<your_folder_id>?usp=sharing[cite: 10]To fetch it automatically into data/raw/, run:

python src/download_data.py

## Running the Notebooks
Execute the notebooks in numerical order from top to bottom on a clean kernel:
notebooks/01_data_audit_and_eda.ipynb  
notebooks/02_preprocessing.ipynb  
notebooks/03_feature_engineering.ipynb  
notebooks/04_modeling_and_tuning.ipynb  
notebooks/05_evaluation_and_error_analysis.ipynb

