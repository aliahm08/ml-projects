# Titanic Data Pipeline & EDA

## Overview

This project implements a small but realistic **data pipeline** on the classic Titanic dataset.
It demonstrates:

- Loading raw CSV data
- Cleaning column names
- Handling missing values
- Feature engineering (family size, is_alone, title)
- Encoding categorical variables
- Selecting a clean feature set for downstream modeling
- Basic EDA (summary stats, survival breakdown, distributions)

This is meant as a portfolio-ready example of an **ETL + EDA workflow** for ML engineering.

---

## Project Structure

```text
titanic-data-pipeline/
  ├── data/
  │   ├── raw/
  │   │   └── titanic.csv
  │   └── processed/
  │       └── titanic_clean.csv
  ├── src/
  │   ├── etl.py
  │   └── eda.py
  ├── main.py
  ├── requirements.txt
  └── README.md
```

## Dataset

Go to Kaggle and download the Titanic dataset (train.csv). Save it as:

```
data/raw/titanic.csv
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full pipeline + EDA:

```bash
python main.py
```

This will:

- Load `data/raw/titanic.csv`
- Produce `data/processed/titanic_clean.csv`
- Print summary statistics and survival analysis
- Optionally create basic plots if configured
