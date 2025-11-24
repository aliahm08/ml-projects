from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


PathLike = Union[str, Path]


def load_raw_data(path: PathLike) -> pd.DataFrame:
    """Load raw Titanic data from a CSV file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found at: {path}")
    df = pd.read_csv(path)
    return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Convert column names to snake_case."""
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(".", "_", regex=False)
    )
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values with simple, explainable rules."""
    df = df.copy()

    # Embarked: fill with most frequent
    if "embarked" in df.columns:
        most_common_embarked = df["embarked"].mode(dropna=True)[0]
        df["embarked"] = df["embarked"].fillna(most_common_embarked)

    # Age: fill with median age per (sex, pclass) group if possible
    if "age" in df.columns and "sex" in df.columns and "pclass" in df.columns:
        df["age"] = df["age"].astype(float)

        def fill_age(row: pd.Series) -> float:
            if not np.isnan(row["age"]):
                return row["age"]
            mask = (df["sex"] == row["sex"]) & (df["pclass"] == row["pclass"])
            group_median = df.loc[mask, "age"].median()
            if np.isnan(group_median):
                return df["age"].median()
            return group_median

        df["age"] = df.apply(fill_age, axis=1)

    # Cabin: often very sparse, we might drop it in feature selection step
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional features useful for modeling/analysis."""
    df = df.copy()

    # Family size
    if {"sibsp", "parch"}.issubset(df.columns):
        df["family_size"] = df["sibsp"] + df["parch"] + 1

    # Is alone
    if "family_size" in df.columns:
        df["is_alone"] = (df["family_size"] == 1).astype(int)

    # Title extracted from name (Mr, Mrs, Miss, etc.)
    if "name" in df.columns:
        df["title"] = (
            df["name"]
            .str.extract(r",\s*([^\.]+)\.", expand=False)
            .str.strip()
        )

        # Map rare titles
        title_counts = df["title"].value_counts()
        rare_titles = title_counts[title_counts < 10].index
        df["title"] = df["title"].replace(rare_titles, "Other")

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical variables using simple mappings / one-hot encoding."""
    df = df.copy()

    # Binary encoding for sex
    if "sex" in df.columns:
        df["sex_male"] = (df["sex"] == "male").astype(int)
        df["sex_female"] = (df["sex"] == "female").astype(int)

    # One-hot encode embarked
    if "embarked" in df.columns:
        embarked_dummies = pd.get_dummies(df["embarked"], prefix="embarked")
        df = pd.concat([df, embarked_dummies], axis=1)

    # One-hot encode title (if created)
    if "title" in df.columns:
        title_dummies = pd.get_dummies(df["title"], prefix="title")
        df = pd.concat([df, title_dummies], axis=1)

    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep a clean subset of columns for downstream modeling/analysis.

    This is where you decide what the "clean" schema is.
    """
    df = df.copy()

    target_col = "survived" if "survived" in df.columns else None

    numeric_cols = [
        col
        for col in ["age", "fare", "pclass", "sibsp", "parch", "family_size", "is_alone"]
        if col in df.columns
    ]
    encoded_cols = [c for c in df.columns if c.startswith(("sex_", "embarked_", "title_"))]

    keep_cols = []
    if target_col:
        keep_cols.append(target_col)
    keep_cols.extend(numeric_cols)
    keep_cols.extend(encoded_cols)

    # Drop duplicates and rows with missing target
    if target_col:
        df = df.dropna(subset=[target_col]).drop_duplicates()

    return df[keep_cols]


def save_processed_data(df: pd.DataFrame, path: PathLike) -> None:
    """Save cleaned data to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_pipeline(raw_path: PathLike, processed_path: PathLike) -> pd.DataFrame:
    """
    Full ETL pipeline:
    - Load raw
    - Clean column names
    - Handle missing values
    - Engineer features
    - Encode categoricals
    - Select final feature set
    - Save processed data
    """
    df_raw = load_raw_data(raw_path)
    df = clean_column_names(df_raw)
    df = handle_missing_values(df)
    df = engineer_features(df)
    df = encode_categoricals(df)
    df_clean = select_features(df)
    save_processed_data(df_clean, processed_path)
    return df_clean
