from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def summarize_dataframe(df: pd.DataFrame) -> None:
    """Print high-level info and basic stats."""
    print("\n--- DataFrame Info ---")
    print(df.info())

    print("\n--- Head ---")
    print(df.head())

    print("\n--- Describe (numeric) ---")
    print(df.describe())

    print("\n--- Missing values per column ---")
    print(df.isna().sum())


def survival_breakdown(df: pd.DataFrame) -> None:
    """Show survival rates by some key features."""
    if "survived" not in df.columns:
        print("\nNo 'survived' column found; skipping survival analysis.")
        return

    print("\n--- Survival Rate Overall ---")
    print(df["survived"].value_counts(normalize=True))

    # Survival by sex (if we still have a 'sex_male' or similar)
    sex_cols = [c for c in df.columns if c.startswith("sex_")]
    if sex_cols:
        print("\n--- Survival by Sex (using encoded columns) ---")
        for col in sex_cols:
            print(f"\nSurvival when {col}=1")
            print(df[df[col] == 1]["survived"].value_counts(normalize=True))

    # Survival by family_size (if present)
    if "family_size" in df.columns:
        print("\n--- Survival by Family Size ---")
        print(
            df.groupby("family_size")["survived"]
            .mean()
            .sort_index()
        )


def plot_distributions(df: pd.DataFrame, output_dir: Optional[str] = None) -> None:
    """Generate a few basic distribution plots."""
    numeric_cols = ["age", "fare", "family_size"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    for col in numeric_cols:
        plt.figure()
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Distribution of {col}")
        if output_dir:
            plt.savefig(f"{output_dir}/{col}_distribution.png", bbox_inches="tight")
        else:
            plt.show()
        plt.close()

    if "survived" in df.columns and "age" in df.columns:
        plt.figure()
        sns.boxplot(x="survived", y="age", data=df)
        plt.title("Age vs Survived")
        if output_dir:
            plt.savefig(f"{output_dir}/age_vs_survived.png", bbox_inches="tight")
        else:
            plt.show()
        plt.close()


def run_eda(df: pd.DataFrame, output_dir: Optional[str] = None) -> None:
    """
    Run EDA steps:
    - Summary / schema
    - Survival breakdown
    - Basic plots
    """
    summarize_dataframe(df)
    survival_breakdown(df)
    plot_distributions(df, output_dir=output_dir)
