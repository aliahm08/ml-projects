from pathlib import Path

from src.etl import run_pipeline
from src.eda import run_eda


def main() -> None:
    project_root = Path(__file__).parent
    raw_path = project_root / "data" / "raw" / "titanic.csv"
    processed_path = project_root / "data" / "processed" / "titanic_clean.csv"

    # 1. Run ETL pipeline
    print("🔧 Running ETL pipeline...")
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean = run_pipeline(raw_path, processed_path)
    print(f"✅ ETL complete. Clean data saved to: {processed_path}")
    print(f"Clean shape: {df_clean.shape}")

    # 2. Run EDA
    print("\n📊 Running EDA...")
    run_eda(df_clean)
    print("✅ EDA complete.")


if __name__ == "__main__":
    main()
