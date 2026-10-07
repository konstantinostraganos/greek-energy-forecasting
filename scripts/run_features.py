"""Run the feature engineering pipeline."""

import pandas as pd
from energy_forecasting.config import settings
from energy_forecasting.data.feature_engineer import FeatureEngineer
from energy_forecasting.logger import logger

# Load processed data
logger.info("Loading processed dataset")
df = pd.read_parquet(settings.data.processed_dir / "dataset.parquet")

# Build features
engineer = FeatureEngineer()
df = engineer.build_features(df)

# Summary
print(f"\n=== FEATURE DATASET ===")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst rows:")
print(df.head())