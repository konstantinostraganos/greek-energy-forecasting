"""Run the data validation and cleaning pipeline."""

import pandas as pd
from energy_forecasting.config import settings
from energy_forecasting.data.validator import DataValidator
from energy_forecasting.logger import logger

# Load raw data
logger.info("Loading raw data")
energy = pd.read_parquet(settings.data.raw_dir / "entso_e_load.parquet")
weather = pd.read_parquet(settings.data.raw_dir / "weather.parquet")

# Validate and clean
validator = DataValidator()
energy_clean = validator.clean_energy(energy)
weather_clean = validator.clean_weather(weather)

# Merge and save
df = validator.merge_and_save(energy_clean, weather_clean)

# Final summary
print(f"\n=== FINAL DATASET ===")
print(f"Shape: {df.shape}")
print(f"Period: {df.timestamp.min()} to {df.timestamp.max()}")
print(f"Nulls:\n{df.isnull().sum()}")
print(f"\nFirst rows:\n{df.head()}")