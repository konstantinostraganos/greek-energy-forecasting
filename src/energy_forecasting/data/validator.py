"""Data validation and cleaning pipeline.

Takes raw data from APIs and produces clean, hourly-aligned
datasets ready for feature engineering.
"""

import pandas as pd
from loguru import logger

from energy_forecasting.config import settings

class DataValidator:
    """Validates and cleans raw energy and weather data.

    Handles:
    - Resampling quarter-hourly data to hourly
    - Removing duplicates
    - Detecting and handling missing values
    - Flagging outliers
    - Aligning energy and weather on the same time range
    """

    # Reasonable bounds for Greek electricity demand (MW)
    LOAD_MIN = 2000
    LOAD_MAX = 12000

    # Reasonable bounds for Greek weather
    TEMP_MIN = -15.0
    TEMP_MAX = 50.0

    def __init__(self):
        """Initialize with paths from config."""

        self.raw_dir = settings.data.raw_dir
        self.processed_dir = settings.data.processed_dir

    def clean_energy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and resample energy data to hourly frequency.

        Steps:
        1. Remove duplicate timestamps
        2. Resample quarter-hourly data to hourly (mean)
        3. Flag outliers outside reasonable bounds
        4. Interpolate small gaps
        """

        logger.info(f"Cleaning energy data: {len(df)} raw rows")
        initial_rows = len(df)

        # Step 1: Remove duplicates
        dupes = df.timestamp.duplicated().sum()
        if dupes > 0:
            df = df.drop_duplicates(subset="timestamp", keep="first")
            logger.info(f"  Removed {dupes} duplicate timestamps")

        # Step 2: Set timestamp as index for resampling
        df = df.set_index("timestamp").sort_index()

        # Step 3: Resample to hourly — handles quarter-hourly data
        df = df.resample("1h").mean()
        logger.info(f"  Resampled to hourly: {len(df)} rows")

        # Step 4: Flag outliers
        for col in ["actual_load_mw", "forecast_load_mw"]:
            if col not in df.columns:
                continue

            outliers = (df[col] < self.LOAD_MIN) | (df[col] > self.LOAD_MAX)
            n_outliers = outliers.sum()

            if n_outliers > 0:
                logger.warning(
                    f"  {col}: {n_outliers} outliers outside "
                    f"[{self.LOAD_MIN}, {self.LOAD_MAX}] MW — set to NaN"
                )
                df.loc[outliers, col] = pd.NA

        # Step 5: Interpolate small gaps (up to 3 consecutive hours)
        for col in ["actual_load_mw", "forecast_load_mw"]:
            if col not in df.columns:
                continue

            missing_before = df[col].isna().sum()
            df[col] = df[col].interpolate(method="linear", limit=3)
            filled = missing_before - df[col].isna().sum()

            if filled > 0:
                logger.info(f"  {col}: interpolated {filled} missing values")

            remaining = df[col].isna().sum()
            if remaining > 0:
                logger.warning(f"  {col}: {remaining} values still missing")

        # Reset index — timestamp back to column
        df = df.reset_index()

        logger.info(
            f"  Energy cleaning complete: {initial_rows} → {len(df)} rows"
        )

        return df  

    def clean_weather(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean weather data.

        Steps:
        1. Remove duplicate timestamps
        2. Flag temperature outliers
        3. Ensure hourly frequency
        """

        logger.info(f"Cleaning weather data: {len(df)} raw rows")

        # Step 1: Remove duplicates
        dupes = df.timestamp.duplicated().sum()
        if dupes > 0:
            df = df.drop_duplicates(subset="timestamp", keep="first")
            logger.info(f"  Removed {dupes} duplicate timestamps")

        # Step 2: Set timestamp as index
        df = df.set_index("timestamp").sort_index()

        # Step 3: Ensure hourly frequency
        df = df.resample("1h").mean()
        logger.info(f"  Resampled to hourly: {len(df)} rows")

        # Step 4: Flag temperature outliers
        outliers = (df["temperature_c"] < self.TEMP_MIN) | (df["temperature_c"] > self.TEMP_MAX)
        n_outliers = outliers.sum()
        if n_outliers > 0:
            logger.warning(f"  Temperature: {n_outliers} outliers — set to NaN")
            df.loc[outliers, "temperature_c"] = pd.NA

        # Step 5: Interpolate small gaps
        for col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                df[col] = df[col].interpolate(method="linear", limit=3)
                filled = missing - df[col].isna().sum()
                logger.info(f"  {col}: interpolated {filled} missing values")

        df = df.reset_index()

        logger.info(f"  Weather cleaning complete: {len(df)} rows")

        return df

    def merge_and_save(
        self,
        energy: pd.DataFrame,
        weather: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge energy and weather data on timestamp and save.

        Aligns both datasets to their common time range
        and saves the result to processed directory.
        """

        logger.info("Merging energy and weather data")

        # Merge on timestamp — keep only hours that exist in both
        df = pd.merge(energy, weather, on="timestamp", how="inner")

        logger.info(
            f"  Merged: {len(df)} rows "
            f"({df.timestamp.min()} to {df.timestamp.max()})"
        )

        # Save to processed directory
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.processed_dir / "dataset.parquet"
        df.to_parquet(filepath, index=False)

        size_mb = filepath.stat().st_size / (1024 * 1024)
        logger.info(f"  Saved to {filepath} ({size_mb:.1f} MB)")

        return df              