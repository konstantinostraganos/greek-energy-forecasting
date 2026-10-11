"""Feature engineering pipeline.

Transforms the clean dataset into a feature-rich dataset
ready for model training. Creates temporal, weather,
lag, and calendar features.
"""

import pandas as pd
import numpy as np
import holidays
from loguru import logger

from energy_forecasting.config import settings

class FeatureEngineer:
    """Creates features for electricity demand forecasting.

    Feature groups:
    1. Temporal — hour, day, month, season
    2. Calendar — holidays, weekend
    3. Lag — past demand values
    4. Rolling — moving averages and stats
    5. Weather — HDD, CDD, weather lags
    """

    # Greek comfort zone threshold (°C)
    COMFORT_TEMP = 18.0

    def __init__(self):
        """Initialize with paths from config."""

        self.features_dir = settings.data.features_dir

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features with cyclical encoding.

        Cyclical encoding ensures the model understands that
        hour 23 is close to hour 0, not far apart.
        """

        logger.info("Adding temporal features")

        df["hour"] = df.timestamp.dt.hour
        df["day_of_week"] = df.timestamp.dt.dayofweek
        df["month"] = df.timestamp.dt.month
        df["is_weekend"] = df.day_of_week.isin([5, 6]).astype(int)

        # Cyclical encoding — hour
        df["hour_sin"] = np.sin(2 * np.pi * df.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df.hour / 24)

        # Cyclical encoding — day of week
        df["dow_sin"] = np.sin(2 * np.pi * df.day_of_week / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df.day_of_week / 7)

        # Cyclical encoding — month
        df["month_sin"] = np.sin(2 * np.pi * df.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * df.month / 12)

        logger.info(f"  Added 10 temporal features")

        return df

    def add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Greek holiday features.

        Includes the holiday itself plus day-before and day-after,
        which also show reduced demand patterns.
        """

        logger.info("Adding calendar features")

        # Build holiday set for our date range
        years = range(
            df.timestamp.dt.year.min(),
            df.timestamp.dt.year.max() + 1,
        )
        greek_holidays = holidays.Greece(years=years)

        # Is this day a holiday?
        df["is_holiday"] = df.timestamp.dt.date.apply(
            lambda d: d in greek_holidays
        ).astype(int)

        # Day before and after holidays also have different patterns
        holiday_dates = set(greek_holidays.keys())

        df["is_day_before_holiday"] = df.timestamp.dt.date.apply(
            lambda d: (d + pd.Timedelta(days=1)) in holiday_dates
        ).astype(int)

        df["is_day_after_holiday"] = df.timestamp.dt.date.apply(
            lambda d: (d - pd.Timedelta(days=1)) in holiday_dates
        ).astype(int)

        n_holidays = df.is_holiday.sum() // 24  # approximate days
        logger.info(f"  Added 3 calendar features ({n_holidays} holiday days found)")

        return df

    def add_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived weather features.

        HDD/CDD capture the non-linear U-shaped relationship
        between temperature and electricity demand.
        The threshold is configurable for future optimization.
        """

        logger.info("Adding weather features")

        # Heating Degree Days — how much below comfort zone
        df["hdd"] = np.maximum(self.COMFORT_TEMP - df.temperature_c, 0)

        # Cooling Degree Days — how much above comfort zone
        df["cdd"] = np.maximum(df.temperature_c - self.COMFORT_TEMP, 0)

        # Temperature squared — captures non-linearity
        df["temp_squared"] = df.temperature_c ** 2

        # Apparent temperature (simplified heat index)
        # Combines temperature and humidity effect
        df["apparent_temp"] = df.temperature_c + 0.05 * df.humidity_pct

        logger.info(f"  Added 4 weather features")
        logger.info(f"  HDD range: {df.hdd.min():.1f} to {df.hdd.max():.1f}")
        logger.info(f"  CDD range: {df.cdd.min():.1f} to {df.cdd.max():.1f}")

        return df

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full feature engineering pipeline.

        This builds only 'safe' features — temporal, calendar,
        and weather-derived. Lag and rolling features are built
        inside the forecasting pipeline with explicit cutoff
        to prevent data leakage.
        """

        logger.info(f"Starting feature engineering: {len(df)} rows")

        df = self.add_temporal_features(df)
        df = self.add_calendar_features(df)
        df = self.add_weather_features(df)

        # Save to features directory
        self.features_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.features_dir / "dataset.parquet"
        df.to_parquet(filepath, index=False)

        size_mb = filepath.stat().st_size / (1024 * 1024)
        logger.info(
            f"Feature engineering complete: "
            f"{df.shape[1]} columns, {len(df)} rows, "
            f"saved to {filepath} ({size_mb:.1f} MB)"
        )

        return df                            