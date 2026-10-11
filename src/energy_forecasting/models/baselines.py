"""Baseline forecasting models.

Simple models that establish performance floor.
More complex models should be evaluated against this baseline to determine whether they provide meaningful predictive improvement.
"""

import pandas as pd
import numpy as np
from loguru import logger


class NaiveForecaster:
    """Predicts that demand = last known value for all 24 hours.

    The simplest possible forecast — assumes nothing changes.
    If our models can't beat this, they're useless.

    Example:
        cutoff = Monday 12:00, last known load = 5800 MW
        Monday 13:00 → 5800
        Monday 14:00 → 5800
        ...
        Tuesday 12:00 → 5800
    """

    def predict(self, df: pd.DataFrame, cutoff_idx: int) -> pd.DataFrame:
        """Generate 24h forecast from a cutoff point.

        Args:
            df: Full dataset with actual_load_mw
            cutoff_idx: Index of the last known observation

        Returns:
            DataFrame with timestamp, predicted_load_mw,
            and actual_load_mw for the next 24 hours
        """

        # Last known value at cutoff
        last_known = df.iloc[cutoff_idx]["actual_load_mw"]

        # The 24 hours we want to predict
        forecast_start = cutoff_idx + 1
        forecast_end = min(cutoff_idx + 25, len(df))
        future = df.iloc[forecast_start:forecast_end].copy()

        # All predictions = last known value
        future["predicted_load_mw"] = last_known

        return future[["timestamp", "predicted_load_mw", "actual_load_mw"]]


class SeasonalNaiveForecaster:
    """Predicts that demand = same hour yesterday.

    Uses the daily seasonal pattern — much smarter than
    flat Naive because it captures the daily cycle
    (low at night, high during day).

    Example:
        cutoff = Monday 12:00
        Monday 13:00 → Sunday 13:00 actual
        Monday 14:00 → Sunday 14:00 actual
        ...
        Tuesday 12:00 → Monday 12:00 actual
    """

    def predict(self, df: pd.DataFrame, cutoff_idx: int) -> pd.DataFrame:
        """Generate 24h forecast using yesterday's values.

        Args:
            df: Full dataset with actual_load_mw
            cutoff_idx: Index of the last known observation

        Returns:
            DataFrame with timestamp, predicted_load_mw,
            and actual_load_mw for the next 24 hours
        """

        forecast_start = cutoff_idx + 1
        forecast_end = min(cutoff_idx + 25, len(df))
        future = df.iloc[forecast_start:forecast_end].copy()

        # Each hour's prediction = same hour, 24 hours ago
        predictions = []
        for idx in range(forecast_start, forecast_end):
            lag_idx = idx - 24
            if lag_idx >= 0:
                predictions.append(df.iloc[lag_idx]["actual_load_mw"])
            else:
                predictions.append(np.nan)

        future["predicted_load_mw"] = predictions

        return future[["timestamp", "predicted_load_mw", "actual_load_mw"]]    