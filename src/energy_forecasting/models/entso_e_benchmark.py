"""ENTSO-E day-ahead forecast benchmark.

Uses the TSO's own day-ahead load forecast (forecast_load_mw)
as a benchmark. This is not a model we train — it wraps the
published forecasts from ADMIE/ENTSO-E so they can be
evaluated through the same pipeline as our models.

Important: the TSO's forecast origin and methodology differ
from ours (different cutoff time, different weather inputs,
proprietary models). The comparison is indicative, not
a controlled experiment.
"""

import pandas as pd
from loguru import logger


class EntsoEBenchmark:
    """Wrapper for ENTSO-E day-ahead forecasts.

    Provides the same predict() interface as other models
    so it can be evaluated with the same cutoffs, metrics,
    and MLflow logging.

    No training is needed — the forecasts already exist
    in the dataset as 'forecast_load_mw'.
    """

    HORIZON = 24

    def predict(self, df: pd.DataFrame, cutoff_idx: int) -> pd.DataFrame:
        """Return ENTSO-E's day-ahead forecast for next 24 hours.

        Args:
            df: Dataset containing 'forecast_load_mw' column
            cutoff_idx: Index of the forecast origin

        Returns:
            DataFrame with timestamp, predicted_load_mw,
            and actual_load_mw — same format as all other models
        """

        forecast_start = cutoff_idx + 1
        forecast_end = min(forecast_start + self.HORIZON, len(df))
        future = df.iloc[forecast_start:forecast_end].copy()

        if "forecast_load_mw" not in future.columns:
            raise ValueError(
                "Column 'forecast_load_mw' not found. "
                "Use the processed dataset that includes "
                "ENTSO-E day-ahead forecasts."
            )

        future["predicted_load_mw"] = future["forecast_load_mw"]

        return future[["timestamp", "predicted_load_mw", "actual_load_mw"]]