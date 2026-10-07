"""Prediction intervals using conformal prediction.

Constructs prediction intervals from historical backtest
residuals. The interval width at each horizon step reflects
the model's actual empirical accuracy, not theoretical
assumptions.

Method:
    1. Collect residuals from backtesting (actual - predicted)
    2. For each horizon step (1-24), compute the empirical
       quantiles of absolute residuals
    3. Add/subtract the quantile from the point forecast

This is distribution-free — no normality assumption needed.
The intervals are calibrated: a 90% interval should contain
the actual value ~90% of the time.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger


class ConformalIntervals:
    """Compute prediction intervals from backtest residuals.

    Uses the empirical distribution of errors at each
    horizon step to construct intervals that reflect
    how accurate the model actually is.
    """

    def __init__(self, confidence: float = 0.90):
        """Initialize with desired confidence level.

        Args:
            confidence: Coverage probability (0.90 = 90% interval)
        """

        if not 0 < confidence < 1:
            raise ValueError("Confidence must be between 0 and 1")

        self.confidence = confidence
        self.quantiles = None
        self.is_fitted = False

    def fit(self, results_df: pd.DataFrame) -> None:
        """Learn interval widths from backtesting results.

        Args:
            results_df: Must contain 'actual_load_mw',
                        'predicted_load_mw', and 'horizon_step'
        """

        required = {"actual_load_mw", "predicted_load_mw", "horizon_step"}
        missing = required - set(results_df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Compute residuals
        df = results_df.copy()
        df["residual"] = df["actual_load_mw"] - df["predicted_load_mw"]
        df["abs_residual"] = df["residual"].abs()

        # Compute quantile of absolute residuals per horizon step
        alpha = (1 + self.confidence) / 2
        self.quantiles = (
            df.groupby("horizon_step")["abs_residual"]
            .quantile(alpha)
            .to_dict()
        )

        self.is_fitted = True

        # Log summary
        mean_width = np.mean(list(self.quantiles.values()))
        logger.info(
            f"Conformal intervals fitted: {self.confidence:.0%} confidence, "
            f"mean half-width: {mean_width:.0f} MW"
        )

    def predict(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        """Add prediction intervals to a forecast.

        Args:
            forecast_df: Must contain 'predicted_load_mw'.
                         If 'horizon_step' is missing, assumes
                         rows are ordered 1..N.

        Returns:
            DataFrame with added 'lower_bound' and 'upper_bound'.
        """

        if not self.is_fitted:
            raise RuntimeError("Not fitted. Call fit() first.")

        result = forecast_df.copy()

        # Add horizon_step if missing
        if "horizon_step" not in result.columns:
            result["horizon_step"] = range(1, len(result) + 1)

        # Add interval bounds
        half_widths = result["horizon_step"].map(self.quantiles)

        # Fallback for horizon steps not seen during fitting
        max_quantile = max(self.quantiles.values())
        half_widths = half_widths.fillna(max_quantile)

        result["lower_bound"] = result["predicted_load_mw"] - half_widths
        result["upper_bound"] = result["predicted_load_mw"] + half_widths
        result["interval_width"] = half_widths * 2

        return result

    @classmethod
    def from_parquet(cls, path: str, confidence: float = 0.90):
        """Create and fit from a saved predictions parquet file.

        Convenience method for loading MLflow artifacts.
        """

        df = pd.read_parquet(path)
        intervals = cls(confidence=confidence)
        intervals.fit(df)
        return intervals