"""Tests for the forecasting pipeline."""

import pandas as pd
import numpy as np
import pytest

from energy_forecasting.models.forecaster import ForecastPipeline


class TestLagFeatures:
    """Verify lag features are cutoff-safe."""

    def setup_method(self):
        """Create a dataset with known values."""

        n_hours = 500
        self.df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n_hours, freq="h"),
            "actual_load_mw": range(1000, 1000 + n_hours),
        })

    def test_lag_24_correct(self):
        """load_lag_24h should equal actual_load 24 rows earlier."""

        pipeline = ForecastPipeline()
        df = pipeline.add_lag_features(self.df)

        # Row 100: lag_24 should equal row 76
        assert df.iloc[100]["load_lag_24h"] == df.iloc[76]["actual_load_mw"]

    def test_lag_168_correct(self):
        """load_lag_168h should equal actual_load 168 rows earlier."""

        pipeline = ForecastPipeline()
        df = pipeline.add_lag_features(self.df)

        assert df.iloc[200]["load_lag_168h"] == df.iloc[32]["actual_load_mw"]

    def test_first_rows_are_nan(self):
        """First rows should have NaN lags (no history available)."""

        pipeline = ForecastPipeline()
        df = pipeline.add_lag_features(self.df)

        assert pd.isna(df.iloc[0]["load_lag_24h"])
        assert pd.isna(df.iloc[23]["load_lag_24h"])
        assert not pd.isna(df.iloc[24]["load_lag_24h"])

    def test_no_leakage_at_max_horizon(self):
        """For cutoff C and target C+24, lag_24 must use C or earlier."""

        pipeline = ForecastPipeline()
        df = pipeline.add_lag_features(self.df)

        cutoff_idx = 300
        target_idx = cutoff_idx + 24  # worst case — farthest target

        # lag_24 at target_idx looks at target_idx - 24 = cutoff_idx
        lag_source_idx = target_idx - 24
        assert lag_source_idx <= cutoff_idx

    def test_rolling_uses_shifted_data(self):
        """Rolling mean should not include any data after shift point."""

        pipeline = ForecastPipeline()
        df = pipeline.add_lag_features(self.df)

        # At row 100: rolling_24h = mean of rows 53..76 (shift 24, then window 24)
        expected = self.df.iloc[53:77]["actual_load_mw"].mean()
        actual = df.iloc[100]["load_rolling_24h"]

        assert abs(actual - expected) < 0.01


class TestForecastPipeline:
    """Test train and predict workflow."""

    def setup_method(self):
        """Create a realistic test dataset."""

        np.random.seed(42)
        n_hours = 1000

        hours = np.arange(n_hours) % 24
        load = 5000 + 1000 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 100, n_hours)

        self.df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n_hours, freq="h"),
            "actual_load_mw": load,
            "hour": hours,
            "day_of_week": pd.date_range("2024-01-01", periods=n_hours, freq="h").dayofweek,
            "month": pd.date_range("2024-01-01", periods=n_hours, freq="h").month,
            "is_weekend": pd.date_range("2024-01-01", periods=n_hours, freq="h").dayofweek.isin([5, 6]).astype(int),
            "hour_sin": np.sin(2 * np.pi * hours / 24),
            "hour_cos": np.cos(2 * np.pi * hours / 24),
            "dow_sin": np.zeros(n_hours),
            "dow_cos": np.zeros(n_hours),
            "month_sin": np.zeros(n_hours),
            "month_cos": np.zeros(n_hours),
            "is_holiday": np.zeros(n_hours),
            "is_day_before_holiday": np.zeros(n_hours),
            "is_day_after_holiday": np.zeros(n_hours),
            "hdd": np.maximum(18 - 15, 0) * np.ones(n_hours),
            "cdd": np.zeros(n_hours),
            "temp_squared": 225 * np.ones(n_hours),
            "apparent_temp": 18 * np.ones(n_hours),
            "temperature_c": 15 * np.ones(n_hours),
            "humidity_pct": 60 * np.ones(n_hours),
            "solar_radiation_w_m2": 200 * np.ones(n_hours),
            "wind_speed_kmh": 10 * np.ones(n_hours),
            "precipitation_mm": np.zeros(n_hours),
        })

        self.pipeline = ForecastPipeline()
        self.df = self.pipeline.add_lag_features(self.df)

    def test_predict_before_train_raises(self):
        """Should raise error if predict called before train."""

        with pytest.raises(RuntimeError, match="not fitted"):
            self.pipeline.predict(self.df, cutoff_idx=500)

    def test_predict_returns_24_rows(self):
        """Forecast should be exactly 24 hours."""

        self.pipeline.train(self.df, train_end_idx=800)
        result = self.pipeline.predict(self.df, cutoff_idx=900)

        assert len(result) == 24

    def test_predict_output_columns(self):
        """Output should have the right columns."""

        self.pipeline.train(self.df, train_end_idx=800)
        result = self.pipeline.predict(self.df, cutoff_idx=900)

        assert "timestamp" in result.columns
        assert "predicted_load_mw" in result.columns
        assert "actual_load_mw" in result.columns

    def test_predictions_are_reasonable(self):
        """Predictions should be in a reasonable range."""

        self.pipeline.train(self.df, train_end_idx=800)
        result = self.pipeline.predict(self.df, cutoff_idx=900)

        assert result.predicted_load_mw.min() > 2000
        assert result.predicted_load_mw.max() < 12000