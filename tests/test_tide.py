"""Tests for TiDE model with exogenous features."""

import numpy as np
import pandas as pd
import pytest

from energy_forecasting.models.tide_model import TiDEPipeline, EXOG_FEATURES


class TestTiDEPipeline:
    """Test TiDE pipeline train and predict."""

    def setup_method(self):
        """Create realistic synthetic dataset with all exogenous features."""

        np.random.seed(42)
        n_hours = 3000

        hours = np.arange(n_hours) % 24
        load = 5000 + 1000 * np.sin(2 * np.pi * hours / 24)
        load += np.random.normal(0, 100, n_hours)

        self.df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n_hours, freq="h"),
            "actual_load_mw": load,
            "temperature_c": 15 + 10 * np.sin(2 * np.pi * hours / 24),
            "humidity_pct": 60 + np.random.normal(0, 5, n_hours),
            "solar_radiation_w_m2": np.maximum(0, 500 * np.sin(2 * np.pi * hours / 24)),
            "wind_speed_kmh": 10 + np.random.normal(0, 2, n_hours),
            "precipitation_mm": np.zeros(n_hours),
            "hdd": np.maximum(18 - 15, 0) * np.ones(n_hours),
            "cdd": np.zeros(n_hours),
            "temp_squared": 225 * np.ones(n_hours),
            "apparent_temp": 18 * np.ones(n_hours),
            "hour_sin": np.sin(2 * np.pi * hours / 24),
            "hour_cos": np.cos(2 * np.pi * hours / 24),
            "dow_sin": np.zeros(n_hours),
            "dow_cos": np.ones(n_hours),
            "month_sin": np.zeros(n_hours),
            "month_cos": np.ones(n_hours),
            "is_weekend": np.zeros(n_hours),
            "is_holiday": np.zeros(n_hours),
            "is_day_before_holiday": np.zeros(n_hours),
            "is_day_after_holiday": np.zeros(n_hours),
        })

    def test_predict_before_train_raises(self):
        """Should raise error if predict called before train."""

        pipeline = TiDEPipeline(max_steps=5)
        with pytest.raises(RuntimeError, match="not fitted"):
            pipeline.predict(self.df, cutoff_idx=500)

    def test_predict_returns_24_rows(self):
        """Forecast should be exactly 24 hours."""

        pipeline = TiDEPipeline(max_steps=5)
        pipeline.train(self.df, train_end_idx=2500)
        result = pipeline.predict(self.df, cutoff_idx=2700)

        assert len(result) == 24

    def test_predict_output_columns(self):
        """Output should match other model interfaces."""

        pipeline = TiDEPipeline(max_steps=5)
        pipeline.train(self.df, train_end_idx=2500)
        result = pipeline.predict(self.df, cutoff_idx=2700)

        assert list(result.columns) == [
            "timestamp", "predicted_load_mw", "actual_load_mw"
        ]

    def test_nixtla_format_has_exogenous(self):
        """Nixtla format should include all exogenous features."""

        pipeline = TiDEPipeline(max_steps=5)
        nf_df = pipeline._to_nixtla_format(self.df)

        assert "unique_id" in nf_df.columns
        assert "ds" in nf_df.columns
        assert "y" in nf_df.columns
        for col in EXOG_FEATURES:
            assert col in nf_df.columns

    def test_validate_features_catches_missing(self):
        """Should raise ValueError if required features are missing."""

        pipeline = TiDEPipeline(max_steps=5)
        broken_df = self.df.drop(columns=["hdd", "cdd"])

        with pytest.raises(ValueError, match="Missing required columns"):
            pipeline.train(broken_df, train_end_idx=2500)

    def test_predict_insufficient_future_raises(self):
        """Should raise ValueError if not enough rows after cutoff."""

        pipeline = TiDEPipeline(max_steps=5)
        pipeline.train(self.df, train_end_idx=2500)

        with pytest.raises(ValueError, match="Need 24 rows"):
            pipeline.predict(self.df, cutoff_idx=len(self.df) - 10)