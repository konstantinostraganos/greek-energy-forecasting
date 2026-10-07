"""Tests for LSTM forecasting model."""

import numpy as np
import pandas as pd
import torch
import pytest

from energy_forecasting.models.lstm_model import (
    EnergyDataset,
    LSTMForecaster,
    LSTMPipeline,
    SEQUENCE_FEATURES,
)


class TestEnergyDataset:
    """Verify dataset creates correct 168→24 samples."""

    def setup_method(self):
        """Create test data."""

        n = 500
        self.X = np.random.randn(n, 20).astype(np.float32)
        self.y = np.random.randn(n).astype(np.float32)

    def test_sample_shapes(self):
        """X should be (168, features), y should be (24,)."""

        dataset = EnergyDataset(self.X, self.y, seq_len=168, forecast_horizon=24)
        x_sample, y_sample = dataset[0]

        assert x_sample.shape == (168, 20)
        assert y_sample.shape == (24,)

    def test_length_accounts_for_horizon(self):
        """Dataset length should account for both seq_len and horizon."""

        dataset = EnergyDataset(self.X, self.y, seq_len=168, forecast_horizon=24)

        # 500 - 168 - 24 + 1 = 309
        assert len(dataset) == 500 - 168 - 24 + 1

    def test_no_overlap_input_target(self):
        """Input sequence must end before target starts."""

        dataset = EnergyDataset(self.X, self.y, seq_len=168, forecast_horizon=24)

        # For sample at idx=0:
        # Input: rows 0..167
        # Target: rows 168..191
        x_sample, y_sample = dataset[0]

        # Last input value should NOT equal first target value
        # (they come from different arrays, but verify the indexing)
        assert y_sample[0] == self.y[168]
        assert y_sample[-1] == self.y[191]


class TestLSTMForecaster:
    """Verify model architecture and output shapes."""

    def test_output_shape(self):
        """Model output should be (batch_size, 24)."""

        model = LSTMForecaster(n_features=20, forecast_horizon=24)
        x = torch.randn(8, 168, 20)  # batch=8

        output = model(x)

        assert output.shape == (8, 24)

    def test_single_sample(self):
        """Should work with batch_size=1."""

        model = LSTMForecaster(n_features=20, forecast_horizon=24)
        x = torch.randn(1, 168, 20)

        output = model(x)

        assert output.shape == (1, 24)


class TestLSTMPipeline:
    """Test full pipeline train and predict."""

    def setup_method(self):
        """Create realistic test dataset."""

        np.random.seed(42)
        n_hours = 1000

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

        pipeline = LSTMPipeline(epochs=1)
        with pytest.raises(RuntimeError, match="not fitted"):
            pipeline.predict(self.df, cutoff_idx=500)

    def test_predict_returns_24_rows(self):
        """Forecast should be exactly 24 hours."""

        pipeline = LSTMPipeline(epochs=2, batch_size=32)
        pipeline.train(self.df, train_end_idx=800)
        result = pipeline.predict(self.df, cutoff_idx=900)

        assert len(result) == 24

    def test_predict_output_columns(self):
        """Output should match baseline interface."""

        pipeline = LSTMPipeline(epochs=2, batch_size=32)
        pipeline.train(self.df, train_end_idx=800)
        result = pipeline.predict(self.df, cutoff_idx=900)

        assert "timestamp" in result.columns
        assert "predicted_load_mw" in result.columns
        assert "actual_load_mw" in result.columns

    def test_insufficient_history_raises(self):
        """Should raise if not enough history for sequence."""

        pipeline = LSTMPipeline(epochs=2, batch_size=32)
        pipeline.train(self.df, train_end_idx=800)

        with pytest.raises(ValueError, match="Not enough history"):
            pipeline.predict(self.df, cutoff_idx=100)