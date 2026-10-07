"""Tests for PatchTST transformer model (univariate)."""

import numpy as np
import pandas as pd
import pytest

from energy_forecasting.models.transformer_model import TransformerPipeline


class TestTransformerPipeline:
    """Test PatchTST pipeline train and predict."""

    def setup_method(self):
        """Create realistic synthetic dataset.

        Uses 3000 hours — enough for NeuralForecast windowing
        with input_size=168 and horizon=24.
        """

        np.random.seed(42)
        n_hours = 3000

        hours = np.arange(n_hours) % 24
        load = 5000 + 1000 * np.sin(2 * np.pi * hours / 24)
        load += np.random.normal(0, 100, n_hours)

        self.df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n_hours, freq="h"),
            "actual_load_mw": load,
        })

    def test_predict_before_train_raises(self):
        """Should raise error if predict called before train."""

        pipeline = TransformerPipeline(max_steps=5)
        with pytest.raises(RuntimeError, match="not fitted"):
            pipeline.predict(self.df, cutoff_idx=500)

    def test_predict_returns_24_rows(self):
        """Forecast should be exactly 24 hours."""

        pipeline = TransformerPipeline(max_steps=5)
        pipeline.train(self.df, train_end_idx=2500)
        result = pipeline.predict(self.df, cutoff_idx=2700)

        assert len(result) == 24

    def test_predict_output_columns(self):
        """Output should match other model interfaces."""

        pipeline = TransformerPipeline(max_steps=5)
        pipeline.train(self.df, train_end_idx=2500)
        result = pipeline.predict(self.df, cutoff_idx=2700)

        assert list(result.columns) == [
            "timestamp", "predicted_load_mw", "actual_load_mw"
        ]

    def test_nixtla_format_conversion(self):
        """Nixtla format should have required columns (univariate)."""

        pipeline = TransformerPipeline(max_steps=5)
        nf_df = pipeline._to_nixtla_format(self.df)

        assert "unique_id" in nf_df.columns
        assert "ds" in nf_df.columns
        assert "y" in nf_df.columns
        assert len(nf_df.columns) == 3

    def test_predict_insufficient_future_raises(self):
        """Should raise ValueError if not enough rows after cutoff."""

        pipeline = TransformerPipeline(max_steps=5)
        pipeline.train(self.df, train_end_idx=2500)

        with pytest.raises(ValueError, match="Need 24 rows"):
            pipeline.predict(self.df, cutoff_idx=len(self.df) - 10)