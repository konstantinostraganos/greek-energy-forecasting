"""Tests for baseline forecasting models."""

import pandas as pd
import numpy as np

from energy_forecasting.models.baselines import NaiveForecaster


class TestNaiveForecaster:
    """Tests for Naive baseline."""

    def setup_method(self):
        """Create a small test dataset."""

        self.df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=48, freq="h"),
            "actual_load_mw": np.random.uniform(4000, 7000, 48),
        })

    def test_all_predictions_equal_last_known(self):
        """All 24 predictions should equal the cutoff value."""

        forecaster = NaiveForecaster()
        cutoff_idx = 23  # hour 23 of day 1
        result = forecaster.predict(self.df, cutoff_idx)

        last_known = self.df.iloc[cutoff_idx]["actual_load_mw"]

        assert len(result) == 24
        assert all(result.predicted_load_mw == last_known)

    def test_output_columns(self):
        """Output should have the right columns."""

        forecaster = NaiveForecaster()
        result = forecaster.predict(self.df, cutoff_idx=23)

        assert "timestamp" in result.columns
        assert "predicted_load_mw" in result.columns
        assert "actual_load_mw" in result.columns

from energy_forecasting.models.baselines import NaiveForecaster, SeasonalNaiveForecaster


class TestSeasonalNaiveForecaster:
    """Tests for Seasonal Naive baseline."""

    def setup_method(self):
        """Create test dataset with a clear daily pattern."""

        # 72 hours — enough for lag_24h to work
        self.df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=72, freq="h"),
            "actual_load_mw": list(range(4000, 4024)) * 3,  # repeating pattern
        })

    def test_predictions_match_yesterday(self):
        """Each prediction should equal same hour 24h ago."""

        forecaster = SeasonalNaiveForecaster()
        cutoff_idx = 47  # end of day 2
        result = forecaster.predict(self.df, cutoff_idx)

        # Hour 48 should match hour 24, hour 49 should match hour 25, etc.
        for i, row in result.iterrows():
            lag_value = self.df.iloc[i - 24]["actual_load_mw"]
            assert row.predicted_load_mw == lag_value

    def test_output_length(self):
        """Should produce 24 predictions."""

        forecaster = SeasonalNaiveForecaster()
        result = forecaster.predict(self.df, cutoff_idx=47)
        assert len(result) == 24        