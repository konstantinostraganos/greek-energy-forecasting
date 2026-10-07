"""Tests for data collection clients."""

import pandas as pd
import pytest

from energy_forecasting.data.weather_client import WeatherClient


class TestWeatherClient:
    """Tests for Open-Meteo weather client."""

    def test_client_initializes(self):
        """Client should load settings without errors."""
        client = WeatherClient()
        assert client.archive_url is not None
        assert len(client.locations) > 0

    def test_fetch_single_location(self):
        """Should fetch weather data for one location."""
        client = WeatherClient()
        df = client._fetch_location(
            latitude=37.98,
            longitude=23.73,
            start_date="2024-01-01",
            end_date="2024-01-03",
        )
        assert not df.empty
        assert "timestamp" in df.columns
        assert "temperature_c" in df.columns
        # 3 days × 24 hours = 72 rows
        assert len(df) == 72

    def test_fetch_weather_data_weighted(self):
        """Should return weighted average across all cities."""
        client = WeatherClient()
        df = client.fetch_weather_data("2024-01-01", "2024-01-02")
        assert not df.empty
        # 2 days × 24 hours = 48 rows
        assert len(df) == 48

    def test_weather_columns(self):
        """Should have all expected weather columns."""
        client = WeatherClient()
        df = client.fetch_weather_data("2024-01-01", "2024-01-02")
        expected_columns = [
            "timestamp",
            "temperature_c",
            "humidity_pct",
            "solar_radiation_w_m2",
            "wind_speed_kmh",
            "precipitation_mm",
        ]
        for col in expected_columns:
            assert col in df.columns


class TestEntsoEClient:
    """Tests for ENTSO-E client — no API key needed."""

    def test_client_initializes(self):
        """Client should load settings without errors."""
        from energy_forecasting.data.entso_e_client import EntsoEClient
        client = EntsoEClient()
        assert client.base_url is not None
        assert client.area_code == "10YGR-HTSO-----Y"

    def test_fetch_without_key_raises(self):
        """Should raise clear error when API key is missing."""
        from energy_forecasting.data.entso_e_client import EntsoEClient
        client = EntsoEClient()
        if client.api_key is None:
            with pytest.raises(ValueError, match="API key not found"):
                from datetime import datetime
                client._fetch_data(
                    document_type="A65",
                    process_type="A16",
                    start=datetime(2024, 1, 1),
                    end=datetime(2024, 1, 2),
                )