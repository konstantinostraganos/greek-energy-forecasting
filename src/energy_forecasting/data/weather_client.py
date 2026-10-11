"""Open-Meteo Weather API client.

Fetches historical and forecast weather data for major Greek cities,
weighted by population/consumption contribution.
"""

from datetime import datetime

import pandas as pd
import requests
from loguru import logger

from energy_forecasting.config import settings


class WeatherClient:
    """Client for the Open-Meteo Weather API.

    Fetches hourly weather data for multiple Greek cities
    and computes population-weighted averages to represent
    national-level weather conditions.
    """

    # Weather variables that affect electricity demand
    HOURLY_VARIABLES = [
        "temperature_2m",
        "relative_humidity_2m",
        "shortwave_radiation",
        "wind_speed_10m",
        "precipitation",
    ]

    def __init__(self):
        """Initialize with locations from config."""

        self.forecast_url = settings.weather.forecast_url
        self.archive_url = settings.weather.archive_url
        self.locations = settings.weather.locations

    def _fetch_location(
        self,
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch hourly weather data for a single location.

        Automatically splits request across archive and forecast
        APIs when the date range spans both.
        """

        cutoff = datetime.now().strftime("%Y-%m-%d")

        parts = []

        # Historical part (archive API)
        if start_date < cutoff:
            archive_end = min(end_date, cutoff)
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date,
                "end_date": archive_end,
                "hourly": ",".join(self.HOURLY_VARIABLES),
                "timezone": "UTC",
            }
            response = requests.get(self.archive_url, params=params, timeout=60)
            response.raise_for_status()
            parts.append(response.json()["hourly"])

        # Recent/future part (forecast API)
        if end_date >= cutoff:
            forecast_start = max(start_date, cutoff)
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": forecast_start,
                "end_date": end_date,
                "hourly": ",".join(self.HOURLY_VARIABLES),
                "timezone": "UTC",
            }
            response = requests.get(self.forecast_url, params=params, timeout=60)
            response.raise_for_status()
            parts.append(response.json()["hourly"])

        # Combine all parts
        combined = {
            "time": [],
            "temperature_2m": [],
            "relative_humidity_2m": [],
            "shortwave_radiation": [],
            "wind_speed_10m": [],
            "precipitation": [],
        }
        for part in parts:
            for key in combined:
                combined[key].extend(part[key])

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(combined["time"], utc=True),
            "temperature_c": combined["temperature_2m"],
            "humidity_pct": combined["relative_humidity_2m"],
            "solar_radiation_w_m2": combined["shortwave_radiation"],
            "wind_speed_kmh": combined["wind_speed_10m"],
            "precipitation_mm": combined["precipitation"],
        })

        return df

    def fetch_weather_data(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Fetch weather data for all locations and compute weighted average.

        Each city's weather is weighted by its contribution to
        national electricity consumption (configured in YAML).

        Args:
            start_date: "YYYY-MM-DD" or None (uses config)
            end_date: "YYYY-MM-DD" or None (uses config)

        Returns:
            DataFrame with weighted average weather for Greece
        """

        start = start_date or settings.entso_e.start_date
        end = end_date or settings.entso_e.end_date

        logger.info(f"Fetching weather data: {start} to {end}")

        weather_columns = [
            "temperature_c",
            "humidity_pct",
            "solar_radiation_w_m2",
            "wind_speed_kmh",
            "precipitation_mm",
        ]

        weighted_sum = None

        for location in self.locations:
            logger.info(
                f"  Fetching {location.name} (weight={location.weight})"
            )

            try:
                df = self._fetch_location(
                    latitude=location.latitude,
                    longitude=location.longitude,
                    start_date=start,
                    end_date=end,
                )
            except Exception as e:
                logger.error(f"  Failed to fetch {location.name}: {e}")
                continue

            # Multiply weather values by this city's weight
            if weighted_sum is None:
                weighted_sum = df.copy()
                for col in weather_columns:
                    weighted_sum[col] = df[col] * location.weight
            else:
                for col in weather_columns:
                    weighted_sum[col] += df[col] * location.weight

        if weighted_sum is None:
            logger.error("No weather data fetched for any location")
            return pd.DataFrame()

        logger.info(f"Weather data complete: {len(weighted_sum)} rows")

        return weighted_sum

    def save_raw_data(self, df: pd.DataFrame, filename: str = "weather.parquet") -> None:
        """Save raw weather data to Parquet format."""

        save_dir = settings.data.raw_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        filepath = save_dir / filename
        df.to_parquet(filepath, index=False)

        size_mb = filepath.stat().st_size / (1024 * 1024)
        logger.info(f"Saved {len(df)} rows to {filepath} ({size_mb:.1f} MB)")