"""Download the full dataset: actual load + forecast + weather."""

from energy_forecasting.data.entso_e_client import EntsoEClient
from energy_forecasting.data.weather_client import WeatherClient
from energy_forecasting.logger import logger

# --- ENTSO-E Data ---
logger.info("=== Starting ENTSO-E download ===")
entso_client = EntsoEClient()
df_energy = entso_client.fetch_full_dataset()
print(f"\nEnergy data: {len(df_energy)} rows")
print(f"From: {df_energy.timestamp.min()}")
print(f"To:   {df_energy.timestamp.max()}")
print(df_energy.head())
entso_client.save_raw_data(df_energy)

# --- Weather Data ---
logger.info("=== Starting Weather download ===")
weather_client = WeatherClient()
df_weather = weather_client.fetch_weather_data()
print(f"\nWeather data: {len(df_weather)} rows")
print(f"From: {df_weather.timestamp.min()}")
print(f"To:   {df_weather.timestamp.max()}")
print(df_weather.head())
weather_client.save_raw_data(df_weather)

logger.info("=== Download complete ===")