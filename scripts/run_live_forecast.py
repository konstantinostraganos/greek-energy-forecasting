"""Live production forecast pipeline.

Downloads fresh data, updates the dataset, and generates
a 24-hour electricity demand forecast using the latest
available information.

Usage:
    python scripts/run_live_forecast.py
    python scripts/run_live_forecast.py --model xgboost
    python scripts/run_live_forecast.py --model lightgbm
    python scripts/run_live_forecast.py --skip-update --model seasonal_naive

Designed to run daily at ~10:00 UTC (13:00 Greek summer,
12:00 Greek winter), matching the backtesting cutoff.
"""

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import holidays
from loguru import logger

from energy_forecasting.config import settings
from energy_forecasting.data.entso_e_client import EntsoEClient
from energy_forecasting.data.weather_client import WeatherClient
from energy_forecasting.data.validator import DataValidator
from energy_forecasting.data.feature_engineer import FeatureEngineer
from energy_forecasting.evaluation.model_registry import get_model_config


def update_dataset() -> pd.DataFrame:
    """Download fresh data and rebuild the feature dataset.

    Steps:
        1. Load existing raw data
        2. Fetch new load data from ENTSO-E (last 7 days
           to catch any late-arriving data)
        3. Fetch weather (observed + forecast for next 48h)
        4. Validate and merge
        5. Build features
        6. Save updated dataset

    Returns:
        Updated feature-engineered DataFrame.
    """

    logger.info("=== Updating dataset with fresh data ===")

    now = datetime.now(tz=timezone.utc)
    today = now.strftime("%Y-%m-%d")
    week_ago = (now - timedelta(days=7)).strftime("%Y-%m-%d")
    tomorrow = (now + timedelta(days=2)).strftime("%Y-%m-%d")

    # --- 1. Fetch fresh load data ---
    logger.info(f"Fetching load data: {week_ago} to {today}")
    try:
        entso_e = EntsoEClient()
        fresh_load = entso_e.fetch_full_dataset(week_ago, today)

        if fresh_load.empty:
            logger.warning("No new load data available")
        else:
            logger.info(f"  Fetched {len(fresh_load)} new load rows")

            # Merge with existing raw data
            raw_load_path = settings.data.raw_dir / "entso_e_load.parquet"
            if raw_load_path.exists():
                existing_load = pd.read_parquet(raw_load_path)
                combined_load = pd.concat(
                    [existing_load, fresh_load]
                ).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
            else:
                combined_load = fresh_load

            combined_load.to_parquet(raw_load_path, index=False)
            logger.info(f"  Load dataset: {len(combined_load)} total rows")

    except Exception as e:
        logger.error(f"Failed to fetch load data: {e}")
        logger.info("Continuing with existing data")

    # --- 2. Fetch fresh weather data ---
    logger.info(f"Fetching weather data: {week_ago} to {tomorrow}")
    try:
        weather = WeatherClient()
        fresh_weather = weather.fetch_weather_data(week_ago, tomorrow)

        if fresh_weather.empty:
            logger.warning("No new weather data available")
        else:
            logger.info(f"  Fetched {len(fresh_weather)} new weather rows")

            # Merge with existing raw data
            raw_weather_path = settings.data.raw_dir / "weather.parquet"
            if raw_weather_path.exists():
                existing_weather = pd.read_parquet(raw_weather_path)
                combined_weather = pd.concat(
                    [existing_weather, fresh_weather]
                ).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
            else:
                combined_weather = fresh_weather

            combined_weather.to_parquet(raw_weather_path, index=False)
            logger.info(f"  Weather dataset: {len(combined_weather)} total rows")

    except Exception as e:
        logger.error(f"Failed to fetch weather data: {e}")
        logger.info("Continuing with existing data")

    # --- 3. Validate and merge ---
    logger.info("Running validation and merge")
    validator = DataValidator()

    raw_load = pd.read_parquet(settings.data.raw_dir / "entso_e_load.parquet")
    raw_weather = pd.read_parquet(settings.data.raw_dir / "weather.parquet")

    clean_load = validator.clean_energy(raw_load)
    clean_weather = validator.clean_weather(raw_weather)
    merged = validator.merge_and_save(clean_load, clean_weather)

    logger.info(f"  Merged dataset: {len(merged)} rows")

    # --- 4. Build features ---
    logger.info("Building features")
    engineer = FeatureEngineer()
    df = engineer.build_features(merged)

    logger.info(f"  Feature dataset: {len(df)} rows, {len(df.columns)} columns")

    return df


def _extend_with_skeleton(df: pd.DataFrame, cutoff_idx: int) -> pd.DataFrame:
    """Extend dataset with skeleton rows for future hours.

    When the dataset doesn't have 24 rows after the cutoff
    (because actual load hasn't happened yet), create
    placeholder rows with calendar features (known in advance)
    and last-known weather values.

    Args:
        df: Current dataset
        cutoff_idx: Index of the cutoff point

    Returns:
        Extended DataFrame with at least 24 rows after cutoff.
    """

    rows_after = len(df) - cutoff_idx - 1
    if rows_after >= 24:
        return df

    missing_hours = 24 - rows_after
    last_ts = df.iloc[-1]["timestamp"]

    logger.info(
        f"  Only {rows_after} rows after cutoff — "
        f"extending with {missing_hours} skeleton rows"
    )

    # Create timestamps for missing future hours
    future_timestamps = pd.date_range(
        start=last_ts + pd.Timedelta(hours=1),
        periods=missing_hours,
        freq="h",
    )

    skeleton = pd.DataFrame({"timestamp": future_timestamps})

    # Calendar features — known in advance
    skeleton["hour"] = skeleton["timestamp"].dt.hour
    skeleton["day_of_week"] = skeleton["timestamp"].dt.dayofweek
    skeleton["month"] = skeleton["timestamp"].dt.month
    skeleton["is_weekend"] = skeleton["day_of_week"].isin([5, 6]).astype(int)
    skeleton["hour_sin"] = np.sin(2 * np.pi * skeleton["hour"] / 24)
    skeleton["hour_cos"] = np.cos(2 * np.pi * skeleton["hour"] / 24)
    skeleton["dow_sin"] = np.sin(2 * np.pi * skeleton["day_of_week"] / 7)
    skeleton["dow_cos"] = np.cos(2 * np.pi * skeleton["day_of_week"] / 7)
    skeleton["month_sin"] = np.sin(2 * np.pi * skeleton["month"] / 12)
    skeleton["month_cos"] = np.cos(2 * np.pi * skeleton["month"] / 12)

    # Holiday features — known in advance
    gr_holidays = holidays.Greece()
    skeleton["is_holiday"] = skeleton["timestamp"].dt.date.apply(
        lambda d: 1 if d in gr_holidays else 0
    )
    skeleton["is_day_before_holiday"] = skeleton["timestamp"].apply(
        lambda t: 1 if (t + pd.Timedelta(days=1)).date() in gr_holidays else 0
    )
    skeleton["is_day_after_holiday"] = skeleton["timestamp"].apply(
        lambda t: 1 if (t - pd.Timedelta(days=1)).date() in gr_holidays else 0
    )

    # Weather — use last known values as placeholder
    # (ideally these come from weather forecast API)
    weather_cols = [
        "temperature_c", "humidity_pct", "solar_radiation_w_m2",
        "wind_speed_kmh", "precipitation_mm",
        "hdd", "cdd", "temp_squared", "apparent_temp",
    ]
    for col in weather_cols:
        if col in df.columns:
            skeleton[col] = df[col].iloc[-1]

    # Load columns stay NaN (unknown future)
    skeleton["actual_load_mw"] = np.nan
    if "forecast_load_mw" in df.columns:
        skeleton["forecast_load_mw"] = np.nan

    # Fill any remaining columns with 0
    for col in df.columns:
        if col not in skeleton.columns:
            skeleton[col] = 0

    # Match column order
    skeleton = skeleton[df.columns]

    extended = pd.concat([df, skeleton], ignore_index=True)
    logger.info(f"  Extended dataset to {len(extended)} rows")

    return extended


def generate_live_forecast(df: pd.DataFrame, model_name: str) -> dict:
    """Generate a live 24h forecast from the latest data.

    Finds the most recent valid cutoff (10:00 UTC),
    trains the model on all data up to that point,
    and predicts the next 24 hours.

    Args:
        df: Updated feature-engineered dataset
        model_name: Model to use (from registry)

    Returns:
        Dictionary with forecast results and metadata.
    """

    logger.info(f"=== Generating live forecast with {model_name} ===")

    config = get_model_config(model_name)
    pipeline, needs_lags = config["factory"]()

        # Find the latest valid cutoff (hour == 10 UTC)
    cutoff_idx = None
    for i in range(len(df) - 1, max(len(df) - 200, 0), -1):
        ts = df.iloc[i]["timestamp"]
        hour = ts.hour if hasattr(ts, "hour") else pd.Timestamp(ts).hour
        if hour == 10:
            cutoff_idx = i
            break

    if cutoff_idx is None:
        raise ValueError("No valid 10:00 UTC cutoff found in recent data")

    cutoff_ts = df.iloc[cutoff_idx]["timestamp"]
    logger.info(f"  Cutoff: {cutoff_ts}")

    # Extend dataset BEFORE lag features — skeleton rows
    # need to exist so shift() computes lags correctly
    df = _extend_with_skeleton(df, cutoff_idx)

         # Add lag features AFTER extension
    if needs_lags:
        df = pipeline.add_lag_features(df)

        # Forward-fill any NaN in lag/rolling features —
        # skeleton rows have NaN actual_load_mw which
        # causes rolling features to be NaN at the edge
        lag_cols = [c for c in df.columns if "lag" in c or "rolling" in c]
        df[lag_cols] = df[lag_cols].ffill()
    # Train on all data up to cutoff
    if config["training_strategy"] != "none":
        logger.info(f"  Training {model_name} on {cutoff_idx + 1} rows")
        pipeline.train(df, train_end_idx=cutoff_idx)

    # Generate forecast
    result = pipeline.predict(df, cutoff_idx=cutoff_idx)

    logger.info(f"  Generated {len(result)} hourly predictions")

    # Save forecast
    forecasts_dir = Path("data/forecasts")
    forecasts_dir.mkdir(parents=True, exist_ok=True)

    forecast_date = (
        cutoff_ts.strftime("%Y-%m-%d")
        if hasattr(cutoff_ts, "strftime")
        else str(cutoff_ts)[:10]
    )
    forecast_path = forecasts_dir / f"{model_name}_{forecast_date}.parquet"
    result.to_parquet(forecast_path, index=False)

    # Also save as JSON for easy reading
    now_utc = datetime.now(tz=timezone.utc)
    forecast_json = {
        "model": model_name,
        "cutoff_timestamp": str(cutoff_ts),
        "weather_mode": config["weather_mode"],
        "generated_at": now_utc.isoformat(),
        "predictions": [
            {
                "timestamp": str(row["timestamp"]),
                "predicted_load_mw": round(float(row["predicted_load_mw"]), 1),
                "horizon_step": i + 1,
            }
            for i, (_, row) in enumerate(result.iterrows())
        ],
    }

    json_path = forecasts_dir / f"{model_name}_{forecast_date}.json"
    with open(json_path, "w") as f:
        json.dump(forecast_json, f, indent=2)

    logger.info(f"  Saved to {forecast_path}")

    # Print forecast
    print(f"\n{'=' * 55}")
    print(f"  LIVE FORECAST — {model_name.upper()}")
    print(f"{'=' * 55}")
    print(f"  Cutoff:     {cutoff_ts}")
    print(f"  Weather:    {config['weather_mode']}")
    print(f"  Generated:  {now_utc.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"")
    print(f"  {'Hour':<6} {'Prediction':>12}")
    print(f"  {'-' * 20}")
    for i, (_, row) in enumerate(result.iterrows()):
        ts = str(row["timestamp"])[-14:-6]
        load = row["predicted_load_mw"]
        print(f"  h+{i+1:<3}  {load:>10.0f} MW")
    print(f"{'=' * 55}\n")

    return forecast_json


def main():
    parser = argparse.ArgumentParser(
        description="Run live electricity demand forecast"
    )
    parser.add_argument(
        "--model",
        default="xgboost",
        help="Model to use (default: xgboost)",
    )
    parser.add_argument(
        "--skip-update",
        action="store_true",
        help="Skip data download, use existing dataset",
    )

    args = parser.parse_args()

    # Update dataset with fresh data
    if args.skip_update:
        logger.info("Skipping data update (--skip-update)")
        df = pd.read_parquet(
            settings.data.features_dir / "dataset.parquet"
        )
    else:
        df = update_dataset()

    # Generate forecast
    generate_live_forecast(df, args.model)


if __name__ == "__main__":
    main()