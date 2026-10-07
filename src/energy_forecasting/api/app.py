"""FastAPI application for electricity demand forecasting.

Serves trained models as a REST API, enabling external
applications (Streamlit, scheduled jobs, other services)
to request 24h load forecasts.

Start the server:
    uvicorn energy_forecasting.api.app:app --reload

API docs (auto-generated):
    http://localhost:8000/docs
"""

from pathlib import Path

import numpy as np
import pandas as pd
import holidays
from fastapi import FastAPI, HTTPException
from loguru import logger

from energy_forecasting.config import settings
from energy_forecasting.api.schemas import (
    ForecastRequest,
    ForecastResponse,
    HourlyForecast,
    ModelInfo,
    ModelsResponse,
    HealthResponse,
)
from energy_forecasting.evaluation.model_registry import (
    get_model_config,
    get_available_models,
)
from energy_forecasting.evaluation.prediction_intervals import ConformalIntervals


app = FastAPI(
    title="Greek Energy Forecasting API",
    description="24-hour electricity demand forecasting for Greece",
    version="0.1.0",
)

_dataset = None


def _get_dataset() -> pd.DataFrame:
    """Load dataset on first request, cache for subsequent ones."""
    global _dataset
    if _dataset is None:
        logger.info("Loading dataset for API")
        path = settings.data.features_dir / "dataset.parquet"
        _dataset = pd.read_parquet(path)
        logger.info(f"Dataset loaded: {len(_dataset)} rows")
    return _dataset


def _extend_with_skeleton(df, cutoff_idx):
    """Extend dataset with skeleton rows for future hours."""
    rows_after = len(df) - cutoff_idx - 1
    if rows_after >= 24:
        return df

    last_ts = df.iloc[-1]["timestamp"]
    missing_hours = 24 - rows_after
    logger.info(f"  Extending with {missing_hours} skeleton rows")

    future_timestamps = pd.date_range(
        start=last_ts + pd.Timedelta(hours=1),
        periods=missing_hours,
        freq="h",
    )
    skeleton = pd.DataFrame({"timestamp": future_timestamps})
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

    weather_cols = [
        "temperature_c", "humidity_pct", "solar_radiation_w_m2",
        "wind_speed_kmh", "precipitation_mm",
        "hdd", "cdd", "temp_squared", "apparent_temp",
    ]
    for col in weather_cols:
        if col in df.columns:
            skeleton[col] = df[col].iloc[-1]

    skeleton["actual_load_mw"] = np.nan
    if "forecast_load_mw" in df.columns:
        skeleton["forecast_load_mw"] = np.nan

    for col in df.columns:
        if col not in skeleton.columns:
            skeleton[col] = 0

    skeleton = skeleton[df.columns]
    return pd.concat([df, skeleton], ignore_index=True)


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check if the API is running and data is available."""
    df = _get_dataset()
    return HealthResponse(
        status="healthy",
        dataset_rows=len(df),
        available_models=len(get_available_models()),
    )


@app.get("/models", response_model=ModelsResponse)
def list_models():
    """List all available forecasting models."""
    models = []
    for name in get_available_models():
        config = get_model_config(name)
        models.append(ModelInfo(
            name=name,
            model_family=config["model_family"],
            feature_mode=config["feature_mode"],
            weather_mode=config["weather_mode"],
            training_strategy=config["training_strategy"],
        ))
    return ModelsResponse(models=models)


@app.post("/forecast", response_model=ForecastResponse)
def generate_forecast(request: ForecastRequest):
    """Generate a 24-hour electricity demand forecast."""
    model_name = request.model

    try:
        config = get_model_config(model_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    df = _get_dataset().copy()

    cutoff_idx = None
    for i in range(len(df) - 25, max(len(df) - 200, 0), -1):
        ts = df.iloc[i]["timestamp"]
        if hasattr(ts, "hour"):
            hour = ts.hour
        else:
            hour = pd.Timestamp(ts).hour
        if hour == 10:
            cutoff_idx = i
            break

    if cutoff_idx is None:
        raise HTTPException(status_code=500, detail="No valid cutoff found")

    try:
        df = _extend_with_skeleton(df, cutoff_idx)

        pipeline, needs_lags = config["factory"]()

        if needs_lags:
            df = pipeline.add_lag_features(df)
            lag_cols = [c for c in df.columns if "lag" in c or "rolling" in c]
            df[lag_cols] = df[lag_cols].ffill()

        if config["training_strategy"] != "none":
            pipeline.train(df, train_end_idx=cutoff_idx)

        result = pipeline.predict(df, cutoff_idx=cutoff_idx)

    except Exception as e:
        logger.error(f"Forecast failed for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    cutoff_ts = str(df.iloc[cutoff_idx]["timestamp"])

    artifacts_dir = Path("mlflow_artifacts_tmp")
    predictions_file = artifacts_dir / f"{model_name}_predictions.parquet"

    lower_bounds = {}
    upper_bounds = {}

    if predictions_file.exists():
        try:
            intervals = ConformalIntervals.from_parquet(
                str(predictions_file), confidence=0.90
            )
            result_with_intervals = intervals.predict(result)
            for _, row in result_with_intervals.iterrows():
                step = int(row["horizon_step"])
                lower_bounds[step] = round(float(row["lower_bound"]), 1)
                upper_bounds[step] = round(float(row["upper_bound"]), 1)
        except Exception as e:
            logger.warning(f"Could not compute intervals: {e}")

    predictions = []
    for i, (_, row) in enumerate(result.iterrows()):
        step = i + 1
        predictions.append(HourlyForecast(
            timestamp=str(row["timestamp"]),
            predicted_load_mw=round(float(row["predicted_load_mw"]), 1),
            horizon_step=step,
            lower_bound=lower_bounds.get(step),
            upper_bound=upper_bounds.get(step),
        ))

    return ForecastResponse(
        model=model_name,
        cutoff_timestamp=cutoff_ts,
        horizon=len(predictions),
        weather_mode=config["weather_mode"],
        training_strategy=config["training_strategy"],
        predictions=predictions,
    )