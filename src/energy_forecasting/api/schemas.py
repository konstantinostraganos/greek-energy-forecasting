"""Request and response schemas for the forecasting API.

Pydantic models that define the exact shape of API
inputs and outputs. FastAPI uses these for automatic
validation, serialization, and Swagger documentation.
"""

from pydantic import BaseModel


class ForecastRequest(BaseModel):
    """Request body for the /forecast endpoint."""

    model: str = "xgboost"


class HourlyForecast(BaseModel):
    """A single hourly prediction."""

    timestamp: str
    predicted_load_mw: float
    horizon_step: int
    lower_bound: float | None = None
    upper_bound: float | None = None

class ForecastResponse(BaseModel):
    """Response from the /forecast endpoint."""

    model: str
    cutoff_timestamp: str
    horizon: int
    weather_mode: str
    training_strategy: str
    predictions: list[HourlyForecast]


class ModelInfo(BaseModel):
    """Information about a single model."""

    name: str
    model_family: str
    feature_mode: str
    weather_mode: str
    training_strategy: str


class ModelsResponse(BaseModel):
    """Response from the /models endpoint."""

    models: list[ModelInfo]


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""

    status: str
    dataset_rows: int
    available_models: int