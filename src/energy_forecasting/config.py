"""Central configuration management.

Loads settings from configs/default.yaml and .env,
validates them via Pydantic, and exposes a single
`settings` object used across the entire project.
"""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# Project root — everything is relative to this
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class EntsoEConfig(BaseModel):
    """ENTSO-E API settings."""

    base_url: str
    area_code: str
    start_date: str
    end_date: str


class WeatherLocation(BaseModel):
    """Single weather station with consumption weight."""

    name: str
    latitude: float
    longitude: float
    weight: float = Field(ge=0.0, le=1.0)


class WeatherConfig(BaseModel):
    """Open-Meteo API settings."""

    forecast_url: str
    archive_url: str
    locations: list[WeatherLocation]


class DataConfig(BaseModel):
    """Data directory paths."""

    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    features_dir: Path = Path("data/features")
    models_dir: Path = Path("models")


class Settings(BaseSettings):
    """Main application settings.

    Loads secrets from .env, project config from YAML,
    and validates everything on startup.
    """

    # Secret — loaded from .env file, not from YAML
    entsoe_api_key: Optional[str] = None

    # Sub-configurations — loaded from YAML
    entso_e: EntsoEConfig
    weather: WeatherConfig
    data: DataConfig = DataConfig()

    model_config = {
        "env_file": str(PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


def load_settings(config_path: Optional[Path] = None) -> Settings:
    """Load and validate settings from YAML + .env."""

    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "default.yaml"

    with open(config_path, "r") as f:
        yaml_config = yaml.safe_load(f)

    return Settings(**yaml_config)


# Singleton — import this everywhere
settings = load_settings()