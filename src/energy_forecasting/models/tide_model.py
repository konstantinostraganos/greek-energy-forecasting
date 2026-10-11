"""TiDE-based electricity demand forecaster.

Uses the Time-series Dense Encoder (TiDE) architecture from
NeuralForecast. TiDE was proposed by Google Research (2023) as
a simpler alternative to Transformer-based forecasters:

- Dense encoder compresses historical data + covariates
  into a compact latent representation
- Dense decoder maps the latent space to multi-step forecasts
- Residual connections stabilize training
- Future covariates (weather, calendar) are projected and
  combined with the decoder output

Uses the same exogenous features as tree-based models,
enabling a direct deep learning vs XGBoost comparison
on equal footing.

Historical evaluation uses observed future weather as an
oracle feature. Production forecasting must use weather
forecasts available at the forecast cutoff.
"""

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import TiDE
from loguru import logger


# Exogenous features available at forecast time (weather + calendar)
EXOG_FEATURES = [
    "temperature_c", "humidity_pct", "solar_radiation_w_m2",
    "wind_speed_kmh", "precipitation_mm",
    "hdd", "cdd", "temp_squared", "apparent_temp",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    "is_weekend", "is_holiday",
    "is_day_before_holiday", "is_day_after_holiday",
]


class TiDEPipeline:
    """Training and prediction pipeline for TiDE.

    Converts data to NeuralForecast format, trains TiDE with
    exogenous features, and provides the same predict() interface
    as other models.

    TiDE architecture:
        Inputs (168h history + 19 exogenous features)
        → Feature projection (per-covariate MLPs)
        → Dense encoder with residual connections
        → Latent representation
        → Dense decoder with residual connections
        → Temporal decoder (per-horizon refinement)
        → 24 hour predictions

    Research question: Can a DL encoder-decoder with the same
    handcrafted features beat XGBoost/LightGBM?
    """

    SEQ_LEN = 168
    HORIZON = 24

    def __init__(self, max_steps=500, learning_rate=0.001):
        """Initialize TiDE configuration.

        Args:
            max_steps: Training iterations (not epochs — NeuralForecast
                       uses steps, which are individual batch updates)
            learning_rate: Adam optimizer learning rate
        """

        self.max_steps = max_steps
        self.learning_rate = learning_rate

        self.nf = None
        self.is_fitted = False

    def _validate_features(self, df: pd.DataFrame) -> None:
        """Check that all required features exist in the dataframe."""

        required = {"timestamp", "actual_load_mw"} | set(EXOG_FEATURES)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}. "
                f"Run FeatureEngineer.build_features() first."
            )

    def _to_nixtla_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert our dataset to NeuralForecast format.

        NeuralForecast requires:
        - unique_id: identifier for each time series
        - ds: timestamp column
        - y: target variable
        - plus any exogenous features
        """

        nf_df = pd.DataFrame({
            "unique_id": "GR",
            "ds": df["timestamp"].values,
            "y": df["actual_load_mw"].values,
        })

        for col in EXOG_FEATURES:
            nf_df[col] = df[col].values

        return nf_df

    def train(self, df: pd.DataFrame, train_end_idx: int) -> None:
        """Train TiDE on data up to train_end_idx."""

        logger.info("Preparing TiDE training data")
        self._validate_features(df)

        train_df = df.iloc[:train_end_idx + 1].copy()

        # Check for NaN in required columns
        required_cols = ["actual_load_mw"] + EXOG_FEATURES
        nan_mask = train_df[required_cols].isna().any(axis=1)
        nan_count = nan_mask.sum()

        if nan_count > 0:
            logger.warning(
                f"  {nan_count} rows with NaN in required columns — "
                f"dropping (data should be pre-cleaned by validator)"
            )
            train_df = train_df.dropna(subset=required_cols)

        nf_train = self._to_nixtla_format(train_df)

        model = TiDE(
            h=self.HORIZON,
            input_size=self.SEQ_LEN,
            futr_exog_list=EXOG_FEATURES,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            scaler_type="robust",
            random_seed=42,
        )

        self.nf = NeuralForecast(
            models=[model],
            freq="h",
        )

        logger.info(
            f"Training TiDE: {len(nf_train)} rows, "
            f"{len(EXOG_FEATURES)} exogenous features, "
            f"max_steps={self.max_steps}"
        )

        self.nf.fit(df=nf_train)

        self.is_fitted = True
        logger.info("  TiDE training complete")

    def predict(self, df: pd.DataFrame, cutoff_idx: int) -> pd.DataFrame:
        """Generate 24h forecast from a cutoff point.

        Same interface as XGBoost, LSTM, and PatchTST pipelines.
        """

        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call train() first.")

        # Guard: need at least HORIZON rows after cutoff
        rows_available = len(df) - (cutoff_idx + 1)
        if rows_available < self.HORIZON:
            raise ValueError(
                f"Need {self.HORIZON} rows after cutoff_idx={cutoff_idx}, "
                f"but only {rows_available} available. "
                f"Choose an earlier cutoff."
            )

        # History: everything up to cutoff
        history_df = df.iloc[:cutoff_idx + 1].copy()
        nf_history = self._to_nixtla_format(history_df)

        # Future exogenous features (oracle weather for backtesting)
        forecast_start = cutoff_idx + 1
        forecast_end = forecast_start + self.HORIZON
        future_df = df.iloc[forecast_start:forecast_end].copy()

        nf_future = pd.DataFrame({
            "unique_id": "GR",
            "ds": future_df["timestamp"].values,
        })
        for col in EXOG_FEATURES:
            nf_future[col] = future_df[col].values

        forecast = self.nf.predict(df=nf_history, futr_df=nf_future)

        result = future_df[["timestamp", "actual_load_mw"]].copy()
        result["predicted_load_mw"] = forecast["TiDE"].values[:len(result)]

        return result[["timestamp", "predicted_load_mw", "actual_load_mw"]]