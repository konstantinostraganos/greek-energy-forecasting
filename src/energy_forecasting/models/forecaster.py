"""Forecasting pipeline with cutoff-aware feature generation.

Generates lag and rolling features respecting the forecast cutoff
to prevent data leakage. No future information is ever used.

Weather note: backtesting uses observed (oracle) weather for
future hours. Production should use weather forecasts instead.
"""

import pandas as pd
from loguru import logger
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


class ForecastPipeline:
    """Generates cutoff-aware features and runs forecasts.

    Key design decisions:
    - Load lags [24, 48, 168] are >= horizon (24h), guaranteeing
      no future load information leaks into features
    - Weather features use observed values (oracle backtest)
    - Training happens once on all data before eval period
    - Same predict() interface as baseline models
    """

    LOAD_LAGS = [24, 48, 168]
    HORIZON = 24

    STATIC_FEATURES = [
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "is_holiday",
        "is_day_before_holiday",
        "is_day_after_holiday",
        "hdd",
        "cdd",
        "temp_squared",
        "apparent_temp",
        "temperature_c",
        "humidity_pct",
        "solar_radiation_w_m2",
        "wind_speed_kmh",
        "precipitation_mm",
    ]

    def __init__(self, model=None):
        """Initialize with a model. Defaults to XGBoost."""

        if model is None:
            self.model = XGBRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            )
        else:
            self.model = model

        self.is_fitted = False
        self.feature_columns = None

    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add load lag features using shift().

        Safe for 24h forecasting because min(LOAD_LAGS) >= HORIZON.
        Each lag looks at least 24h back from the target timestamp,
        which is always at or before the forecast cutoff.
        """

        logger.info("Adding lag features")

        for lag in self.LOAD_LAGS:
            col_name = f"load_lag_{lag}h"
            df[col_name] = df["actual_load_mw"].shift(lag)
            logger.info(f"  {col_name}: {df[col_name].notna().sum()} valid values")

        # Rolling mean of last 24h (computed from lag_24 perspective)
        df["load_rolling_24h"] = (
            df["actual_load_mw"].shift(24).rolling(window=24, min_periods=24).mean()
        )

        # Rolling mean of last 168h (1 week)
        df["load_rolling_168h"] = (
            df["actual_load_mw"].shift(24).rolling(window=168, min_periods=168).mean()
        )

        logger.info("  Added 5 lag/rolling features")

        return df

    def _get_feature_columns(self):
        """Build the full feature list."""

        lag_features = [f"load_lag_{lag}h" for lag in self.LOAD_LAGS]
        rolling_features = ["load_rolling_24h", "load_rolling_168h"]

        return self.STATIC_FEATURES + lag_features + rolling_features

    def train(self, df: pd.DataFrame, train_end_idx: int) -> None:
        """Train model on data up to train_end_idx.

        Args:
            df: Full dataset with lag features already added
            train_end_idx: Last index to include in training
        """

        self.feature_columns = self._get_feature_columns()

        # Training data — only up to train_end_idx
        train_df = df.iloc[: train_end_idx + 1].copy()

        # Drop rows with NaN in features (from lag computation)
        train_df = train_df.dropna(subset=self.feature_columns + ["actual_load_mw"])

        X_train = train_df[self.feature_columns]
        y_train = train_df["actual_load_mw"]

        logger.info(f"Training on {len(X_train)} rows, {len(self.feature_columns)} features")

        self.model.fit(X_train, y_train)
        self.is_fitted = True

        # Training error as sanity check
        train_pred = self.model.predict(X_train)
        train_mae = mean_absolute_error(y_train, train_pred)
        logger.info(f"  Training MAE: {train_mae:.0f} MW")

    def predict(self, df: pd.DataFrame, cutoff_idx: int) -> pd.DataFrame:
        """Generate 24h forecast from a cutoff point.

        Uses the same interface as baseline models for
        consistent evaluation.

        Args:
            df: Full dataset with lag features
            cutoff_idx: Index of the last known observation

        Returns:
            DataFrame with timestamp, predicted_load_mw,
            and actual_load_mw for the next 24 hours
        """

        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call train() first.")

        forecast_start = cutoff_idx + 1
        forecast_end = min(cutoff_idx + 1 + self.HORIZON, len(df))
        future = df.iloc[forecast_start:forecast_end].copy()

        # Validate — no missing features allowed
        X_future = future[self.feature_columns]
        if X_future.isna().any().any():
            missing_cols = X_future.columns[X_future.isna().any()].tolist()
            raise ValueError(f"Missing features in forecast horizon: {missing_cols}")

        future["predicted_load_mw"] = self.model.predict(X_future)

        return future[["timestamp", "predicted_load_mw", "actual_load_mw"]]
