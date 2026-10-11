"""PatchTST-based electricity demand forecaster (univariate).

Uses the PatchTST architecture from NeuralForecast library.
PatchTST splits the input sequence into patches (chunks)
and applies Transformer self-attention across patches,
capturing long-range dependencies more efficiently than
processing each hour individually.

This is a UNIVARIATE model — it uses only the historical
load signal, with no exogenous features. This tests whether
the Transformer's self-attention can learn temporal patterns
(daily/weekly cycles, weather effects) purely from the
demand history, competing against tree-based models that
use 27 handcrafted features.
"""

import pandas as pd
from loguru import logger
from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST


class TransformerPipeline:
    """Training and prediction pipeline for PatchTST (univariate).

    Converts data to NeuralForecast format, trains PatchTST
    on raw load only, and provides the same predict() interface
    as other models.

    PatchTST architecture:
        Input sequence (168 hours = 7 days)
        → split into patches (e.g., 24h each = 7 patches)
        → Transformer encoder with self-attention across patches
        → Linear head
        → 24 hour predictions

    Research question: Can a Transformer learn demand patterns
    from raw load alone, without handcrafted features?
    """

    SEQ_LEN = 168
    HORIZON = 24

    def __init__(self, max_steps=500, learning_rate=0.001, patch_len=24):
        """Initialize PatchTST configuration.

        Args:
            max_steps: Training iterations (not epochs — NeuralForecast
                       uses steps, which are individual batch updates)
            learning_rate: Adam optimizer learning rate
            patch_len: Size of each patch (24 = one daily cycle per patch)
        """

        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.patch_len = patch_len

        self.nf = None
        self.is_fitted = False

    def _to_nixtla_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert our dataset to NeuralForecast format.

        NeuralForecast requires:
        - unique_id: identifier for each time series
        - ds: timestamp column
        - y: target variable
        """

        return pd.DataFrame(
            {
                "unique_id": "GR",
                "ds": df["timestamp"].values,
                "y": df["actual_load_mw"].values,
            }
        )

    def train(self, df: pd.DataFrame, train_end_idx: int) -> None:
        """Train PatchTST on data up to train_end_idx."""

        logger.info("Preparing PatchTST training data")

        train_df = df.iloc[: train_end_idx + 1].copy()

        # Drop NaN only in the target column
        nan_count = train_df["actual_load_mw"].isna().sum()
        if nan_count > 0:
            logger.warning(
                f"  {nan_count} rows with NaN in actual_load_mw — "
                f"dropping (data should be pre-cleaned by validator)"
            )
            train_df = train_df.dropna(subset=["actual_load_mw"])

        nf_train = self._to_nixtla_format(train_df)

        # PatchTST — univariate, no exogenous features
        model = PatchTST(
            h=self.HORIZON,
            input_size=self.SEQ_LEN,
            patch_len=self.patch_len,
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
            f"Training PatchTST (univariate): {len(nf_train)} rows, "
            f"patch_len={self.patch_len}, "
            f"max_steps={self.max_steps}"
        )

        self.nf.fit(df=nf_train)

        self.is_fitted = True
        logger.info("  PatchTST training complete")

    def predict(self, df: pd.DataFrame, cutoff_idx: int) -> pd.DataFrame:
        """Generate 24h forecast from a cutoff point.

        Same interface as XGBoost, LSTM, and TiDE pipelines.
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
        history_df = df.iloc[: cutoff_idx + 1].copy()
        nf_history = self._to_nixtla_format(history_df)

        # Predict — no future exogenous needed
        forecast = self.nf.predict(df=nf_history)

        # Build output DataFrame matching other models
        forecast_start = cutoff_idx + 1
        forecast_end = forecast_start + self.HORIZON
        future_df = df.iloc[forecast_start:forecast_end].copy()

        result = future_df[["timestamp", "actual_load_mw"]].copy()
        result["predicted_load_mw"] = forecast["PatchTST"].values[: len(result)]

        return result[["timestamp", "predicted_load_mw", "actual_load_mw"]]
