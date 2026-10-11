"""LSTM-based electricity demand forecaster.

Uses a sequence of 168 past hours to directly predict
the next 24 hours (direct multi-step forecasting).

Historical evaluation uses observed future weather as an
oracle feature. Production forecasting must use weather
forecasts available at the forecast cutoff.
"""

import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# Features the LSTM sees in its input sequence
SEQUENCE_FEATURES = [
    "actual_load_mw",
    "temperature_c",
    "humidity_pct",
    "solar_radiation_w_m2",
    "wind_speed_kmh",
    "precipitation_mm",
    "hdd",
    "cdd",
    "temp_squared",
    "apparent_temp",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "is_holiday",
    "is_day_before_holiday",
    "is_day_after_holiday",
]


class EnergyDataset(Dataset):
    """PyTorch Dataset that creates 168h→24h samples.

    For each valid position in the data, creates:
    - Input: 168 hours of features (X)
    - Target: next 24 hours of actual load (y)
    """

    def __init__(self, X, y, seq_len=168, forecast_horizon=24):
        """
        Args:
            X: Feature array (n_samples, n_features)
            y: Target array (n_samples,) — actual_load_mw
            seq_len: Input sequence length (default: 1 week)
            forecast_horizon: Output length (default: 24 hours)
        """

        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon

    def __len__(self):
        # Ensure every sample has full input AND full target
        return len(self.X) - self.seq_len - self.forecast_horizon + 1

    def __getitem__(self, idx):
        # Input: seq_len hours of features
        x_seq = self.X[idx : idx + self.seq_len]

        # Target: next forecast_horizon hours of load
        target_start = idx + self.seq_len
        y_seq = self.y[target_start : target_start + self.forecast_horizon]

        return x_seq, y_seq


class LSTMForecaster(nn.Module):
    """LSTM neural network for 24h demand forecasting.

    Architecture:
        Input (168 × n_features)
        → LSTM (2 layers, hidden_size=64)
        → take last hidden state
        → Linear (64 → 32) + ReLU
        → Linear (32 → 24)
        → Output (24 hourly predictions)
    """

    def __init__(self, n_features, hidden_size=64, num_layers=2, forecast_horizon=24):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, forecast_horizon),
        )

    def forward(self, x):
        # x shape: (batch, seq_len, n_features)
        lstm_out, _ = self.lstm(x)

        # Take only the last timestep output
        last_hidden = lstm_out[:, -1, :]

        # Predict all 24 hours at once
        output = self.fc(last_hidden)

        return output


class LSTMPipeline:
    """Training and prediction pipeline for LSTM forecaster.

    Handles scaling, training, and prediction with the same
    interface as the XGBoost ForecastPipeline.
    """

    SEQ_LEN = 168
    HORIZON = 24

    def __init__(self, epochs=50, batch_size=64, learning_rate=0.001):
        """Initialize pipeline parameters."""

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"LSTM Pipeline using device: {self.device}")

    def train(self, df: pd.DataFrame, train_end_idx: int) -> None:
        """Train LSTM on data up to train_end_idx.

        Scaler is fit ONLY on training data to prevent leakage.
        Last 10% of training data used as temporal validation set.
        """

        logger.info("Preparing LSTM training data")

        # Extract features and target — training data only
        train_df = df.iloc[: train_end_idx + 1].copy()
        train_df = train_df.dropna(subset=SEQUENCE_FEATURES)

        X_raw = train_df[SEQUENCE_FEATURES].values
        y_raw = train_df["actual_load_mw"].values.reshape(-1, 1)

        # Fit scalers ONLY on training data
        X_scaled = self.scaler_X.fit_transform(X_raw)
        y_scaled = self.scaler_y.fit_transform(y_raw).flatten()

        # Split training data into train/validation (temporal, not random)
        val_size = int(len(X_scaled) * 0.1)
        train_size = len(X_scaled) - val_size

        train_dataset = EnergyDataset(
            X_scaled[:train_size],
            y_scaled[:train_size],
            seq_len=self.SEQ_LEN,
            forecast_horizon=self.HORIZON,
        )
        val_dataset = EnergyDataset(
            X_scaled[train_size - self.SEQ_LEN :],
            y_scaled[train_size - self.SEQ_LEN :],
            seq_len=self.SEQ_LEN,
            forecast_horizon=self.HORIZON,
        )

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Initialize model
        n_features = len(SEQUENCE_FEATURES)
        self.model = LSTMForecaster(
            n_features=n_features,
            forecast_horizon=self.HORIZON,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        # Training loop
        logger.info(
            f"Training LSTM: {len(train_dataset)} train samples, "
            f"{len(val_dataset)} val samples, "
            f"{self.epochs} epochs, {n_features} features"
        )

        for epoch in range(self.epochs):
            # Train
            self.model.train()
            train_loss = 0
            n_train = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                n_train += 1

            # Validate
            self.model.eval()
            val_loss = 0
            n_val = 0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    predictions = self.model(X_batch)
                    loss = criterion(predictions, y_batch)

                    val_loss += loss.item()
                    n_val += 1

            if (epoch + 1) % 10 == 0:
                avg_train = train_loss / n_train
                avg_val = val_loss / n_val
                logger.info(
                    f"  Epoch {epoch + 1}/{self.epochs} — "
                    f"Train Loss: {avg_train:.6f}, Val Loss: {avg_val:.6f}"
                )

        self.is_fitted = True
        logger.info("  LSTM training complete")

    def predict(self, df: pd.DataFrame, cutoff_idx: int) -> pd.DataFrame:
        """Generate 24h forecast from a cutoff point.

        Same interface as XGBoost ForecastPipeline.predict()
        for consistent evaluation.
        """

        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call train() first.")

        # Input: last SEQ_LEN hours up to and including cutoff
        seq_start = cutoff_idx - self.SEQ_LEN + 1
        if seq_start < 0:
            raise ValueError(
                f"Not enough history: need {self.SEQ_LEN} hours, only {cutoff_idx + 1} available"
            )

        # Get input sequence
        input_df = df.iloc[seq_start : cutoff_idx + 1]
        X_raw = input_df[SEQUENCE_FEATURES].values

        # Scale using training-fitted scaler (transform only, no fit)
        X_scaled = self.scaler_X.transform(X_raw)

        # Convert to tensor and predict
        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            y_scaled = self.model(X_tensor).cpu().numpy().flatten()

        # Inverse scale predictions
        y_pred = self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()

        # Build output DataFrame
        forecast_start = cutoff_idx + 1
        forecast_end = min(cutoff_idx + 1 + self.HORIZON, len(df))
        future = df.iloc[forecast_start:forecast_end].copy()

        future["predicted_load_mw"] = y_pred[: len(future)]

        return future[["timestamp", "predicted_load_mw", "actual_load_mw"]]
