"""Standard metric definitions for forecast evaluation.

All models use these exact definitions to ensure
comparable results. Metrics are computed over all
individual forecast observations (not averaged
per cutoff first).
"""

import numpy as np
import pandas as pd


def compute_metrics(results_df: pd.DataFrame) -> dict:
    """Compute MAE, MAPE, RMSE from a results DataFrame.

    Args:
        results_df: Must contain 'actual_load_mw' and
                    'predicted_load_mw' columns. Rows with
                    NaN in either column are dropped.

    Returns:
        Dictionary with mae, mape, rmse, and n_observations.

    Raises:
        ValueError: If no valid observations remain.
    """

    df = results_df.dropna(
        subset=["actual_load_mw", "predicted_load_mw"]
    )

    if df.empty:
        raise ValueError("No valid observations for metric computation")

    actual = df["actual_load_mw"].values
    predicted = df["predicted_load_mw"].values

    errors = actual - predicted
    abs_errors = np.abs(errors)

    # Guard against division by zero in MAPE
    nonzero_mask = actual != 0
    if nonzero_mask.sum() == 0:
        raise ValueError("All actual values are zero — cannot compute MAPE")

    mae = abs_errors.mean()
    rmse = np.sqrt((errors ** 2).mean())
    mape = (abs_errors[nonzero_mask] / np.abs(actual[nonzero_mask])).mean() * 100

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "n_observations": int(len(df)),
    }