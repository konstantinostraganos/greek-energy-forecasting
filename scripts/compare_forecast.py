"""Compare a live forecast with actual values.

Loads the forecast file and matches predictions against
actuals from the updated dataset (not from the forecast
file, which may have NaN for skeleton rows).
"""

import pandas as pd
from energy_forecasting.config import settings

# Load forecast
forecast_df = pd.read_parquet("data/forecasts/xgboost_2026-10-05.parquet")

# Load updated dataset for actuals
dataset = pd.read_parquet(settings.data.features_dir / "dataset.parquet")

# Merge to get fresh actuals
merged = forecast_df[["timestamp", "predicted_load_mw"]].merge(
    dataset[["timestamp", "actual_load_mw"]],
    on="timestamp",
    how="left",
)

merged["error_mw"] = (merged["actual_load_mw"] - merged["predicted_load_mw"]).round(0)
merged["pct_error"] = (
    (merged["actual_load_mw"] - merged["predicted_load_mw"]).abs()
    / merged["actual_load_mw"] * 100
).round(2)

print("\n=== FORECAST vs ACTUAL ===")
print(f"{'Timestamp':<28} {'Predicted':>10} {'Actual':>10} {'Error':>8} {'%':>7}")
print("-" * 65)

for _, row in merged.iterrows():
    actual = f"{row['actual_load_mw']:>10.0f}" if pd.notna(row['actual_load_mw']) else "     N/A"
    error = f"{row['error_mw']:>8.0f}" if pd.notna(row['error_mw']) else "    N/A"
    pct = f"{row['pct_error']:>6.2f}%" if pd.notna(row['pct_error']) else "   N/A"
    
    print(
        f"{str(row['timestamp']):<28} "
        f"{row['predicted_load_mw']:>10.0f} "
        f"{actual} "
        f"{error} "
        f"{pct}"
    )

# Compute metrics only for rows with actuals
valid = merged.dropna(subset=["actual_load_mw"])
if len(valid) > 0:
    print(f"\nValid hours: {len(valid)}/24")
    print(f"MAE:  {valid['error_mw'].abs().mean():.0f} MW")
    print(f"MAPE: {valid['pct_error'].mean():.2f}%")
else:
    print("\nNo actuals available yet — try again later")