"""Model comparison page — view evaluation results."""

import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Model Comparison", page_icon="📊", layout="wide")

st.title("📊 Model Comparison")

# Load results from MLflow artifacts
artifacts_dir = Path("mlflow_artifacts_tmp")


@st.cache_data
def load_all_results():
    """Load all model predictions from artifact parquet files."""

    results = {}
    if not artifacts_dir.exists():
        return results

    for f in artifacts_dir.glob("*_predictions.parquet"):
        model_name = f.stem.replace("_predictions", "")
        df = pd.read_parquet(f)
        results[model_name] = df

    return results


results = load_all_results()

if not results:
    st.warning(
        "No evaluation results found. Run evaluations first:\n\n"
        "`python scripts/run_evaluation.py --model all`"
    )
    st.stop()

# Compute metrics for each model
metrics_data = []
for model_name, df in results.items():
    actual = df["actual_load_mw"]
    predicted = df["predicted_load_mw"]
    errors = (actual - predicted).abs()

    mae = errors.mean()
    rmse = ((actual - predicted) ** 2).mean() ** 0.5
    mape = (errors / actual).mean() * 100

    metrics_data.append({
        "Model": model_name,
        "MAPE (%)": round(mape, 2),
        "MAE (MW)": round(mae, 0),
        "RMSE (MW)": round(rmse, 0),
    })

metrics_df = pd.DataFrame(metrics_data).sort_values("MAPE (%)")

# Display table
st.subheader("Performance Summary")
st.dataframe(metrics_df, use_container_width=True, hide_index=True)

# Bar chart
st.subheader("MAPE Comparison")
chart_df = metrics_df.set_index("Model")["MAPE (%)"]
st.bar_chart(chart_df)

# MAE chart
st.subheader("MAE Comparison")
mae_df = metrics_df.set_index("Model")["MAE (MW)"]
st.bar_chart(mae_df)