"""Forecast page — generate 24h demand predictions."""

import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Forecast", page_icon="📈", layout="wide")

st.title("📈 24h Demand Forecast")

import os
API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Fetch available models from API
@st.cache_data(ttl=60)
def get_models():
    """Fetch model list from FastAPI."""
    try:
        response = requests.get(f"{API_URL}/models", timeout=10)
        response.raise_for_status()
        return response.json()["models"]
    except Exception:
        return None


models = get_models()

if models is None:
    st.error(
        "Cannot connect to API. Start it with: "
        "`uvicorn energy_forecasting.api.app:app --reload`"
    )
    st.stop()

# Model selector
model_names = [m["name"] for m in models]
selected = st.selectbox(
    "Select model",
    model_names,
    index=model_names.index("xgboost") if "xgboost" in model_names else 0,
)

# Show model info
selected_info = next(m for m in models if m["name"] == selected)
col1, col2, col3 = st.columns(3)
col1.metric("Family", selected_info["model_family"])
col2.metric("Weather", selected_info["weather_mode"])
col3.metric("Strategy", selected_info["training_strategy"])

st.markdown("---")

# Generate forecast button
if st.button("Generate Forecast", type="primary"):
    with st.spinner(f"Running {selected} forecast..."):
        try:
            response = requests.post(
                f"{API_URL}/forecast",
                json={"model": selected},
                timeout=300,
            )
            response.raise_for_status()
            data = response.json()

            # Parse predictions into DataFrame
            predictions = pd.DataFrame(data["predictions"])
            predictions["timestamp"] = pd.to_datetime(predictions["timestamp"])
            predictions = predictions.set_index("timestamp")

            # Display chart with prediction intervals
            st.subheader(f"Forecast from {data['cutoff_timestamp']}")

            has_intervals = predictions["lower_bound"].notna().any()

            if has_intervals:
                chart_df = predictions[["predicted_load_mw", "lower_bound", "upper_bound"]]
                chart_df.columns = ["Forecast", "Lower (90%)", "Upper (90%)"]
                st.line_chart(chart_df)
            else:
                st.line_chart(predictions["predicted_load_mw"])
                st.caption("No prediction intervals available — run evaluation first.")

            # Display table
            st.subheader("Hourly Predictions")
            st.dataframe(
                predictions[["horizon_step", "predicted_load_mw"]].rename(
                    columns={
                        "horizon_step": "Hour",
                        "predicted_load_mw": "Load (MW)",
                    }
                ),
                use_container_width=True,
            )

            # Summary stats
            st.subheader("Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Min Load", f"{predictions['predicted_load_mw'].min():.0f} MW")
            col2.metric("Max Load", f"{predictions['predicted_load_mw'].max():.0f} MW")
            col3.metric("Mean Load", f"{predictions['predicted_load_mw'].mean():.0f} MW")

        except requests.exceptions.ConnectionError:
            st.error("API not running. Start FastAPI first.")
        except Exception as e:
            st.error(f"Forecast failed: {e}")