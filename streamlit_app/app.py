"""Greek Energy Forecasting Dashboard.

Main entry point for the Streamlit application.
Pages are auto-discovered from the pages/ directory.

Run with:
    streamlit run streamlit_app/app.py
"""

import streamlit as st


st.set_page_config(
    page_title="Greek Energy Forecasting",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Greek Energy Forecasting")
st.markdown(
    "24-hour electricity demand forecasting for Greece, "
    "using ENTSO-E data and multiple ML models."
)

st.markdown("---")

st.subheader("Available Pages")

st.markdown(
    "- **Forecast** — Generate 24h demand predictions\n"
    "- **Model Comparison** — Compare model performance\n"
    "- **Historical Data** — Explore demand patterns"
)

st.markdown("---")

st.caption(
    "Data: ENTSO-E Transparency Platform · "
    "Weather: Open-Meteo API · "
    "Models: XGBoost, LightGBM, LSTM, PatchTST, TiDE"
)