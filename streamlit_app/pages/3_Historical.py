"""Historical data exploration page."""

import streamlit as st
import pandas as pd
from energy_forecasting.config import settings

st.set_page_config(page_title="Historical Data", page_icon="📋", layout="wide")

st.title("📋 Historical Data Explorer")


@st.cache_data
def load_dataset():
    """Load the feature-engineered dataset."""
    path = settings.data.features_dir / "dataset.parquet"
    return pd.read_parquet(path)


df = load_dataset()

st.metric("Total Hours", f"{len(df):,}")
st.metric("Date Range", f"{df['timestamp'].min()} → {df['timestamp'].max()}")

st.markdown("---")

# Demand over time
st.subheader("Demand Over Time")
time_df = df.set_index("timestamp")["actual_load_mw"]

# Resample to daily mean for readability
daily = time_df.resample("D").mean()
st.line_chart(daily)

st.markdown("---")

# Daily pattern
st.subheader("Average Daily Pattern")
hourly_avg = df.groupby("hour")["actual_load_mw"].mean()
st.bar_chart(hourly_avg)

st.markdown("---")

# Temperature vs demand
st.subheader("Temperature vs Demand")
scatter_df = df[["temperature_c", "actual_load_mw"]].dropna().sample(5000)
st.scatter_chart(scatter_df, x="temperature_c", y="actual_load_mw")