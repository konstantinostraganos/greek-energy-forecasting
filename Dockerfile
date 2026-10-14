# Base image — same Python version as development
FROM python:3.13-slim

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Working directory inside the container
WORKDIR /app

# Install system dependencies needed by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (avoids downloading 2GB of CUDA)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy dependency specification and install
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir -e .

# Copy application code only; data is mounted at runtime and secrets
# are injected by docker-compose, so neither is baked into the image
COPY configs/ configs/
COPY scripts/ scripts/
COPY mlflow_artifacts_tmp/ mlflow_artifacts_tmp/
COPY streamlit_app/ streamlit_app/

# Default command (overridden by docker-compose)
CMD ["uvicorn", "energy_forecasting.api.app:app", "--host", "0.0.0.0", "--port", "8000"]