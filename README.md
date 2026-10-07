# Greek Electricity Demand Forecasting

End-to-end machine learning platform for 24-hour electricity demand forecasting in Greece, using real data from the ENTSO-E Transparency Platform and Open-Meteo weather API.

## Key Results

| Model | Family | MAPE | MAE (MW) | RMSE (MW) | Strategy |
|---|---|---|---|---|---|
| **XGBoost** | tree | **2.14%** | **139** | **186** | expanding |
| **LightGBM** | tree | **2.19%** | **142** | **192** | expanding |
| PatchTST | transformer | 2.86% | 185 | 267 | fixed |
| TiDE | mlp | 3.34% | 220 | 318 | fixed |
| LSTM | recurrent | 3.65% | 245 | 314 | fixed |
| ENTSO-E (ADMIE) | external benchmark | 3.81% | 254 | 325 | — |
| Seasonal Naive | baseline | 6.03% | 399 | 583 | — |
| Naive | baseline | 20.21% | 1199 | 1529 | — |

Evaluated on 89 daily cutoffs (10:00 UTC) with 24-hour horizon. All models tracked in MLflow.

**Live forecast validation:** XGBoost achieved **1.40% MAPE / 76 MW MAE** on a real 24-hour production forecast (cutoff: 5 Oct 2026).

> **Note:** Tree-based models use oracle weather (observed future weather). The ENTSO-E benchmark uses proprietary weather forecasts and a different forecast origin. The comparison is indicative, not controlled.

## Architecture

```
ENTSO-E API ──► Data Pipeline ──► Feature Engineering ──► Models ──► Evaluation
Open-Meteo  ──┘                                             │           │
                                                            ▼           ▼
                                                        FastAPI      MLflow
                                                            │
                                                        Streamlit
                                                        Dashboard
```

## Project Structure

```
greek-energy-forecasting/
├── src/energy_forecasting/
│   ├── config.py                    # Settings and paths
│   ├── logger.py                    # Logging configuration
│   ├── data/
│   │   ├── entso_e_client.py        # ENTSO-E API client
│   │   ├── weather_client.py        # Open-Meteo API client
│   │   ├── validator.py             # Data validation and merging
│   │   └── feature_engineer.py      # Feature engineering pipeline
│   ├── models/
│   │   ├── baselines.py             # Naive and Seasonal Naive
│   │   ├── forecaster.py            # XGBoost/LightGBM pipeline
│   │   ├── lstm_model.py            # LSTM pipeline
│   │   ├── transformer_model.py     # PatchTST pipeline
│   │   ├── tide_model.py            # TiDE pipeline
│   │   └── entso_e_benchmark.py     # ENTSO-E day-ahead benchmark
│   ├── evaluation/
│   │   ├── metrics.py               # MAE, MAPE, RMSE
│   │   ├── model_registry.py        # Model configs and factories
│   │   ├── evaluate.py              # Unified backtesting evaluator
│   │   └── prediction_intervals.py  # Conformal prediction intervals
│   └── api/
│       ├── app.py                   # FastAPI application
│       └── schemas.py               # Pydantic request/response models
├── streamlit_app/
│   ├── app.py                       # Dashboard entry point
│   └── pages/
│       ├── 1_Forecast.py            # 24h forecast with intervals
│       ├── 2_Model_Comparison.py    # Performance comparison
│       └── 3_Historical.py          # EDA charts
├── scripts/
│   ├── run_evaluation.py            # Run backtesting evaluation
│   ├── run_live_forecast.py         # Live production forecast
│   ├── compare_forecast.py          # Compare forecast vs actuals
│   ├── download_data.py             # Download raw data
│   ├── run_validation.py            # Run validation pipeline
│   └── run_features.py              # Run feature engineering
├── tests/                           # 45 tests across all modules
├── configs/default.yaml             # Default configuration
├── Dockerfile                       # Docker image (CPU-only PyTorch)
├── docker-compose.yml               # API + Dashboard services
├── pyproject.toml                   # Project dependencies
└── .env.example                             # API keys (not tracked)
```

## Features

- **27 engineered features**: lag features (24h, 48h, 168h), rolling averages, cyclical encodings (hour, day, month), Greek holiday calendar (including moving Easter), weather-derived features (HDD, CDD, apparent temperature)
- **Data leakage prevention**: all lags use `shift(24)` before rolling; min(lag) >= horizon; scaler fit only on train data; documented oracle weather assumption
- **Expanding-window backtesting**: XGBoost/LightGBM retrain at every cutoff using only data available up to that point (89 retrains). Neural models use fixed strategy (train once) due to compute cost — clearly labeled in MLflow
- **Conformal prediction intervals**: 90% coverage from empirical backtest residuals per horizon step
- **Live production pipeline**: downloads fresh ENTSO-E load + Open-Meteo weather, extends dataset with skeleton rows for forecast horizon, generates and saves predictions
- **REST API**: FastAPI with `/health`, `/models`, `/forecast` endpoints; auto-generated Swagger docs
- **Interactive dashboard**: Streamlit with forecast visualization (including prediction intervals), model comparison, and historical data exploration
- **Experiment tracking**: MLflow logs parameters, metrics, tags, and artifacts for every evaluation run
- **Containerized**: Docker Compose runs API + Dashboard with one command

## Data Sources

| Source | Data | Period |
|---|---|---|
| [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) | Actual load + day-ahead forecast for Greece | 2020–present |
| [Open-Meteo](https://open-meteo.com/) | Hourly weather for 5 Greek cities (population-weighted average: Athens 0.50, Thessaloniki 0.20, Patras 0.10, Heraklion 0.10, Larissa 0.10) | 2020–present |

## Quick Start

### Option 1: Docker (recommended)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/greek-energy-forecasting.git
cd greek-energy-forecasting

# Add your ENTSO-E API key
cp .env.example .env
# Edit .env and add your key

# Start API + Dashboard
docker compose up

# API:       http://localhost:8000/docs
# Dashboard: http://localhost:8501
```

### Option 2: Local Development

```bash
# Clone and setup
git clone https://github.com/YOUR_USERNAME/greek-energy-forecasting.git
cd greek-energy-forecasting
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -e .

# Add your ENTSO-E API key
cp .env.example .env
# Edit .env and add your key

# Download data
python scripts/download_data.py

# Run validation and feature engineering
python scripts/run_validation.py
python scripts/run_features.py

# Run evaluation
python scripts/run_evaluation.py --model all

# Start API
uvicorn energy_forecasting.api.app:app --reload

# Start Dashboard (new terminal)
streamlit run streamlit_app/app.py
```

## Usage

### Evaluate Models

```bash
# Evaluate a single model
python scripts/run_evaluation.py --model xgboost

# Evaluate all models
python scripts/run_evaluation.py --model all

# List available models
python scripts/run_evaluation.py --list

# View results in MLflow
mlflow ui
# Open http://localhost:5000
```

### Live Forecast

```bash
# Download fresh data and generate forecast
python scripts/run_live_forecast.py --model xgboost

# Quick forecast (skip data download)
python scripts/run_live_forecast.py --skip-update --model xgboost

# Compare forecast vs actuals (after data becomes available)
python scripts/compare_forecast.py
```

### API

```bash
# Start the API
uvicorn energy_forecasting.api.app:app --reload

# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/models

# Generate forecast
curl -X POST http://localhost:8000/forecast \
     -H "Content-Type: application/json" \
     -d '{"model": "xgboost"}'
```

## Key Findings

1. **Tree-based models dominate** at this data scale (~58K rows). XGBoost (2.14% MAPE) outperforms all deep learning models, consistent with time-series forecasting literature for tabular datasets.

2. **Expanding-window retraining improves performance.** XGBoost improved from 2.25% to 2.14% MAPE with expanding strategy versus fixed, proving the value of incorporating recent data.

3. **PatchTST (univariate) beats TiDE (19 features).** Self-attention on raw load signal (2.86%) outperforms a DL model with full exogenous features (3.34%), suggesting that for this problem, the temporal patterns in the load signal itself are more informative than explicit weather/calendar features for neural architectures.

4. **LSTM shows overfitting.** Validation loss bottoms at epoch 10 while training loss continues decreasing — classic overfitting signature at this data scale.

5. **ENTSO-E comparison is indicative.** Our models achieve lower MAPE than the ADMIE benchmark (3.81%), but the comparison is not controlled — different weather inputs (oracle vs proprietary forecast), different cutoff times, and different forecast origins.

## Known Limitations

- **Oracle weather**: backtesting uses observed future weather, giving an optimistic estimate. Production performance with weather forecasts would be slightly worse
- **ENTSO-E data delay**: the TSO publishes actual load with ~24-48h delay, meaning live forecasts always start from a cutoff 1-2 days behind real-time
- **Neural model strategy**: LSTM, PatchTST, and TiDE use fixed training strategy (train once) due to CPU compute constraints. Expanding-window evaluation would require GPU
- **No model persistence**: neural models retrain from scratch on every live forecast run. A save/load mechanism would make them practical for daily use

## Tech Stack

- **Python 3.13** with type hints
- **ML/DL**: XGBoost, LightGBM, PyTorch, NeuralForecast (PatchTST, TiDE)
- **Data**: pandas, NumPy, scikit-learn
- **API**: FastAPI, uvicorn, Pydantic
- **Dashboard**: Streamlit
- **Experiment Tracking**: MLflow
- **Containerization**: Docker, Docker Compose
- **Testing**: pytest (45 tests)

## License

MIT