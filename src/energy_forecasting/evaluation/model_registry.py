"""Model registry — maps model names to their classes and configs.

Central place to define how each model is instantiated,
what training strategy it uses, and what feature mode
it operates in. Adding a new model means adding one
entry here.
"""


def get_model_config(model_name: str) -> dict:
    """Return configuration for a model by name.

    Each config contains:
        model_family: tree / recurrent / transformer / mlp / baseline
        feature_mode: full (27 features) / sequence (20) /
                      univariate (load only) / exogenous (19 weather+cal)
        weather_mode: oracle / none
        training_strategy: expanding / fixed / none
        factory: callable that returns (pipeline, needs_lags)
    """

    registry = {
        "naive": {
            "model_family": "baseline",
            "feature_mode": "none",
            "weather_mode": "none",
            "training_strategy": "none",
            "factory": _create_naive,
        },
        "seasonal_naive": {
            "model_family": "baseline",
            "feature_mode": "none",
            "weather_mode": "none",
            "training_strategy": "none",
            "factory": _create_seasonal_naive,
        },
                "entso_e": {
            "model_family": "external_benchmark",
            "feature_mode": "proprietary",
            "weather_mode": "proprietary",
            "training_strategy": "none",
            "factory": _create_entso_e,
        },
        "xgboost": {
            "model_family": "tree",
            "feature_mode": "full",
            "weather_mode": "oracle",
            "training_strategy": "expanding",
            "factory": _create_xgboost,
        },
        "lightgbm": {
            "model_family": "tree",
            "feature_mode": "full",
            "weather_mode": "oracle",
            "training_strategy": "expanding",
            "factory": _create_lightgbm,
        },
        "lstm": {
            "model_family": "recurrent",
            "feature_mode": "sequence",
            "weather_mode": "oracle",
            "training_strategy": "fixed",
            "factory": _create_lstm,
        },
        "patchtst": {
            "model_family": "transformer",
            "feature_mode": "univariate",
            "weather_mode": "none",
            "training_strategy": "fixed",
            "factory": _create_patchtst,
        },
        "tide": {
            "model_family": "mlp_encoder_decoder",
            "feature_mode": "exogenous",
            "weather_mode": "oracle",
            "training_strategy": "fixed",
            "factory": _create_tide,
        },
    }

    if model_name not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Available: {available}"
        )

    return registry[model_name]


def get_available_models() -> list:
    """Return list of all registered model names."""

    return [
        "naive", "seasonal_naive", "entso_e",
        "xgboost", "lightgbm",
        "lstm", "patchtst", "tide",
    ]


# --- Factory functions (lazy imports to avoid loading all frameworks) ---

def _create_naive():
    from energy_forecasting.models.baselines import NaiveForecaster
    return NaiveForecaster(), False

def _create_seasonal_naive():
    from energy_forecasting.models.baselines import SeasonalNaiveForecaster
    return SeasonalNaiveForecaster(), False

def _create_xgboost():
    from energy_forecasting.models.forecaster import ForecastPipeline
    return ForecastPipeline(), True

def _create_lightgbm():
    from lightgbm import LGBMRegressor
    from energy_forecasting.models.forecaster import ForecastPipeline
    lgbm = LGBMRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    return ForecastPipeline(model=lgbm), True

def _create_lstm():
    from energy_forecasting.models.lstm_model import LSTMPipeline
    return LSTMPipeline(epochs=50, batch_size=64, learning_rate=0.001), False

def _create_patchtst():
    from energy_forecasting.models.transformer_model import TransformerPipeline
    return TransformerPipeline(max_steps=500, patch_len=24), False

def _create_tide():
    from energy_forecasting.models.tide_model import TiDEPipeline
    return TiDEPipeline(max_steps=500), False

def _create_entso_e():
    from energy_forecasting.models.entso_e_benchmark import EntsoEBenchmark
    return EntsoEBenchmark(), False