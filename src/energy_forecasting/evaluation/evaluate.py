"""Unified model evaluation with MLflow tracking.

Runs a model through the standard backtesting protocol
and logs everything to MLflow: parameters, metrics,
tags, and result artifacts.

Every evaluation uses the same cutoffs, the same metrics,
and the same result schema — making results comparable
across all models.

Training strategies:
    expanding: retrain from scratch at every cutoff using
               only data available up to that point.
               Methodologically correct but slow.
    fixed:     train once on pre-eval data, reuse for all
               cutoffs. Fast but model never sees new data.
    none:      baselines that need no training.

Cutoff convention: hour 10 UTC (12:00-13:00 Greek local).
All timestamps in the dataset are UTC.
"""

import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
from loguru import logger

from energy_forecasting.config import settings
from energy_forecasting.evaluation.metrics import compute_metrics
from energy_forecasting.evaluation.model_registry import (
    get_model_config,
    get_available_models,
)


# Evaluation constants
EVAL_DAYS = 90
CUTOFF_HOUR_UTC = 10
HORIZON = 24
EXPERIMENT_NAME = "greek-energy-forecasting"


def run_evaluation(model_name: str) -> dict:
    """Run full evaluation pipeline for a single model.

    Steps:
        1. Load feature-engineered dataset
        2. Instantiate model from registry
        3. Add lag features if needed
        4. Find cutoff indices
        5. Run forecasts (strategy-dependent)
        6. Compute metrics
        7. Log everything to MLflow

    Args:
        model_name: Key from model_registry (e.g. 'xgboost')

    Returns:
        Dictionary with metrics and run metadata.
    """

    # --- 1. Load data ---
    logger.info(f"=== Evaluating: {model_name} ===")
    data_path = settings.data.features_dir / "dataset.parquet"
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} rows from {data_path}")

    # --- 2. Get model config ---
    config = get_model_config(model_name)
    strategy = config["training_strategy"]

    # --- 3. Initial model + lag features ---
    pipeline, needs_lags = config["factory"]()
    if needs_lags:
        df = pipeline.add_lag_features(df)

    # --- 4. Define eval period and find cutoffs ---
    eval_start = len(df) - (EVAL_DAYS * 24)
    train_end_idx = eval_start - 1

    eval_start_date = str(df.iloc[eval_start]["timestamp"])
    eval_end_date = str(df.iloc[-1]["timestamp"])
    train_start_date = str(df.iloc[0]["timestamp"])
    train_end_date = str(df.iloc[train_end_idx]["timestamp"])

    cutoff_indices = []
    for i in range(eval_start, len(df) - HORIZON):
        ts = df.iloc[i]["timestamp"]
        if hasattr(ts, "hour"):
            hour = ts.hour
        else:
            hour = pd.Timestamp(ts).hour
        if hour == CUTOFF_HOUR_UTC:
            cutoff_indices.append(i)

    logger.info(f"Found {len(cutoff_indices)} daily cutoffs")

    if not cutoff_indices:
        raise ValueError("No valid cutoffs found in evaluation period")

    # --- 5. Run forecasts based on training strategy ---
    all_results = []
    total_train_time = 0.0

    if strategy == "expanding":
        all_results, total_train_time = _run_expanding(
            config, df, cutoff_indices, needs_lags, model_name,
        )

    elif strategy == "fixed":
        all_results, total_train_time = _run_fixed(
            pipeline, df, train_end_idx, cutoff_indices, model_name,
        )

    elif strategy == "none":
        all_results, total_train_time = _run_no_training(
            pipeline, df, cutoff_indices, model_name,
        )

    if not all_results:
        raise ValueError(f"{model_name}: no valid forecasts produced")

    results_df = pd.concat(all_results, ignore_index=True)
    logger.info(
        f"Generated {len(results_df)} forecast observations "
        f"from {len(all_results)} cutoffs"
    )

    # --- 6. Compute metrics ---
    metrics = compute_metrics(results_df)
    metrics["n_cutoffs"] = len(all_results)
    metrics["train_duration_seconds"] = round(total_train_time, 1)

    # --- 7. Log to MLflow ---
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=model_name):

        # Parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_family", config["model_family"])
        mlflow.log_param("feature_mode", config["feature_mode"])
        mlflow.log_param("weather_mode", config["weather_mode"])
        mlflow.log_param("training_strategy", strategy)
        mlflow.log_param("horizon", HORIZON)
        mlflow.log_param("cutoff_hour_utc", CUTOFF_HOUR_UTC)
        mlflow.log_param("eval_days", EVAL_DAYS)
        mlflow.log_param("n_cutoffs", len(all_results))
        mlflow.log_param("train_start", train_start_date)
        mlflow.log_param("eval_start", eval_start_date)
        mlflow.log_param("eval_end", eval_end_date)

        # Model-specific hyperparameters
        _log_model_params(pipeline, config)

        # Metrics
        mlflow.log_metric("mape", metrics["mape"])
        mlflow.log_metric("mae", metrics["mae"])
        mlflow.log_metric("rmse", metrics["rmse"])
        mlflow.log_metric("n_observations", metrics["n_observations"])
        mlflow.log_metric("train_duration_seconds", total_train_time)

        # Tags
        mlflow.set_tag("model", model_name)
        mlflow.set_tag("model_family", config["model_family"])
        mlflow.set_tag("weather_mode", config["weather_mode"])
        mlflow.set_tag("training_strategy", strategy)
        mlflow.set_tag("evaluation_type", "backtest")
        mlflow.set_tag("status", "completed")
        mlflow.set_tag("dataset_version", str(data_path))

        # Artifacts
        artifacts_dir = Path("mlflow_artifacts_tmp")
        artifacts_dir.mkdir(exist_ok=True)

        results_path = artifacts_dir / f"{model_name}_predictions.parquet"
        results_df.to_parquet(results_path, index=False)
        mlflow.log_artifact(str(results_path))

        metrics_path = artifacts_dir / f"{model_name}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact(str(metrics_path))

        run_id = mlflow.active_run().info.run_id

    # --- 8. Print results ---
    print(f"\n{'=' * 55}")
    print(f"  {model_name.upper()} — Evaluation Results")
    print(f"{'=' * 55}")
    print(f"  MAPE:       {metrics['mape']:.2f}%")
    print(f"  MAE:        {metrics['mae']:.0f} MW")
    print(f"  RMSE:       {metrics['rmse']:.0f} MW")
    print(f"  Cutoffs:    {metrics['n_cutoffs']}")
    print(f"  Obs:        {metrics['n_observations']}")
    print(f"  Train time: {total_train_time:.1f}s")
    print(f"  Weather:    {config['weather_mode']}")
    print(f"  Strategy:   {strategy}")
    print(f"  MLflow run: {run_id}")
    print(f"{'=' * 55}\n")

    return {
        "model_name": model_name,
        "metrics": metrics,
        "run_id": run_id,
        "config": {k: v for k, v in config.items() if k != "factory"},
    }


def _run_expanding(config, df, cutoff_indices, needs_lags, model_name):
    """Expanding-window: retrain from scratch at every cutoff.

    At each cutoff, a fresh model is created and trained
    on all data up to that cutoff. This ensures the model
    never sees future data, at the cost of N retrainings.
    """

    all_results = []
    total_train_time = 0.0

    for i, cutoff_idx in enumerate(cutoff_indices):
        # Fresh model for each cutoff
        pipeline, _ = config["factory"]()

        # Add lag features if needed (reuse already-computed df)
        # Lag features are safe because min(LOAD_LAGS) >= HORIZON

        # Train on everything up to this cutoff
        t0 = time.time()
        pipeline.train(df, train_end_idx=cutoff_idx)
        total_train_time += time.time() - t0

        # Predict next 24 hours
        result = _make_forecast(pipeline, df, cutoff_idx, model_name)
        if result is not None:
            all_results.append(result)

        if (i + 1) % 10 == 0:
            logger.info(
                f"  Expanding: {i + 1}/{len(cutoff_indices)} cutoffs done"
            )

    return all_results, total_train_time


def _run_fixed(pipeline, df, train_end_idx, cutoff_indices, model_name):
    """Fixed: train once before eval, reuse for all cutoffs.

    The model is trained on all data before the evaluation
    period and used as-is for every cutoff. Faster but the
    model never incorporates new observations.
    """

    logger.info(f"Training once on rows 0-{train_end_idx}")

    t0 = time.time()
    pipeline.train(df, train_end_idx=train_end_idx)
    total_train_time = time.time() - t0

    logger.info(f"Training completed in {total_train_time:.1f}s")

    all_results = []
    for i, cutoff_idx in enumerate(cutoff_indices):
        result = _make_forecast(pipeline, df, cutoff_idx, model_name)
        if result is not None:
            all_results.append(result)

        if (i + 1) % 10 == 0:
            logger.info(
                f"  Fixed: {i + 1}/{len(cutoff_indices)} cutoffs done"
            )

    return all_results, total_train_time


def _run_no_training(pipeline, df, cutoff_indices, model_name):
    """Baselines: no training needed, just predict."""

    all_results = []
    for cutoff_idx in cutoff_indices:
        result = _make_forecast(pipeline, df, cutoff_idx, model_name)
        if result is not None:
            all_results.append(result)

    return all_results, 0.0


def _make_forecast(pipeline, df, cutoff_idx, model_name):
    """Generate a single 24h forecast and add metadata columns.

    Returns None if the forecast fails or produces no
    valid observations.
    """

    try:
        result = pipeline.predict(df, cutoff_idx)
        result = result.dropna(
            subset=["predicted_load_mw", "actual_load_mw"]
        )

        if result.empty:
            return None

        cutoff_ts = df.iloc[cutoff_idx]["timestamp"]
        result = result.copy()
        result["cutoff_timestamp"] = cutoff_ts
        result["model_name"] = model_name
        result["horizon_step"] = range(1, len(result) + 1)

        return result

    except Exception as e:
        logger.warning(f"Cutoff {cutoff_idx} failed: {e}")
        return None


def _log_model_params(pipeline, config: dict) -> None:
    """Log model-specific hyperparameters to MLflow."""

    family = config["model_family"]

    if family == "tree":
        model = pipeline.model
        mlflow.log_param("n_estimators", model.n_estimators)
        mlflow.log_param("max_depth", model.max_depth)
        mlflow.log_param("learning_rate", model.learning_rate)
        mlflow.log_param("random_seed", model.random_state)

    elif family == "recurrent":
        mlflow.log_param("epochs", pipeline.epochs)
        mlflow.log_param("batch_size", pipeline.batch_size)
        mlflow.log_param("learning_rate", pipeline.learning_rate)
        mlflow.log_param("seq_len", pipeline.SEQ_LEN)

    elif family == "transformer":
        mlflow.log_param("max_steps", pipeline.max_steps)
        mlflow.log_param("learning_rate", pipeline.learning_rate)
        mlflow.log_param("patch_len", pipeline.patch_len)
        mlflow.log_param("random_seed", 42)

    elif family == "mlp_encoder_decoder":
        mlflow.log_param("max_steps", pipeline.max_steps)
        mlflow.log_param("learning_rate", pipeline.learning_rate)
        mlflow.log_param("random_seed", 42)