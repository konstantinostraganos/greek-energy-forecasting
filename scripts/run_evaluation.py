"""Run model evaluation with MLflow tracking.

Usage:
    python scripts/run_evaluation.py --model xgboost
    python scripts/run_evaluation.py --model xgboost lightgbm
    python scripts/run_evaluation.py --model all
    python scripts/run_evaluation.py --list
"""

import argparse
import sys

from energy_forecasting.evaluation.evaluate import run_evaluation
from energy_forecasting.evaluation.model_registry import get_available_models


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate forecasting models with MLflow tracking"
    )
    parser.add_argument(
        "--model",
        nargs="+",
        help="Model name(s) to evaluate, or 'all'",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit",
    )

    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for name in get_available_models():
            print(f"  - {name}")
        return

    if not args.model:
        parser.print_help()
        sys.exit(1)

    # Resolve 'all' to full model list
    if args.model == ["all"]:
        models = get_available_models()
    else:
        models = args.model

    # Run each model
    results = []
    for model_name in models:
        try:
            result = run_evaluation(model_name)
            results.append(result)
        except Exception as e:
            print(f"\nERROR evaluating {model_name}: {e}\n")
            continue

    # Summary table if multiple models
    if len(results) > 1:
        print(f"\n{'=' * 60}")
        print(f"  SUMMARY — All Models")
        print(f"{'=' * 60}")
        print(f"  {'Model':<20} {'MAPE':>8} {'MAE':>10} {'RMSE':>10}")
        print(f"  {'-' * 50}")
        for r in results:
            m = r["metrics"]
            print(
                f"  {r['model_name']:<20} "
                f"{m['mape']:>7.2f}% "
                f"{m['mae']:>9.0f} "
                f"{m['rmse']:>9.0f}"
            )
        print(f"\n  View in MLflow: mlflow ui")
        print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()