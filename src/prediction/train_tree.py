from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from baselines import historical_mean_predictions, persistence_predictions
from dataset import build_supervised_dataset, load_observations, test_frame, tree_matrices
from evaluate import write_predictions

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OBSERVATIONS = PROJECT_ROOT / "data" / "processed" / "traffic_observations.csv"
DEFAULT_PREDICTIONS = PROJECT_ROOT / "experiments" / "prediction" / "predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train tabular traffic queue-length prediction models."
    )
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--window-steps", type=int, default=5)
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[5, 10, 15],
        help="Forecast horizons in one-minute observation steps.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    observations = load_observations(args.observations)
    if args.predictions.exists():
        args.predictions.unlink()

    for horizon_steps in args.horizons:
        dataset = build_supervised_dataset(
            observations,
            window_steps=args.window_steps,
            horizon_steps=horizon_steps,
        )
        train_features, train_target, test_features, test_target, _ = tree_matrices(dataset)
        metadata = test_frame(dataset)
        metadata["horizon_minutes"] = horizon_steps

        baseline_models = {
            "persistence": persistence_predictions(dataset),
            "historical_mean": historical_mean_predictions(dataset),
        }
        for name, predictions in baseline_models.items():
            write_predictions(metadata, test_target, predictions, name, args.predictions)

        random_forest = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=args.random_state,
            n_jobs=-1,
        )
        random_forest.fit(train_features, train_target)
        write_predictions(
            metadata,
            test_target,
            random_forest.predict(test_features),
            "random_forest",
            args.predictions,
        )

        xgboost = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=args.random_state,
            n_jobs=1,
        )
        xgboost.fit(train_features, train_target)
        write_predictions(
            metadata,
            test_target,
            xgboost.predict(test_features),
            "xgboost",
            args.predictions,
        )
        print(
            f"horizon_minutes={horizon_steps} train_samples={len(train_target)} "
            f"test_samples={len(test_target)}"
        )
    print(f"predictions={args.predictions.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        sys.exit(f"Error: {error}")
