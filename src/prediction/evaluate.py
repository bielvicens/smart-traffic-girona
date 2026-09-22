from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDICTIONS = PROJECT_ROOT / "experiments" / "prediction" / "predictions.csv"
DEFAULT_METRICS = PROJECT_ROOT / "experiments" / "prediction" / "metrics.csv"
DEFAULT_PLOT = PROJECT_ROOT / "experiments" / "prediction" / "model_comparison.png"
DEFAULT_METADATA = PROJECT_ROOT / "experiments" / "prediction" / "run_metadata.json"


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
    }


def write_predictions(
    metadata: pd.DataFrame,
    actual: np.ndarray,
    predicted: np.ndarray,
    model: str,
    output_file: Path,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output = metadata.copy()
    output["model"] = model
    output["actual_queue_length"] = actual
    output["predicted_queue_length"] = predicted
    if output_file.exists():
        output.to_csv(output_file, mode="a", index=False, header=False)
    else:
        output.to_csv(output_file, index=False)


def collect_metrics(predictions_file: Path) -> pd.DataFrame:
    predictions = pd.read_csv(predictions_file)
    rows = []
    for (horizon_minutes, model), model_predictions in predictions.groupby(
        ["horizon_minutes", "model"]
    ):
        metrics = regression_metrics(
            model_predictions["actual_queue_length"].to_numpy(),
            model_predictions["predicted_queue_length"].to_numpy(),
        )
        rows.append({"horizon_minutes": horizon_minutes, "model": model, **metrics})
    return pd.DataFrame(rows).sort_values(["horizon_minutes", "mae"]).reset_index(
        drop=True
    )


def plot_metrics(metrics: pd.DataFrame, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for model, model_metrics in metrics.groupby("model"):
        axes[0].plot(
            model_metrics["horizon_minutes"], model_metrics["mae"], marker="o", label=model
        )
        axes[1].plot(
            model_metrics["horizon_minutes"], model_metrics["rmse"], marker="o", label=model
        )
    axes[0].set_xlabel("Forecast horizon (minutes)")
    axes[0].set_ylabel("MAE: queue length")
    axes[1].set_xlabel("Forecast horizon (minutes)")
    axes[1].set_ylabel("RMSE: queue length")
    axes[1].legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(output_file, dpi=150)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate prediction model outputs.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = collect_metrics(args.predictions)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.metrics, index=False)
    plot_metrics(metrics, args.plot)
    metadata = {
        "target": "queue_length",
        "metrics": ["mae", "rmse", "r2"],
        "models": sorted(metrics["model"].unique()),
        "horizons_minutes": sorted(metrics["horizon_minutes"].unique().tolist()),
    }
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False, float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
