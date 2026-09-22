from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import build_supervised_dataset, load_observations, lstm_matrices, test_frame
from evaluate import write_predictions

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OBSERVATIONS = PROJECT_ROOT / "data" / "processed" / "traffic_observations.csv"
DEFAULT_PREDICTIONS = PROJECT_ROOT / "experiments" / "prediction" / "predictions.csv"
DEFAULT_MODEL = PROJECT_ROOT / "experiments" / "prediction" / "lstm.pt"


class QueueLengthLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(inputs)
        return self.output(hidden[-1]).squeeze(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an LSTM queue-length predictor.")
    parser.add_argument("--observations", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--window-steps", type=int, default=5)
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[5, 10, 15],
        help="Forecast horizons in one-minute observation steps.",
    )
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    observations = load_observations(args.observations)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)

    for horizon_steps in args.horizons:
        set_seed(args.seed)
        dataset = build_supervised_dataset(
            observations,
            window_steps=args.window_steps,
            horizon_steps=horizon_steps,
        )
        train_inputs, train_targets, test_inputs, test_targets = lstm_matrices(dataset)
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(train_inputs), torch.from_numpy(train_targets)),
            batch_size=args.batch_size,
            shuffle=True,
        )

        model = QueueLengthLSTM(train_inputs.shape[2], args.hidden_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        loss_function = nn.MSELoss()
        model.train()
        for _ in range(args.epochs):
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                loss = loss_function(model(inputs), targets)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            predictions = model(torch.from_numpy(test_inputs)).numpy()

        metadata = test_frame(dataset)
        metadata["horizon_minutes"] = horizon_steps
        write_predictions(metadata, test_targets, predictions, "lstm", args.predictions)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "input_size": train_inputs.shape[2],
                "hidden_size": args.hidden_size,
                "window_steps": args.window_steps,
                "horizon_steps": horizon_steps,
            },
            args.model_output.with_stem(f"lstm_{horizon_steps}m"),
        )
        print(f"horizon_minutes={horizon_steps} test_samples={len(test_targets)}")
    print(f"predictions={args.predictions.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        sys.exit(f"Error: {error}")
