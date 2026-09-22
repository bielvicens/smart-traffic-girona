from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "prediction"))

from baselines import historical_mean_predictions, persistence_predictions
from dataset import build_supervised_dataset


class PredictionDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        records = []
        for seed in (1, 2):
            for time in range(60, 660, 60):
                records.append(
                    {
                        "scenario": "normal",
                        "seed": seed,
                        "simulation_time_seconds": time,
                        "intersection_id": "tls_a",
                        "queue_length": float(time / 60 + seed),
                        "vehicle_count": float(seed),
                        "mean_speed_mps": 10.0,
                        "throughput_vehicles": seed,
                        "tls_phase": 0,
                    }
                )
        self.observations = pd.DataFrame(records)

    def test_targets_are_future_queue_lengths(self) -> None:
        dataset = build_supervised_dataset(
            self.observations, window_steps=2, horizon_steps=2, test_fraction=0.3
        )
        first_row = dataset.frame.iloc[0]
        self.assertEqual(first_row["origin_time_seconds"], 120)
        self.assertEqual(first_row["target_time_seconds"], 240)
        self.assertEqual(first_row["target_queue_length"], 5.0)

    def test_temporal_split_keeps_test_targets_later(self) -> None:
        dataset = build_supervised_dataset(
            self.observations, window_steps=2, horizon_steps=2, test_fraction=0.3
        )
        for _, group in dataset.frame.groupby(["scenario", "seed", "intersection_id"]):
            train = group.loc[group["split"].eq("train")]
            test = group.loc[group["split"].eq("test")]
            self.assertLess(train["target_time_seconds"].max(), test["target_time_seconds"].min())

    def test_persistence_uses_latest_observed_queue(self) -> None:
        dataset = build_supervised_dataset(
            self.observations, window_steps=2, horizon_steps=2, test_fraction=0.3
        )
        test_rows = dataset.frame.loc[dataset.frame["split"].eq("test")]
        self.assertEqual(
            persistence_predictions(dataset).tolist(),
            test_rows["queue_length_lag_0"].tolist(),
        )
        self.assertEqual(len(historical_mean_predictions(dataset)), len(test_rows))


if __name__ == "__main__":
    unittest.main()
