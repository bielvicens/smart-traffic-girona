from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

OBSERVATION_FEATURES = (
    "queue_length",
    "vehicle_count",
    "mean_speed_mps",
    "throughput_vehicles",
    "tls_phase",
)
GROUP_COLUMNS = ("scenario", "seed", "intersection_id")
METADATA_COLUMNS = (
    "scenario",
    "seed",
    "intersection_id",
    "origin_time_seconds",
    "target_time_seconds",
    "split",
)


@dataclass(frozen=True)
class SupervisedDataset:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    sequence_features: tuple[str, ...]
    window_steps: int
    horizon_steps: int


def load_observations(path: Path) -> pd.DataFrame:
    observations = pd.read_csv(path)
    required_columns = {
        *GROUP_COLUMNS,
        "simulation_time_seconds",
        *OBSERVATION_FEATURES,
    }
    missing_columns = required_columns.difference(observations.columns)
    if missing_columns:
        raise ValueError(f"Observation file is missing columns: {sorted(missing_columns)}")
    return observations.sort_values([*GROUP_COLUMNS, "simulation_time_seconds"]).reset_index(
        drop=True
    )


def feature_column_name(feature: str, lag_steps: int) -> str:
    return f"{feature}_lag_{lag_steps}"


def build_supervised_dataset(
    observations: pd.DataFrame,
    window_steps: int = 5,
    horizon_steps: int = 5,
    test_fraction: float = 0.2,
) -> SupervisedDataset:
    if window_steps <= 0 or horizon_steps <= 0:
        raise ValueError("Window and horizon sizes must be positive.")
    if not 0 < test_fraction < 1:
        raise ValueError("Test fraction must be between zero and one.")

    feature_columns = tuple(
        feature_column_name(feature, lag)
        for lag in range(window_steps - 1, -1, -1)
        for feature in OBSERVATION_FEATURES
    )
    records: list[dict[str, float | int | str]] = []

    for group_values, group in observations.groupby(list(GROUP_COLUMNS), sort=False):
        group = group.sort_values("simulation_time_seconds").reset_index(drop=True)
        test_start_index = int(len(group) * (1 - test_fraction))
        first_target_index = window_steps - 1 + horizon_steps
        if first_target_index >= len(group) or test_start_index == 0:
            raise ValueError("Not enough observations to create the requested split.")

        scenario, seed, intersection_id = group_values
        for origin_index in range(window_steps - 1, len(group) - horizon_steps):
            target_index = origin_index + horizon_steps
            record: dict[str, float | int | str] = {
                "scenario": scenario,
                "seed": int(seed),
                "intersection_id": intersection_id,
                "origin_time_seconds": int(group.at[origin_index, "simulation_time_seconds"]),
                "target_time_seconds": int(group.at[target_index, "simulation_time_seconds"]),
                "split": "train" if target_index < test_start_index else "test",
                "target_queue_length": float(group.at[target_index, "queue_length"]),
            }
            window = group.iloc[origin_index - window_steps + 1 : origin_index + 1]
            for lag_index, (_, row) in enumerate(window.iterrows()):
                lag = window_steps - 1 - lag_index
                for feature in OBSERVATION_FEATURES:
                    record[feature_column_name(feature, lag)] = float(row[feature])
            records.append(record)

    frame = pd.DataFrame.from_records(records)
    if frame.empty or not {"train", "test"}.issubset(frame["split"].unique()):
        raise ValueError("The generated supervised dataset must contain train and test samples.")
    return SupervisedDataset(
        frame=frame,
        feature_columns=feature_columns,
        sequence_features=OBSERVATION_FEATURES,
        window_steps=window_steps,
        horizon_steps=horizon_steps,
    )


def tree_matrices(
    dataset: SupervisedDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    categorical = pd.get_dummies(
        dataset.frame[["scenario", "intersection_id"]], dtype=float
    )
    features = pd.concat(
        [dataset.frame.loc[:, dataset.feature_columns].astype(float), categorical], axis=1
    )
    train_mask = dataset.frame["split"].eq("train")
    test_mask = dataset.frame["split"].eq("test")
    target = dataset.frame["target_queue_length"].to_numpy(dtype=float)
    return (
        features.loc[train_mask].to_numpy(dtype=float),
        target[train_mask.to_numpy()],
        features.loc[test_mask].to_numpy(dtype=float),
        target[test_mask.to_numpy()],
        tuple(features.columns),
    )


def lstm_matrices(
    dataset: SupervisedDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = dataset.frame.loc[:, dataset.feature_columns].to_numpy(dtype=float)
    sequences = values.reshape(
        len(dataset.frame), dataset.window_steps, len(dataset.sequence_features)
    )
    train_mask = dataset.frame["split"].eq("train").to_numpy()
    test_mask = dataset.frame["split"].eq("test").to_numpy()

    feature_scaler = StandardScaler()
    train_shape = sequences[train_mask].shape
    feature_scaler.fit(sequences[train_mask].reshape(-1, train_shape[-1]))
    scaled_sequences = feature_scaler.transform(
        sequences.reshape(-1, train_shape[-1])
    ).reshape(sequences.shape)

    static_features = pd.get_dummies(
        dataset.frame[["scenario", "intersection_id"]], dtype=float
    ).to_numpy(dtype=float)
    static_sequences = np.repeat(
        static_features[:, np.newaxis, :], dataset.window_steps, axis=1
    )
    inputs = np.concatenate([scaled_sequences, static_sequences], axis=2)
    targets = dataset.frame["target_queue_length"].to_numpy(dtype=np.float32)
    return (
        inputs[train_mask].astype(np.float32),
        targets[train_mask],
        inputs[test_mask].astype(np.float32),
        targets[test_mask],
    )


def test_frame(dataset: SupervisedDataset) -> pd.DataFrame:
    return dataset.frame.loc[dataset.frame["split"].eq("test"), list(METADATA_COLUMNS)].reset_index(
        drop=True
    )
