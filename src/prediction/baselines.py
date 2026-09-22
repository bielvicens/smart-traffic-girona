from __future__ import annotations

import numpy as np
import pandas as pd

from dataset import SupervisedDataset, feature_column_name


def persistence_predictions(dataset: SupervisedDataset) -> np.ndarray:
    test_rows = dataset.frame.loc[dataset.frame["split"].eq("test")]
    return test_rows[feature_column_name("queue_length", 0)].to_numpy(dtype=float)


def historical_mean_predictions(dataset: SupervisedDataset) -> np.ndarray:
    frame = dataset.frame
    train_rows = frame.loc[frame["split"].eq("train")]
    test_rows = frame.loc[frame["split"].eq("test")]
    means = train_rows.groupby("intersection_id")["target_queue_length"].mean()
    overall_mean = float(train_rows["target_queue_length"].mean())
    return (
        test_rows["intersection_id"].map(means).fillna(overall_mean).to_numpy(dtype=float)
    )
