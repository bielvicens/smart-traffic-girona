# Traffic prediction experiments

This experiment predicts the **per-intersection average queue length** from the preceding five one-minute observations. The forecast horizons are 5, 10, and 15 minutes.

## Dataset

- 25 SUMO runs: five demand scenarios × five route-demand seeds (`42`–`46`)
- 14 traffic-light controllers per run
- 60 one-minute observations per controller and run
- 21,000 observations in `data/processed/traffic_observations.csv`
- Input features: queue length, vehicle count, mean speed, throughput, and current traffic-light phase
- Target: queue length at the requested future horizon

The simulation seed is fixed at `42` so that variation is attributable to the generated route demand. Each scenario/seed/intersection series is split temporally: the first 80% of target times trains models, while the final 20% tests them. This prevents targets from the test period entering the training set. The route demand is synthetic and not calibrated with observed counts.

## Models

- **Persistence**: latest observed queue length
- **Historical mean**: mean training target for each intersection
- **Random Forest**: 300 trees using lagged traffic features plus scenario and intersection identifiers
- **XGBoost**: gradient-boosted trees using the same features
- **LSTM**: one-layer, hidden size 64, trained for 50 epochs on the five-step feature sequence

## Reproduce

```powershell
.\.venv\Scripts\python.exe src\prediction\data_collector.py --seeds 5 --base-seed 42
.\.venv\Scripts\python.exe src\prediction\train_tree.py
.\.venv\Scripts\python.exe src\prediction\train_lstm.py
.\.venv\Scripts\python.exe src\prediction\evaluate.py
```

## Results

| Horizon | Model | MAE | RMSE | R² |
|---:|---|---:|---:|---:|
| 5 min | LSTM | 0.2790 | 0.4943 | 0.5799 |
| 5 min | XGBoost | 0.2827 | **0.4900** | **0.5872** |
| 5 min | Random Forest | 0.2866 | 0.4971 | 0.5751 |
| 5 min | Historical mean | 0.3929 | 0.6649 | 0.2398 |
| 5 min | Persistence | 0.4126 | 0.7664 | -0.0101 |
| 10 min | Random Forest | **0.3138** | **0.5411** | **0.4965** |
| 10 min | XGBoost | 0.3171 | 0.5502 | 0.4795 |
| 10 min | LSTM | 0.3215 | 0.5764 | 0.4287 |
| 10 min | Historical mean | 0.3981 | 0.6680 | 0.2328 |
| 10 min | Persistence | 0.4449 | 0.8194 | -0.1544 |
| 15 min | LSTM | **0.2791** | **0.4953** | **0.5782** |
| 15 min | XGBoost | 0.3021 | 0.5213 | 0.5328 |
| 15 min | Random Forest | 0.3105 | 0.5346 | 0.5087 |
| 15 min | Historical mean | 0.4067 | 0.6741 | 0.2186 |
| 15 min | Persistence | 0.4166 | 0.8074 | -0.1208 |

The learned models outperform both simple baselines at every horizon. No model is presumed superior: XGBoost has the best 5-minute RMSE/R², Random Forest is best at 10 minutes, and the LSTM is best at 15 minutes in this single experimental configuration. These findings require validation across more seeds and, ideally, a held-out traffic distribution before general conclusions.

`predictions.csv`, `metrics.csv`, `model_comparison.png`, `run_metadata.json`, and trained LSTM checkpoints are versioned experiment outputs for this reproducible run.
