# Reinforcement learning and controller comparison

This module implements a centralized traffic-light control environment on the Girona digital twin. It supports **PPO** (`MultiDiscrete`), **DQN** (flattened `Discrete`), **fixed-time**, and **actuated** controllers.

## Environment design

- **Observation space**: continuous vector of 70 features (five per traffic light):
  - Queue length (halting vehicles)
  - Vehicle count
  - Mean speed
  - Current traffic-light phase index
  - Elapsed seconds in the current phase
- **Action space**:
  - PPO: `MultiDiscrete([1, 1, 1, 3, 2, 2, 2, 2, 2, 2, 1, 2, 2, 3])` across 14 traffic lights (26 green phases)
  - DQN: flattened `Discrete(2304)`
- **Reward**: negative network queue length (`-sum(queue_length)`) at each decision step.
- **Decision interval**: 10 seconds.
- **Safety**: automatic yellow and all-red transition phases with minimum green constraints (10s).

## Reproduce locally

### 1. Actuated baseline

```powershell
.\.venv\Scripts\python.exe src\reinforcement_learning\actuated_baseline.py --scenario normal --seeds 42 43 44 45 46
```

### 2. Local smoke training (CPU)

```powershell
.\.venv\Scripts\python.exe src\reinforcement_learning\train.py --algorithm ppo --timesteps 1024
.\.venv\Scripts\python.exe src\reinforcement_learning\train.py --algorithm dqn --timesteps 1024
```

### 3. Evaluation

The full evaluation is compute-intensive and should run in Kaggle:

```powershell
.\.venv\Scripts\python.exe src\reinforcement_learning\evaluate_rl.py `
  --controllers fixed_time actuated ppo dqn `
  --scenarios low normal peak variable incident `
  --seeds 47 48 49 50 51
```

For resumable local debugging, evaluate a missing controller with `--append`; this preserves existing rows:

```powershell
.\.venv\Scripts\python.exe src\reinforcement_learning\evaluate_rl.py `
  --controllers ppo `
  --scenarios incident `
  --seeds 47 48 49 50 51 `
  --append
```

## Running full training and evaluation on Kaggle

Full training and the 100-run controller evaluation use `notebooks/kaggle_rl_training.ipynb`:
1. Push the repo to GitHub or upload it as a Kaggle dataset.
2. Enable GPU acceleration (T4 x2 or P100) in the Kaggle notebook settings.
3. Run the notebook to install SUMO, train PPO and DQN for 100,000 timesteps, and evaluate the four controllers on held-out seeds (`47`–`51`).
4. Download `experiments/rl/controller_metrics.csv` for local post-processing.

The final 100-run metrics and derived figures/tables are versioned under `experiments/rl/` and `experiments/evaluation/`. See [`report/README.md`](../../report/README.md) for the research analysis.
