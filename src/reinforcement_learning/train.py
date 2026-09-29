from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RL_SOURCE = PROJECT_ROOT / "src" / "reinforcement_learning"
sys.path.insert(0, str(RL_SOURCE))

from sumo_env import FlattenedActionEnv, GironaTrafficEnv, load_config

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "rl" / "default.json"
DEFAULT_MODEL_DIRECTORY = PROJECT_ROOT / "models" / "reinforcement_learning"
DEFAULT_LOG_DIRECTORY = PROJECT_ROOT / "experiments" / "rl" / "tensorboard"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train centralized traffic-light RL agents.")
    parser.add_argument("--algorithm", choices=["ppo", "dqn"], required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-directory", type=Path, default=DEFAULT_MODEL_DIRECTORY)
    parser.add_argument("--log-directory", type=Path, default=DEFAULT_LOG_DIRECTORY)
    parser.add_argument("--checkpoint-frequency", type=int, default=50000)
    return parser.parse_args()


def make_environment(
    config_path: Path, scenario_name: str | None, algorithm: str
) -> DummyVecEnv:
    config = load_config(config_path)
    seeds = config["training_seeds"]

    def factory() -> Monitor:
        environment = GironaTrafficEnv(
            config_path=config_path,
            scenario_name=scenario_name,
            route_seeds=seeds,
        )
        if algorithm == "dqn":
            environment = FlattenedActionEnv(environment)
        return Monitor(environment)

    return DummyVecEnv([factory])


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    algorithm_config = config[args.algorithm]
    total_timesteps = args.timesteps or algorithm_config["total_timesteps"]
    device = args.device
    if device == "auto" and not torch.cuda.is_available():
        device = "cpu"

    environment = make_environment(args.config, args.scenario, args.algorithm)
    normalized_environment = VecNormalize(
        environment,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )
    args.model_directory.mkdir(parents=True, exist_ok=True)
    args.log_directory.mkdir(parents=True, exist_ok=True)
    checkpoint = CheckpointCallback(
        save_freq=args.checkpoint_frequency,
        save_path=str(args.model_directory / "checkpoints" / args.algorithm),
        name_prefix=args.algorithm,
        save_vecnormalize=True,
    )

    if args.algorithm == "ppo":
        model = PPO(
            "MlpPolicy",
            normalized_environment,
            learning_rate=algorithm_config["learning_rate"],
            n_steps=algorithm_config["n_steps"],
            batch_size=algorithm_config["batch_size"],
            gamma=algorithm_config["gamma"],
            tensorboard_log=str(args.log_directory),
            verbose=1,
            device=device,
            seed=42,
        )
    else:
        model = DQN(
            "MlpPolicy",
            normalized_environment,
            learning_rate=algorithm_config["learning_rate"],
            buffer_size=algorithm_config["buffer_size"],
            learning_starts=algorithm_config["learning_starts"],
            batch_size=algorithm_config["batch_size"],
            gamma=algorithm_config["gamma"],
            tensorboard_log=str(args.log_directory),
            verbose=1,
            device=device,
            seed=42,
        )

    model.learn(total_timesteps=total_timesteps, callback=checkpoint)
    model_path = args.model_directory / f"{args.algorithm}_centralized"
    model.save(model_path)
    normalized_environment.save(str(model_path.with_suffix(".vecnormalize.pkl")))
    metadata = {
        "algorithm": args.algorithm,
        "scenario": args.scenario or config["scenario"],
        "total_timesteps": total_timesteps,
        "device": str(device),
        "training_seeds": config["training_seeds"],
        "decision_interval_seconds": config["decision_interval_seconds"],
    }
    model_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    normalized_environment.close()
    print(f"model={model_path.with_suffix('.zip')}")
    print(f"normalization={model_path.with_suffix('.vecnormalize.pkl')}")


if __name__ == "__main__":
    main()
