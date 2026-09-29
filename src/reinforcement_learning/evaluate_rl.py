from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RL_SOURCE = PROJECT_ROOT / "src" / "reinforcement_learning"
sys.path.insert(0, str(RL_SOURCE))

from actuated_baseline import generate_actuated_programs
from metrics import run_controller
from sumo_env import FlattenedActionEnv, GironaTrafficEnv, load_config

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "rl" / "default.json"
DEFAULT_MODEL_DIRECTORY = PROJECT_ROOT / "models" / "reinforcement_learning"
DEFAULT_RESULTS_FILE = PROJECT_ROOT / "experiments" / "rl" / "controller_metrics.csv"
DEFAULT_PLOT_FILE = PROJECT_ROOT / "experiments" / "rl" / "controller_comparison.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate traffic-light controllers across scenarios.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--controllers",
        nargs="+",
        choices=["fixed_time", "actuated", "ppo", "dqn"],
        default=["fixed_time", "actuated"],
    )
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--model-directory", type=Path, default=DEFAULT_MODEL_DIRECTORY)
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS_FILE)
    parser.add_argument("--plot-file", type=Path, default=DEFAULT_PLOT_FILE)
    return parser.parse_args()


def load_rl_model(
    algorithm: str,
    config_path: Path,
    model_directory: Path,
    scenario: str,
) -> tuple[Any, Any]:
    def factory() -> GironaTrafficEnv | FlattenedActionEnv:
        environment = GironaTrafficEnv(
            config_path=config_path,
            scenario_name=scenario,
            route_seeds=[42],
        )
        return FlattenedActionEnv(environment) if algorithm == "dqn" else environment

    environment = DummyVecEnv([factory])
    model_path = model_directory / f"{algorithm}_centralized"
    normalized_environment = VecNormalize.load(
        str(model_path.with_suffix(".vecnormalize.pkl")), environment
    )
    normalized_environment.training = False
    normalized_environment.norm_reward = False
    model_class = PPO if algorithm == "ppo" else DQN
    return model_class.load(model_path, env=normalized_environment), normalized_environment


def rl_callback(model: Any, normalized_environment: Any) -> Any:
    observation: np.ndarray | None = None

    def callback(connection: Any) -> None:
        nonlocal observation
        raw_environment = normalized_environment.venv.envs[0].unwrapped
        if raw_environment._traci is None:
            raw_environment.attach_connection(connection)
        raw_observation = raw_environment._observation()
        observation = normalized_environment.normalize_obs(raw_observation[np.newaxis, :])
        action, _ = model.predict(observation, deterministic=True)
        raw_action = action[0]
        if hasattr(normalized_environment.venv.envs[0], "action"):
            raw_action = normalized_environment.venv.envs[0].action(raw_action)
        raw_environment._apply_actions(np.asarray(raw_action, dtype=np.int64))

    return callback


def evaluate_rl_controller(
    algorithm: str,
    scenario: str,
    seeds: list[int],
    config_path: Path,
    model_directory: Path,
) -> list[dict[str, float | int | str]]:
    model, normalized_environment = load_rl_model(
        algorithm, config_path, model_directory, scenario
    )
    config = load_config(config_path)
    callback = rl_callback(model, normalized_environment)
    results = []
    for seed in seeds:
        metrics = run_controller(
            controller=algorithm,
            scenario_name=scenario,
            route_seed=seed,
            action_callback=callback,
            decision_interval_seconds=config["decision_interval_seconds"],
        ).as_dict()
        results.append(metrics)
    normalized_environment.close()
    return results


def append_rows(rows: list[dict[str, float | int | str]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_file.exists()
    with output_file.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def plot_results(results_file: Path, output_file: Path) -> None:
    import pandas as pd

    results = pd.read_csv(results_file)
    grouped = results.groupby("controller", as_index=False)[
        ["average_waiting_time_seconds", "average_queue_length"]
    ].mean()
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(grouped["controller"], grouped["average_waiting_time_seconds"])
    axes[0].set_ylabel("Average waiting time (s)")
    axes[1].bar(grouped["controller"], grouped["average_queue_length"])
    axes[1].set_ylabel("Average queue length")
    figure.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    scenarios = args.scenarios or [config["scenario"]]
    seeds = args.seeds or config["evaluation_seeds"]
    if args.results_file.exists():
        args.results_file.unlink()
    all_rows = []
    actuated_file = generate_actuated_programs()

    for controller in args.controllers:
        for scenario in scenarios:
            if controller == "fixed_time":
                rows = [
                    run_controller("fixed_time", scenario, seed).as_dict() for seed in seeds
                ]
            elif controller == "actuated":
                rows = [
                    run_controller(
                        "actuated", scenario, seed, additional_files=[actuated_file]
                    ).as_dict()
                    for seed in seeds
                ]
            else:
                rows = evaluate_rl_controller(
                    controller, scenario, seeds, args.config, args.model_directory
                )
            all_rows.extend(rows)
            append_rows(rows, args.results_file)
            print(f"evaluated controller={controller} scenario={scenario} runs={len(rows)}")

    plot_results(args.results_file, args.plot_file)
    print(f"results={args.results_file.resolve()}")
    print(f"plot={args.plot_file.resolve()}")


if __name__ == "__main__":
    main()
