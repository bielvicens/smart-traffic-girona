from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "experiments" / "rl" / "controller_metrics.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "evaluation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze and generate research tables/plots from controller evaluation metrics."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to controller_metrics.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where analysis tables and figures will be saved.",
    )
    return parser.parse_args()


def generate_summary_table(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    metrics = [
        "average_waiting_time_seconds",
        "average_travel_time_seconds",
        "average_queue_length",
        "average_stops",
        "throughput_vehicles_per_hour",
    ]
    summary = (
        df.groupby(["scenario", "controller"])[metrics]
        .agg(["mean", "std"])
        .round(2)
    )
    summary.to_csv(output_dir / "summary_table.csv")
    return summary


def generate_improvement_table(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    mean_df = (
        df.groupby(["scenario", "controller"])[
            ["average_waiting_time_seconds", "average_travel_time_seconds", "average_queue_length"]
        ]
        .mean()
        .reset_index()
    )
    scenarios = mean_df["scenario"].unique()
    controllers = [c for c in mean_df["controller"].unique() if c != "fixed_time"]
    records = []
    for sc in scenarios:
        sc_data = mean_df[mean_df["scenario"] == sc]
        ft_row = sc_data[sc_data["controller"] == "fixed_time"]
        if ft_row.empty:
            continue
        ft_wait = ft_row["average_waiting_time_seconds"].values[0]
        ft_travel = ft_row["average_travel_time_seconds"].values[0]
        ft_queue = ft_row["average_queue_length"].values[0]
        for ctrl in controllers:
            ctrl_row = sc_data[sc_data["controller"] == ctrl]
            if ctrl_row.empty:
                continue
            c_wait = ctrl_row["average_waiting_time_seconds"].values[0]
            c_travel = ctrl_row["average_travel_time_seconds"].values[0]
            c_queue = ctrl_row["average_queue_length"].values[0]
            records.append(
                {
                    "scenario": sc,
                    "controller": ctrl,
                    "waiting_time_reduction_pct": round((ft_wait - c_wait) / ft_wait * 100, 2),
                    "travel_time_reduction_pct": round((ft_travel - c_travel) / ft_travel * 100, 2),
                    "queue_reduction_pct": round((ft_queue - c_queue) / ft_queue * 100, 2),
                }
            )
    imp_df = pd.DataFrame(records)
    imp_df.to_csv(output_dir / "improvement_vs_fixed_time.csv", index=False)
    return imp_df


def generate_paired_significance_table(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    fixed_time = df.loc[df["controller"].eq("fixed_time")].set_index(["scenario", "seed"])
    records = []
    for scenario in sorted(df["scenario"].unique()):
        for controller in sorted(set(df["controller"]) - {"fixed_time"}):
            candidate = df.loc[
                df["scenario"].eq(scenario) & df["controller"].eq(controller)
            ].set_index(["scenario", "seed"])
            improvement = (
                fixed_time.loc[candidate.index, "average_waiting_time_seconds"]
                - candidate["average_waiting_time_seconds"]
            )
            statistic, p_value = wilcoxon(improvement, alternative="greater", method="auto")
            records.append(
                {
                    "scenario": scenario,
                    "controller": controller,
                    "mean_waiting_time_reduction_seconds": round(float(improvement.mean()), 2),
                    "wilcoxon_statistic": round(float(statistic), 2),
                    "p_value_one_sided": round(float(p_value), 4),
                }
            )
    significance = pd.DataFrame(records)
    significance.to_csv(output_dir / "paired_wilcoxon_waiting_time.csv", index=False)
    return significance


def plot_comparison_by_scenario(df: pd.DataFrame, output_dir: Path) -> None:
    scenarios = sorted(df["scenario"].unique())
    controllers = sorted(df["controller"].unique())
    metrics = [
        ("average_waiting_time_seconds", "Average Waiting Time (s)"),
        ("average_travel_time_seconds", "Average Travel Time (s)"),
        ("average_queue_length", "Average Queue Length"),
        ("average_stops", "Average Stops"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (metric, title) in zip(axes, metrics):
        grouped = df.groupby(["scenario", "controller"])[metric].mean().unstack("controller")
        grouped.plot(kind="bar", ax=ax, width=0.8, rot=0)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Scenario")
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.legend(title="Controller")

    fig.tight_layout()
    fig.savefig(output_dir / "controller_comparison_by_scenario.png", dpi=150)
    plt.close(fig)


def plot_boxplots(df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df.boxplot(
        column="average_waiting_time_seconds",
        by="controller",
        ax=axes[0],
        grid=True,
    )
    axes[0].set_title("Waiting Time Distribution across Runs")
    axes[0].set_ylabel("Seconds")
    axes[0].set_xlabel("Controller")

    df.boxplot(
        column="average_queue_length",
        by="controller",
        ax=axes[1],
        grid=True,
    )
    axes[1].set_title("Queue Length Distribution across Runs")
    axes[1].set_ylabel("Halted Vehicles")
    axes[1].set_xlabel("Controller")

    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(output_dir / "boxplots.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        sys.exit(f"Error: input file {args.input} does not exist.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} records from {args.input}")

    generate_summary_table(df, args.output_dir)
    generate_improvement_table(df, args.output_dir)
    generate_paired_significance_table(df, args.output_dir)
    plot_comparison_by_scenario(df, args.output_dir)
    plot_boxplots(df, args.output_dir)

    print(f"Analysis saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
