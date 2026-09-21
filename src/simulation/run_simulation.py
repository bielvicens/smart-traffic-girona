from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile
import xml.etree.ElementTree as element_tree
from pathlib import Path

import traci

from scenarios import PROJECT_ROOT, generate_scenario

DEFAULT_SCENARIO = "normal"
DEFAULT_RESULTS_FILE = PROJECT_ROOT / "experiments" / "baseline" / "fixed_time_metrics.csv"


class SimulationError(RuntimeError):
    pass


def find_sumo_binary(gui: bool) -> str:
    executable = "sumo-gui" if gui else "sumo"
    binary = shutil.which(executable)
    if binary:
        return binary

    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        candidate = Path(sumo_home) / "bin" / f"{executable}.exe"
        if candidate.exists():
            return str(candidate)

    windows_default = Path(r"C:\Program Files (x86)\Eclipse\Sumo")
    candidate = windows_default / "bin" / f"{executable}.exe"
    if candidate.exists():
        return str(candidate)

    raise RuntimeError(
        f"Could not find {executable}. Set SUMO_HOME or add SUMO/bin to PATH."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Girona SUMO scenario and report fixed-time baseline metrics."
    )
    parser.add_argument(
        "--scenario",
        default=DEFAULT_SCENARIO,
        help="Scenario name from configs/scenarios.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the scenario seed used by SUMO.",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=DEFAULT_RESULTS_FILE,
        help="CSV file to append the result to.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print metrics without appending a CSV result.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Run SUMO with its graphical interface.",
    )
    return parser.parse_args()


def run_scenario(scenario_name: str, seed: int | None, gui: bool) -> dict[str, float | int | str]:
    scenario = generate_scenario(scenario_name)
    simulation_seed = scenario.seed if seed is None else seed

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)
        tripinfo_output = temporary_path / "tripinfo.xml"
        summary_output = temporary_path / "summary.xml"
        command = [
            find_sumo_binary(gui),
            "--net-file",
            str(PROJECT_ROOT / "simulation" / "girona_gran_via_eixample.net.xml"),
            "--route-files",
            str(scenario.route_file),
            "--begin",
            "0",
            "--end",
            str(scenario.duration_seconds),
            "--step-length",
            "1",
            "--time-to-teleport",
            "-1",
            "--seed",
            str(simulation_seed),
            "--tripinfo-output",
            str(tripinfo_output),
            "--tripinfo-output.write-unfinished",
            "true",
            "--summary-output",
            str(summary_output),
            "--summary-output.period",
            "1",
            "--no-step-log",
            "true",
            "--no-warnings",
            "true",
            "--duration-log.disable",
            "true",
        ]
        if scenario.additional_file:
            command.extend(
                [
                    "--additional-files",
                    str(scenario.additional_file),
                    "--device.rerouting.mode",
                    "8",
                ]
            )

        connected = False
        try:
            traci.start(command)
            connected = True
            traffic_lights = len(traci.trafficlight.getIDList())
            lane_ids = [lane_id for lane_id in traci.lane.getIDList() if not lane_id.startswith(":")]
            departed_vehicles = 0
            arrived_vehicles = 0
            teleport_events = 0
            total_queue_length = 0
            max_queue_length = 0
            queue_samples = 0

            while traci.simulation.getMinExpectedNumber() > 0:
                traci.simulationStep()
                departed_vehicles += traci.simulation.getDepartedNumber()
                arrived_vehicles += traci.simulation.getArrivedNumber()
                teleport_events += traci.simulation.getStartingTeleportNumber()
                queue_length = sum(
                    traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in lane_ids
                )
                total_queue_length += queue_length
                max_queue_length = max(max_queue_length, queue_length)
                queue_samples += 1

            simulation_time = traci.simulation.getTime()
        except traci.exceptions.FatalTraCIError as error:
            raise SimulationError(str(error)) from error
        finally:
            if connected:
                traci.close()

        tripinfos = element_tree.parse(tripinfo_output).getroot().findall("tripinfo")
        summary_steps = element_tree.parse(summary_output).getroot().findall("step")

    completed_tripinfos = [
        trip for trip in tripinfos if trip.get("arrival", "-1") != "-1"
    ]
    completed_trips = len(completed_tripinfos)
    if completed_trips == 0:
        raise SimulationError("No trips completed; metrics cannot be computed.")

    total_waiting_time = sum(
        float(trip.get("waitingTime", "0")) for trip in completed_tripinfos
    )
    total_travel_time = sum(
        float(trip.get("duration", "0")) for trip in completed_tripinfos
    )
    total_stops = sum(
        int(trip.get("waitingCount", "0")) for trip in completed_tripinfos
    )
    max_running_vehicles = max(
        (int(step.get("running", "0")) for step in summary_steps), default=0
    )

    return {
        "controller": "fixed_time",
        "scenario": scenario.name,
        "seed": simulation_seed,
        "duration_seconds": scenario.duration_seconds,
        "traffic_lights": traffic_lights,
        "departed_vehicles": departed_vehicles,
        "arrived_vehicles": arrived_vehicles,
        "completed_trips": completed_trips,
        "unfinished_trips": len(tripinfos) - completed_trips,
        "teleport_events": teleport_events,
        "throughput_vehicles_per_hour": completed_trips * 3600 / scenario.duration_seconds,
        "average_waiting_time_seconds": total_waiting_time / completed_trips,
        "average_travel_time_seconds": total_travel_time / completed_trips,
        "average_stops": total_stops / completed_trips,
        "average_queue_length": total_queue_length / queue_samples,
        "max_queue_length": max_queue_length,
        "max_running_vehicles": max_running_vehicles,
        "simulation_time_seconds": simulation_time,
    }


def append_result(metrics: dict[str, float | int | str], results_file: Path) -> None:
    results_file = results_file.resolve()
    results_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(metrics)
    file_exists = results_file.exists()
    with results_file.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)


def print_metrics(metrics: dict[str, float | int | str]) -> None:
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}={value:.2f}")
        else:
            print(f"{key}={value}")


def main() -> None:
    args = parse_args()
    metrics = run_scenario(args.scenario, args.seed, args.gui)
    if not args.no_save:
        append_result(metrics, args.results_file)
    print_metrics(metrics)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, SimulationError) as error:
        sys.exit(f"Error: {error}")
