from __future__ import annotations

import tempfile
import xml.etree.ElementTree as element_tree
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import traci

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIMULATION_SOURCE = PROJECT_ROOT / "src" / "simulation"

import sys

sys.path.insert(0, str(SIMULATION_SOURCE))

from run_simulation import find_sumo_binary
from scenarios import generate_scenario_for_seed

NETWORK_FILE = PROJECT_ROOT / "simulation" / "girona_gran_via_eixample.net.xml"


@dataclass(frozen=True)
class SimulationMetrics:
    controller: str
    scenario: str
    seed: int
    duration_seconds: int
    traffic_lights: int
    departed_vehicles: int
    arrived_vehicles: int
    completed_trips: int
    unfinished_trips: int
    teleport_events: int
    throughput_vehicles_per_hour: float
    average_waiting_time_seconds: float
    average_travel_time_seconds: float
    average_stops: float
    average_queue_length: float
    max_queue_length: int
    max_running_vehicles: int
    simulation_time_seconds: float

    def as_dict(self) -> dict[str, float | int | str]:
        return self.__dict__.copy()


def build_sumo_command(
    scenario_name: str,
    route_seed: int,
    additional_files: list[Path] | None = None,
) -> tuple[list[str], Any]:
    scenario = generate_scenario_for_seed(scenario_name, route_seed)
    command = [
        find_sumo_binary(gui=False),
        "--net-file",
        str(NETWORK_FILE),
        "--route-files",
        str(scenario.route_file_for_seed(route_seed)),
        "--begin",
        "0",
        "--end",
        str(scenario.duration_seconds),
        "--step-length",
        "1",
        "--time-to-teleport",
        "-1",
        "--seed",
        "42",
        "--no-step-log",
        "true",
        "--no-warnings",
        "true",
        "--duration-log.disable",
        "true",
    ]
    files = list(additional_files or [])
    if scenario.additional_file:
        files.append(scenario.additional_file)
    if files:
        command.extend(["--additional-files", ",".join(str(file) for file in files)])
    if scenario.additional_file:
        command.extend(["--device.rerouting.mode", "8"])
    return command, scenario


def run_controller(
    controller: str,
    scenario_name: str,
    route_seed: int,
    additional_files: list[Path] | None = None,
    action_callback: Any | None = None,
    decision_interval_seconds: int = 10,
) -> SimulationMetrics:
    command, scenario = build_sumo_command(scenario_name, route_seed, additional_files)
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)
        tripinfo_output = temporary_path / "tripinfo.xml"
        summary_output = temporary_path / "summary.xml"
        command.extend(
            [
                "--tripinfo-output",
                str(tripinfo_output),
                "--tripinfo-output.write-unfinished",
                "true",
                "--summary-output",
                str(summary_output),
                "--summary-output.period",
                "1",
            ]
        )
        connection_label = f"girona_metrics_{id(command)}"
        traci.start(command, label=connection_label)
        connection = traci.getConnection(connection_label)
        lane_ids = [
            lane_id for lane_id in connection.lane.getIDList() if not lane_id.startswith(":")
        ]
        traffic_lights = len(connection.trafficlight.getIDList())
        departed_vehicles = 0
        arrived_vehicles = 0
        teleport_events = 0
        total_queue_length = 0
        max_queue_length = 0
        queue_samples = 0

        try:
            while connection.simulation.getMinExpectedNumber() > 0:
                simulation_time = int(connection.simulation.getTime())
                if action_callback and simulation_time % decision_interval_seconds == 0:
                    action_callback(connection)
                connection.simulationStep()
                departed_vehicles += connection.simulation.getDepartedNumber()
                arrived_vehicles += connection.simulation.getArrivedNumber()
                teleport_events += connection.simulation.getStartingTeleportNumber()
                queue_length = sum(
                    connection.lane.getLastStepHaltingNumber(lane_id) for lane_id in lane_ids
                )
                total_queue_length += queue_length
                max_queue_length = max(max_queue_length, queue_length)
                queue_samples += 1
            simulation_time = connection.simulation.getTime()
        finally:
            connection.close()

        tripinfos = element_tree.parse(tripinfo_output).getroot().findall("tripinfo")
        summary_steps = element_tree.parse(summary_output).getroot().findall("step")

    completed_tripinfos = [
        trip for trip in tripinfos if trip.get("arrival", "-1") != "-1"
    ]
    completed_trips = len(completed_tripinfos)
    if completed_trips == 0:
        raise RuntimeError("No trips completed; metrics cannot be computed.")

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
    return SimulationMetrics(
        controller=controller,
        scenario=scenario.name,
        seed=route_seed,
        duration_seconds=scenario.duration_seconds,
        traffic_lights=traffic_lights,
        departed_vehicles=departed_vehicles,
        arrived_vehicles=arrived_vehicles,
        completed_trips=completed_trips,
        unfinished_trips=len(tripinfos) - completed_trips,
        teleport_events=teleport_events,
        throughput_vehicles_per_hour=completed_trips * 3600 / scenario.duration_seconds,
        average_waiting_time_seconds=total_waiting_time / completed_trips,
        average_travel_time_seconds=total_travel_time / completed_trips,
        average_stops=total_stops / completed_trips,
        average_queue_length=total_queue_length / queue_samples,
        max_queue_length=max_queue_length,
        max_running_vehicles=max_running_vehicles,
        simulation_time_seconds=simulation_time,
    )
