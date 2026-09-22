from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import traci
import traci.constants as tc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIMULATION_SOURCE = PROJECT_ROOT / "src" / "simulation"
sys.path.insert(0, str(SIMULATION_SOURCE))

from run_simulation import find_sumo_binary
from scenarios import available_scenarios, generate_scenario_for_seed

DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "traffic_observations.csv"
SAMPLE_INTERVAL_SECONDS = 60


class CollectionError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect per-traffic-light traffic observations from SUMO scenarios."
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=available_scenarios(),
        default=available_scenarios(),
        help="Scenario names to simulate.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Number of consecutive demand seeds per scenario.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="First demand seed.",
    )
    parser.add_argument(
        "--sample-interval-seconds",
        type=int,
        default=SAMPLE_INTERVAL_SECONDS,
        help="Aggregation interval for observations.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--force-routes",
        action="store_true",
        help="Regenerate seed-specific route files before collection.",
    )
    return parser.parse_args()


def controlled_lanes_by_tls(tls_ids: list[str]) -> dict[str, tuple[str, ...]]:
    lanes = {}
    for tls_id in tls_ids:
        controlled = {
            lane_id
            for lane_id in traci.trafficlight.getControlledLanes(tls_id)
            if not lane_id.startswith(":")
        }
        lanes[tls_id] = tuple(sorted(controlled))
    return lanes


def subscribe_to_lanes(lane_ids: set[str]) -> None:
    for lane_id in lane_ids:
        traci.lane.subscribe(
            lane_id,
            [
                tc.LAST_STEP_VEHICLE_NUMBER,
                tc.LAST_STEP_VEHICLE_HALTING_NUMBER,
                tc.LAST_STEP_MEAN_SPEED,
                tc.LAST_STEP_VEHICLE_ID_LIST,
            ],
        )


def create_accumulators(tls_ids: list[str]) -> dict[str, dict[str, float | set[str]]]:
    return {
        tls_id: {
            "halting_sum": 0.0,
            "vehicle_sum": 0.0,
            "speed_weighted_sum": 0.0,
            "speed_weight": 0.0,
            "throughput": 0.0,
            "previous_vehicles": set(),
        }
        for tls_id in tls_ids
    }


def update_accumulators(
    accumulators: dict[str, dict[str, float | set[str]]],
    lanes_by_tls: dict[str, tuple[str, ...]],
) -> None:
    for tls_id, lanes in lanes_by_tls.items():
        lane_vehicle_ids = set()
        halting = 0
        vehicles = 0
        weighted_speed = 0.0
        for lane_id in lanes:
            results = traci.lane.getSubscriptionResults(lane_id)
            vehicle_count = int(results[tc.LAST_STEP_VEHICLE_NUMBER])
            vehicles += vehicle_count
            halting += int(results[tc.LAST_STEP_VEHICLE_HALTING_NUMBER])
            weighted_speed += float(results[tc.LAST_STEP_MEAN_SPEED]) * vehicle_count
            lane_vehicle_ids.update(results[tc.LAST_STEP_VEHICLE_ID_LIST])

        accumulator = accumulators[tls_id]
        previous_vehicles = accumulator["previous_vehicles"]
        assert isinstance(previous_vehicles, set)
        accumulator["throughput"] += len(previous_vehicles - lane_vehicle_ids)
        accumulator["halting_sum"] += halting
        accumulator["vehicle_sum"] += vehicles
        accumulator["speed_weighted_sum"] += weighted_speed
        accumulator["speed_weight"] += vehicles
        accumulator["previous_vehicles"] = lane_vehicle_ids


def observation_records(
    scenario_name: str,
    seed: int,
    simulation_time: int,
    accumulators: dict[str, dict[str, float | set[str]]],
    sample_interval_seconds: int,
) -> list[dict[str, float | int | str]]:
    records = []
    for tls_id, accumulator in accumulators.items():
        speed_weight = float(accumulator["speed_weight"])
        records.append(
            {
                "scenario": scenario_name,
                "seed": seed,
                "simulation_time_seconds": simulation_time,
                "intersection_id": tls_id,
                "queue_length": float(accumulator["halting_sum"]) / sample_interval_seconds,
                "vehicle_count": float(accumulator["vehicle_sum"]) / sample_interval_seconds,
                "mean_speed_mps": (
                    float(accumulator["speed_weighted_sum"]) / speed_weight
                    if speed_weight
                    else 0.0
                ),
                "throughput_vehicles": int(accumulator["throughput"]),
                "tls_phase": traci.trafficlight.getPhase(tls_id),
            }
        )
    return records


def collect_run(
    scenario_name: str,
    seed: int,
    sample_interval_seconds: int,
    force_routes: bool,
) -> list[dict[str, float | int | str]]:
    scenario = generate_scenario_for_seed(scenario_name, seed, force=force_routes)
    command = [
        find_sumo_binary(gui=False),
        "--net-file",
        str(PROJECT_ROOT / "simulation" / "girona_gran_via_eixample.net.xml"),
        "--route-files",
        str(scenario.route_file_for_seed(seed)),
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
    if scenario.additional_file:
        command.extend(
            [
                "--additional-files",
                str(scenario.additional_file),
                "--device.rerouting.mode",
                "8",
            ]
        )

    records = []
    connected = False
    try:
        traci.start(command)
        connected = True
        tls_ids = sorted(traci.trafficlight.getIDList())
        lanes_by_tls = controlled_lanes_by_tls(tls_ids)
        subscribe_to_lanes({lane_id for lanes in lanes_by_tls.values() for lane_id in lanes})
        accumulators = create_accumulators(tls_ids)

        while traci.simulation.getTime() < scenario.duration_seconds:
            traci.simulationStep()
            update_accumulators(accumulators, lanes_by_tls)
            simulation_time = int(traci.simulation.getTime())
            if simulation_time % sample_interval_seconds == 0:
                records.extend(
                    observation_records(
                        scenario.name,
                        seed,
                        simulation_time,
                        accumulators,
                        sample_interval_seconds,
                    )
                )
                accumulators = create_accumulators(tls_ids)
    except traci.exceptions.FatalTraCIError as error:
        raise CollectionError(str(error)) from error
    finally:
        if connected:
            traci.close()

    return records


def write_records(records: list[dict[str, float | int | str]], output_file: Path) -> None:
    if not records:
        raise CollectionError("No observation records were collected.")

    output_file = output_file.resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0])
    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    if args.seeds <= 0 or args.sample_interval_seconds <= 0:
        raise ValueError("Seeds and sampling interval must be positive.")

    records = []
    for scenario_name in args.scenarios:
        for seed in range(args.base_seed, args.base_seed + args.seeds):
            run_records = collect_run(
                scenario_name,
                seed,
                args.sample_interval_seconds,
                args.force_routes,
            )
            records.extend(run_records)
            print(
                f"collected scenario={scenario_name} seed={seed} "
                f"records={len(run_records)}"
            )

    write_records(records, args.output)
    print(f"output={args.output.resolve()}")
    print(f"total_records={len(records)}")


if __name__ == "__main__":
    try:
        main()
    except (CollectionError, FileNotFoundError, RuntimeError, ValueError) as error:
        sys.exit(f"Error: {error}")
