from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as element_tree
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIRECTORY = PROJECT_ROOT / "configs" / "scenarios"
SIMULATION_DIRECTORY = PROJECT_ROOT / "simulation"
NETWORK_FILE = SIMULATION_DIRECTORY / "girona_gran_via_eixample.net.xml"


@dataclass(frozen=True)
class DemandBlock:
    begin_seconds: int
    end_seconds: int
    period_seconds: float


@dataclass(frozen=True)
class Incident:
    begin_seconds: int
    end_seconds: int
    closed_edge: str
    notification_edges: tuple[str, ...]
    disallow: str


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    duration_seconds: int
    seed: int
    demand_blocks: tuple[DemandBlock, ...]
    fringe_factor: str
    min_distance_meters: int
    incident: Incident | None = None

    @property
    def route_file(self) -> Path:
        return SIMULATION_DIRECTORY / f"{self.name}.rou.xml"

    def route_file_for_seed(self, seed: int) -> Path:
        return SIMULATION_DIRECTORY / "generated" / f"{self.name}_seed_{seed}.rou.xml"

    @property
    def additional_file(self) -> Path | None:
        if self.incident is None:
            return None
        return SIMULATION_DIRECTORY / "additional" / f"{self.name}.add.xml"


def find_sumo_home() -> Path:
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        candidate = Path(sumo_home)
        if (candidate / "tools" / "randomTrips.py").exists():
            return candidate

    sumo_binary = shutil.which("sumo")
    if sumo_binary:
        candidate = Path(sumo_binary).resolve().parent.parent
        if (candidate / "tools" / "randomTrips.py").exists():
            return candidate

    windows_default = Path(r"C:\Program Files (x86)\Eclipse\Sumo")
    if (windows_default / "tools" / "randomTrips.py").exists():
        return windows_default

    raise RuntimeError("Could not find SUMO. Set SUMO_HOME to the SUMO installation.")


def available_scenarios() -> list[str]:
    return sorted(config_file.stem for config_file in CONFIG_DIRECTORY.glob("*.json"))


def scenario_config_path(name: str) -> Path:
    return CONFIG_DIRECTORY / f"{name}.json"


def scenario_is_current(scenario: Scenario) -> bool:
    required_files = [scenario.route_file, scenario_config_path(scenario.name)]
    if scenario.additional_file:
        required_files.append(scenario.additional_file)

    if any(not file.exists() for file in required_files):
        return False

    generated_files = [scenario.route_file]
    if scenario.additional_file:
        generated_files.append(scenario.additional_file)
    newest_source = max(NETWORK_FILE.stat().st_mtime, scenario_config_path(scenario.name).stat().st_mtime)
    return all(file.stat().st_mtime >= newest_source for file in generated_files)


def load_scenario(name: str) -> Scenario:
    config_file = CONFIG_DIRECTORY / f"{name}.json"
    if not config_file.exists():
        available = ", ".join(available_scenarios())
        raise FileNotFoundError(
            f"Unknown scenario '{name}'. Available scenarios: {available}."
        )

    with config_file.open(encoding="utf-8") as file:
        config = json.load(file)

    demand_blocks = tuple(
        DemandBlock(
            begin_seconds=block["begin_seconds"],
            end_seconds=block["end_seconds"],
            period_seconds=block["period_seconds"],
        )
        for block in config["demand_blocks"]
    )
    incident_config = config.get("incident")
    incident = None
    if incident_config:
        incident = Incident(
            begin_seconds=incident_config["begin_seconds"],
            end_seconds=incident_config["end_seconds"],
            closed_edge=incident_config["closed_edge"],
            notification_edges=tuple(incident_config["notification_edges"]),
            disallow=incident_config["disallow"],
        )

    scenario = Scenario(
        name=config["name"],
        description=config["description"],
        duration_seconds=config["duration_seconds"],
        seed=config["seed"],
        demand_blocks=demand_blocks,
        fringe_factor=config["fringe_factor"],
        min_distance_meters=config["min_distance_meters"],
        incident=incident,
    )
    validate_scenario(scenario)
    return scenario


def validate_scenario(scenario: Scenario) -> None:
    if scenario.name not in available_scenarios():
        raise ValueError(f"Scenario name '{scenario.name}' does not match its config file.")
    if scenario.duration_seconds <= 0 or not scenario.demand_blocks:
        raise ValueError("A scenario must have a positive duration and at least one demand block.")

    previous_end = 0
    for block in scenario.demand_blocks:
        if block.begin_seconds != previous_end:
            raise ValueError("Demand blocks must be contiguous and start at time zero.")
        if block.end_seconds <= block.begin_seconds or block.period_seconds <= 0:
            raise ValueError("Demand blocks need positive durations and periods.")
        previous_end = block.end_seconds

    if previous_end != scenario.duration_seconds:
        raise ValueError("Demand blocks must end at duration_seconds.")

    if scenario.incident:
        if not 0 <= scenario.incident.begin_seconds < scenario.incident.end_seconds:
            raise ValueError("Incident times must be ordered and non-negative.")
        if scenario.incident.end_seconds > scenario.duration_seconds:
            raise ValueError("Incident must end within the scenario duration.")
        if not scenario.incident.notification_edges:
            raise ValueError("An incident needs at least one notification edge.")


def generate_scenario(name: str, force: bool = False) -> Scenario:
    scenario = load_scenario(name)
    if not NETWORK_FILE.exists():
        raise FileNotFoundError(f"SUMO network not found: {NETWORK_FILE}")

    if force or not scenario_is_current(scenario):
        generate_route_file(scenario, scenario.route_file, scenario.seed)
        if scenario.incident:
            generate_incident_file(scenario)
    return scenario


def generate_scenario_for_seed(name: str, seed: int, force: bool = False) -> Scenario:
    scenario = load_scenario(name)
    if not NETWORK_FILE.exists():
        raise FileNotFoundError(f"SUMO network not found: {NETWORK_FILE}")

    route_file = scenario.route_file_for_seed(seed)
    newest_source = max(NETWORK_FILE.stat().st_mtime, scenario_config_path(name).stat().st_mtime)
    if force or not route_file.exists() or route_file.stat().st_mtime < newest_source:
        generate_route_file(scenario, route_file, seed)
    if scenario.incident and (force or not scenario.additional_file.exists()):
        generate_incident_file(scenario)
    return scenario


def generate_route_file(scenario: Scenario, output_file: Path, seed: int) -> None:
    sumo_home = find_sumo_home()
    random_trips = sumo_home / "tools" / "randomTrips.py"

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)
        route_files = []
        for index, block in enumerate(scenario.demand_blocks):
            route_file = temporary_path / f"block_{index}.rou.xml"
            trip_file = temporary_path / f"block_{index}.trips.xml"
            command = [
                sys.executable,
                str(random_trips),
                "--net-file",
                str(NETWORK_FILE),
                "--route-file",
                str(route_file),
                "--output-trip-file",
                str(trip_file),
                "--begin",
                str(block.begin_seconds),
                "--end",
                str(block.end_seconds),
                "--period",
                str(block.period_seconds),
                "--fringe-factor",
                scenario.fringe_factor,
                "--min-distance",
                str(scenario.min_distance_meters),
                "--seed",
                str(seed + index),
                "--validate",
                "--remove-loops",
                "--random-departpos",
                "--prefix",
                f"{scenario.name}_{index}_",
            ]
            subprocess.run(command, check=True, cwd=PROJECT_ROOT)
            route_files.append(route_file)

        merge_route_files(route_files, output_file)


def merge_route_files(route_files: list[Path], output_file: Path) -> None:
    root = element_tree.Element("routes")
    for route_file in route_files:
        source_root = element_tree.parse(route_file).getroot()
        for vehicle in source_root.findall("vehicle"):
            root.append(vehicle)

    root[:] = sorted(root, key=lambda vehicle: float(vehicle.get("depart", "0")))
    element_tree.indent(root, space="    ")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    element_tree.ElementTree(root).write(
        output_file, encoding="UTF-8", xml_declaration=True
    )


def generate_incident_file(scenario: Scenario) -> None:
    incident = scenario.incident
    if incident is None or scenario.additional_file is None:
        return

    root = element_tree.Element(
        "additional",
        {
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/additional_file.xsd",
        },
    )
    rerouter = element_tree.SubElement(
        root,
        "rerouter",
        {
            "id": f"closure_{incident.closed_edge.replace('#', '_').replace('-', 'neg_')}",
            "edges": " ".join(incident.notification_edges),
        },
    )
    interval = element_tree.SubElement(
        rerouter,
        "interval",
        {"begin": str(incident.begin_seconds), "end": str(incident.end_seconds)},
    )
    element_tree.SubElement(
        interval,
        "closingReroute",
        {"id": incident.closed_edge, "disallow": incident.disallow},
    )
    element_tree.indent(root, space="    ")
    scenario.additional_file.parent.mkdir(parents=True, exist_ok=True)
    element_tree.ElementTree(root).write(
        scenario.additional_file, encoding="UTF-8", xml_declaration=True
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reproducible SUMO demand and incident inputs."
    )
    parser.add_argument("scenario", choices=available_scenarios())
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Generate a seed-specific route file under simulation/generated/.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Regenerate existing scenario inputs."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is None:
        scenario = generate_scenario(args.scenario, force=args.force)
        route_file = scenario.route_file
    else:
        scenario = generate_scenario_for_seed(args.scenario, args.seed, force=args.force)
        route_file = scenario.route_file_for_seed(args.seed)
    print(f"scenario={scenario.name}")
    print(f"route_file={route_file.relative_to(PROJECT_ROOT)}")
    if scenario.additional_file:
        print(f"additional_file={scenario.additional_file.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        sys.exit(f"Error: {error}")
