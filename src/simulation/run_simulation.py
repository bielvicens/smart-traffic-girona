from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import xml.etree.ElementTree as element_tree
from pathlib import Path

import traci

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "simulation" / "girona_gran_via_eixample.sumocfg"


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

    raise RuntimeError(
        f"Could not find {executable}. Set SUMO_HOME or add SUMO/bin to PATH."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Girona SUMO digital twin and report aggregate metrics."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to a SUMO configuration file.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Run SUMO with its graphical interface.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = args.config.resolve()
    if not config.exists():
        raise FileNotFoundError(f"SUMO configuration not found: {config}")

    with tempfile.TemporaryDirectory() as temporary_directory:
        tripinfo_output = Path(temporary_directory) / "tripinfo.xml"
        traci.start(
            [
                find_sumo_binary(args.gui),
                "-c",
                str(config),
                "--tripinfo-output",
                str(tripinfo_output),
            ]
        )
        tls_ids = traci.trafficlight.getIDList()
        departed_vehicles = 0

        try:
            while traci.simulation.getMinExpectedNumber() > 0:
                traci.simulationStep()
                departed_vehicles += traci.simulation.getDepartedNumber()
        finally:
            simulation_time = traci.simulation.getTime()
            traci.close()

        tripinfos = element_tree.parse(tripinfo_output).getroot().findall("tripinfo")

    completed_trips = len(tripinfos)
    total_waiting_time = sum(float(trip.get("waitingTime", "0")) for trip in tripinfos)
    total_travel_time = sum(float(trip.get("duration", "0")) for trip in tripinfos)
    total_stops = sum(int(trip.get("waitingCount", "0")) for trip in tripinfos)

    print(f"simulation_time_s={simulation_time:.0f}")
    print(f"traffic_lights={len(tls_ids)}")
    print(f"departed_vehicles={departed_vehicles}")
    print(f"completed_trips={completed_trips}")
    print(f"average_waiting_time_s={total_waiting_time / completed_trips:.2f}")
    print(f"average_travel_time_s={total_travel_time / completed_trips:.2f}")
    print(f"average_stops={total_stops / completed_trips:.2f}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError) as error:
        sys.exit(f"Error: {error}")
