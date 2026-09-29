from __future__ import annotations

import argparse
import csv
import sys
import xml.etree.ElementTree as element_tree
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REINFORCEMENT_LEARNING_SOURCE = PROJECT_ROOT / "src" / "reinforcement_learning"
sys.path.insert(0, str(REINFORCEMENT_LEARNING_SOURCE))

from metrics import run_controller

NETWORK_FILE = PROJECT_ROOT / "simulation" / "girona_gran_via_eixample.net.xml"
DEFAULT_ADDITIONAL_FILE = PROJECT_ROOT / "simulation" / "additional" / "actuated.add.xml"
DEFAULT_RESULTS_FILE = PROJECT_ROOT / "experiments" / "rl" / "actuated_metrics.csv"


def is_green_phase(state: str) -> bool:
    return "G" in state or "g" in state


def generate_actuated_programs(output_file: Path = DEFAULT_ADDITIONAL_FILE) -> Path:
    network_root = element_tree.parse(NETWORK_FILE).getroot()
    additional_root = element_tree.Element(
        "additional",
        {
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/additional_file.xsd",
        },
    )
    for logic in network_root.findall("tlLogic"):
        logic_element = element_tree.SubElement(
            additional_root,
            "tlLogic",
            {
                "id": logic.get("id", ""),
                "type": "actuated",
                "programID": "actuated",
                "offset": logic.get("offset", "0"),
            },
        )
        for phase in logic.findall("phase"):
            state = phase.get("state", "")
            attributes = {
                "duration": phase.get("duration", "1"),
                "state": state,
            }
            if is_green_phase(state):
                duration = int(float(phase.get("duration", "1")))
                attributes["minDur"] = str(min(10, duration))
                attributes["maxDur"] = str(max(duration, 60))
            element_tree.SubElement(logic_element, "phase", attributes)

    element_tree.indent(additional_root, space="    ")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    element_tree.ElementTree(additional_root).write(
        output_file, encoding="UTF-8", xml_declaration=True
    )
    return output_file


def append_result(metrics: dict[str, float | int | str], results_file: Path) -> None:
    results_file.parent.mkdir(parents=True, exist_ok=True)
    file_exists = results_file.exists()
    with results_file.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics))
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SUMO's actuated traffic-light baseline.")
    parser.add_argument("--scenario", default="normal")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--additional-file", type=Path, default=DEFAULT_ADDITIONAL_FILE)
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS_FILE)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    additional_file = generate_actuated_programs(args.additional_file)
    for seed in args.seeds:
        metrics = run_controller(
            controller="actuated",
            scenario_name=args.scenario,
            route_seed=seed,
            additional_files=[additional_file],
        ).as_dict()
        if not args.no_save:
            append_result(metrics, args.results_file)
        print(", ".join(f"{key}={value}" for key, value in metrics.items()))


if __name__ == "__main__":
    main()
