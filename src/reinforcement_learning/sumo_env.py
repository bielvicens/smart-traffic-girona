from __future__ import annotations

import json
import shutil
import sys
import tempfile
import xml.etree.ElementTree as element_tree
from itertools import product
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import traci

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIMULATION_SOURCE = PROJECT_ROOT / "src" / "simulation"
sys.path.insert(0, str(SIMULATION_SOURCE))

from run_simulation import find_sumo_binary
from scenarios import generate_scenario_for_seed

NETWORK_FILE = PROJECT_ROOT / "simulation" / "girona_gran_via_eixample.net.xml"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "rl" / "default.json"
OBSERVATION_FEATURES_PER_TLS = 5


class SumoEnvironmentError(RuntimeError):
    pass


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def is_actionable_phase(state: str) -> bool:
    return ("G" in state or "g" in state) and "y" not in state


def network_control_specification() -> tuple[tuple[str, ...], dict[str, tuple[int, ...]]]:
    network_root = element_tree.parse(NETWORK_FILE).getroot()
    phase_indices = {}
    for logic in network_root.findall("tlLogic"):
        tls_id = logic.get("id", "")
        phase_indices[tls_id] = tuple(
            index
            for index, phase in enumerate(logic.findall("phase"))
            if is_actionable_phase(phase.get("state", ""))
        )
    tls_ids = tuple(sorted(phase_indices))
    if not tls_ids or any(not phase_indices[tls_id] for tls_id in tls_ids):
        raise SumoEnvironmentError("The SUMO network must define an actionable phase for each TLS.")
    return tls_ids, phase_indices


class GironaTrafficEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG,
        scenario_name: str | None = None,
        route_seeds: list[int] | tuple[int, ...] | None = None,
        gui: bool = False,
        episode_duration_seconds: int | None = None,
    ) -> None:
        super().__init__()
        self.config = load_config(config_path)
        self.config_path = config_path
        self.scenario_name = scenario_name or self.config["scenario"]
        self.route_seeds = tuple(route_seeds or self.config["training_seeds"])
        self.gui = gui
        self.decision_interval_seconds = int(self.config["decision_interval_seconds"])
        self.episode_duration_seconds = int(
            episode_duration_seconds or self.config["episode_duration_seconds"]
        )
        self.min_green_seconds = int(self.config["min_green_seconds"])
        self.yellow_seconds = int(self.config["yellow_seconds"])
        self.all_red_seconds = int(self.config["all_red_seconds"])
        self._owns_traci = False
        self._connection_label = f"girona_rl_{id(self)}"
        self._traci: Any | None = None
        self._temporary_directory: tempfile.TemporaryDirectory[str] | None = None
        self._tls_ids: tuple[str, ...] = ()
        self._lanes_by_tls: dict[str, tuple[str, ...]] = {}
        self._green_phase_indices: dict[str, tuple[int, ...]] = {}
        self._phase_states: dict[str, tuple[str, ...]] = {}
        self._current_green_actions: np.ndarray | None = None
        self._phase_started_at: dict[str, float] = {}
        self._transition_phase_indices: dict[tuple[str, int, int], tuple[int, ...]] = {}
        self._action_vectors: tuple[tuple[int, ...], ...] = ()
        self._route_seed = 0
        tls_ids, green_indices = network_control_specification()
        dimensions = np.asarray([len(green_indices[tls_id]) for tls_id in tls_ids], dtype=np.int64)
        self.action_space: gym.spaces.Space[np.ndarray] = gym.spaces.MultiDiscrete(dimensions)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=np.finfo(np.float32).max,
            shape=(len(tls_ids) * OBSERVATION_FEATURES_PER_TLS,),
            dtype=np.float32,
        )
        self._action_vectors = tuple(product(*(range(size) for size in dimensions)))

    @property
    def tls_ids(self) -> tuple[str, ...]:
        return self._tls_ids

    @property
    def action_dimensions(self) -> tuple[int, ...]:
        return tuple(len(self._green_phase_indices[tls_id]) for tls_id in self._tls_ids)

    @property
    def discrete_action_count(self) -> int:
        return len(self._action_vectors)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.close()
        options = options or {}
        route_seed = int(options.get("route_seed", self.np_random.choice(self.route_seeds)))
        self._start_sumo(route_seed)
        observation = self._observation()
        info = {
            "route_seed": route_seed,
            "traffic_lights": len(self._tls_ids),
            "discrete_action_count": self.discrete_action_count,
        }
        return observation, info

    def step(
        self, action: np.ndarray | list[int] | tuple[int, ...]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._traci is None:
            raise SumoEnvironmentError("Call reset before step.")

        requested_actions = np.asarray(action, dtype=np.int64)
        if requested_actions.shape != self.action_space.shape:
            raise ValueError(
                f"Action shape {requested_actions.shape} does not match {self.action_space.shape}."
            )
        if not self.action_space.contains(requested_actions):
            raise ValueError("Action is outside the MultiDiscrete action space.")

        transition_seconds = self._apply_actions(requested_actions)
        self._advance(max(0, self.decision_interval_seconds - transition_seconds))
        observation = self._observation()
        queue_length = self._network_queue_length()
        reward = -float(queue_length)
        simulation_time = self._traci.simulation.getTime()
        terminated = simulation_time >= self.episode_duration_seconds
        info = {
            "simulation_time_seconds": simulation_time,
            "queue_length": queue_length,
            "route_seed": self._route_seed,
        }
        return observation, reward, terminated, False, info

    def close(self) -> None:
        if self._traci is not None and self._owns_traci:
            try:
                self._traci.close()
            except KeyError:
                pass
        self._traci = None
        self._owns_traci = False
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
            self._temporary_directory = None

    def attach_connection(self, connection: Any) -> None:
        self._traci = connection
        self._owns_traci = False
        self._configure_traffic_lights()

    def _start_sumo(self, route_seed: int) -> None:
        scenario = generate_scenario_for_seed(self.scenario_name, route_seed)
        self._route_seed = route_seed
        self._temporary_directory = tempfile.TemporaryDirectory()
        network_file = self._temporary_network_file(Path(self._temporary_directory.name))
        command = [
            find_sumo_binary(self.gui),
            "--net-file",
            str(network_file),
            "--route-files",
            str(scenario.route_file_for_seed(route_seed)),
            "--begin",
            "0",
            "--end",
            str(self.episode_duration_seconds),
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
        traci.start(command, label=self._connection_label)
        self._traci = traci.getConnection(self._connection_label)
        self._owns_traci = True
        self._configure_traffic_lights()

    def _temporary_network_file(self, temporary_directory: Path) -> Path:
        output_file = temporary_directory / NETWORK_FILE.name
        shutil.copy2(NETWORK_FILE, output_file)
        root = element_tree.parse(output_file).getroot()
        for logic in root.findall("tlLogic"):
            for phase in logic.findall("phase"):
                state = phase.get("state", "")
                if is_actionable_phase(state):
                    phase.set(
                        "duration",
                        str(self.episode_duration_seconds + self.decision_interval_seconds),
                    )
                else:
                    phase.set("duration", "1")
        element_tree.ElementTree(root).write(output_file, encoding="UTF-8", xml_declaration=True)
        return output_file

    def _configure_traffic_lights(self) -> None:
        assert self._traci is not None
        self._tls_ids = tuple(sorted(self._traci.trafficlight.getIDList()))
        self._lanes_by_tls = {
            tls_id: tuple(
                sorted(
                    {
                        lane_id
                        for lane_id in self._traci.trafficlight.getControlledLanes(tls_id)
                        if not lane_id.startswith(":")
                    }
                )
            )
            for tls_id in self._tls_ids
        }
        self._green_phase_indices = {}
        self._phase_states = {}
        self._transition_phase_indices = {}
        initial_actions = []
        for tls_id in self._tls_ids:
            logic = self._traci.trafficlight.getAllProgramLogics(tls_id)[0]
            phase_states = tuple(phase.state for phase in logic.phases)
            green_indices = tuple(
                index
                for index, state in enumerate(phase_states)
                if is_actionable_phase(state)
            )
            if not green_indices:
                raise SumoEnvironmentError(f"Traffic light '{tls_id}' has no actionable green phase.")
            self._phase_states[tls_id] = phase_states
            self._green_phase_indices[tls_id] = green_indices
            current_phase = self._traci.trafficlight.getPhase(tls_id)
            initial_actions.append(
                green_indices.index(current_phase) if current_phase in green_indices else 0
            )
            for source_action, source_phase in enumerate(green_indices):
                for target_action, target_phase in enumerate(green_indices):
                    if source_phase == target_phase:
                        continue
                    self._transition_phase_indices[(tls_id, source_action, target_action)] = (
                        self._transition_path(phase_states, source_phase, target_phase)
                    )

        self._current_green_actions = np.asarray(initial_actions, dtype=np.int64)
        now = self._traci.simulation.getTime()
        self._phase_started_at = {tls_id: now for tls_id in self._tls_ids}
        dimensions = np.asarray(self.action_dimensions, dtype=np.int64)
        self.action_space = gym.spaces.MultiDiscrete(dimensions)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=np.finfo(np.float32).max,
            shape=(len(self._tls_ids) * OBSERVATION_FEATURES_PER_TLS,),
            dtype=np.float32,
        )
        self._action_vectors = tuple(product(*(range(size) for size in dimensions)))

    def _transition_path(
        self, phase_states: tuple[str, ...], source_phase: int, target_phase: int
    ) -> tuple[int, ...]:
        phase_index = (source_phase + 1) % len(phase_states)
        path = []
        while phase_index != target_phase:
            if not is_actionable_phase(phase_states[phase_index]):
                path.append(phase_index)
            phase_index = (phase_index + 1) % len(phase_states)
        return tuple(path)

    def _apply_actions(self, actions: np.ndarray) -> int:
        assert self._traci is not None
        assert self._current_green_actions is not None
        now = self._traci.simulation.getTime()
        paths: dict[str, tuple[int, ...]] = {}
        requested_green_actions: dict[str, int] = {}
        for index, tls_id in enumerate(self._tls_ids):
            requested_action = int(actions[index])
            current_action = int(self._current_green_actions[index])
            elapsed_green = now - self._phase_started_at[tls_id]
            if requested_action == current_action or elapsed_green < self.min_green_seconds:
                continue
            paths[tls_id] = self._transition_phase_indices.get(
                (tls_id, current_action, requested_action), ()
            )
            requested_green_actions[tls_id] = requested_action

        transition_seconds = 0
        for stage in range(max((len(path) for path in paths.values()), default=0)):
            stage_duration = 0
            for tls_id, path in paths.items():
                if stage >= len(path):
                    continue
                phase_index = path[stage]
                self._traci.trafficlight.setPhase(tls_id, phase_index)
                stage_duration = max(stage_duration, self._phase_duration(tls_id, phase_index))
            self._advance(stage_duration)
            transition_seconds += stage_duration

        for tls_id, requested_action in requested_green_actions.items():
            self._set_green_phase(tls_id, requested_action)
            tls_index = self._tls_ids.index(tls_id)
            self._current_green_actions[tls_index] = requested_action
            self._phase_started_at[tls_id] = self._traci.simulation.getTime()
        return transition_seconds

    def _phase_duration(self, tls_id: str, phase_index: int) -> int:
        state = self._phase_states[tls_id][phase_index]
        return self.yellow_seconds if "y" in state else self.all_red_seconds

    def _set_green_phase(self, tls_id: str, action: int) -> None:
        assert self._traci is not None
        self._traci.trafficlight.setPhase(tls_id, self._green_phase_indices[tls_id][action])

    def _advance(self, seconds: int) -> None:
        assert self._traci is not None
        target_time = min(
            self._traci.simulation.getTime() + seconds,
            self.episode_duration_seconds,
        )
        while self._traci.simulation.getTime() < target_time:
            self._traci.simulationStep()

    def _observation(self) -> np.ndarray:
        assert self._traci is not None
        values = []
        for tls_id in self._tls_ids:
            lanes = self._lanes_by_tls[tls_id]
            queue_length = sum(
                self._traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in lanes
            )
            vehicle_count = sum(
                self._traci.lane.getLastStepVehicleNumber(lane_id) for lane_id in lanes
            )
            weighted_speed = sum(
                self._traci.lane.getLastStepMeanSpeed(lane_id)
                * self._traci.lane.getLastStepVehicleNumber(lane_id)
                for lane_id in lanes
            )
            mean_speed = weighted_speed / vehicle_count if vehicle_count else 0.0
            current_phase = self._traci.trafficlight.getPhase(tls_id)
            elapsed_phase = self._traci.simulation.getTime() - self._phase_started_at[tls_id]
            values.extend(
                [queue_length, vehicle_count, mean_speed, current_phase, elapsed_phase]
            )
        return np.asarray(values, dtype=np.float32)

    def _network_queue_length(self) -> int:
        assert self._traci is not None
        lane_ids = {
            lane_id for lanes in self._lanes_by_tls.values() for lane_id in lanes
        }
        return sum(
            self._traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in lane_ids
        )

    def decode_discrete_action(self, action: int) -> np.ndarray:
        if action < 0 or action >= self.discrete_action_count:
            raise ValueError(f"Discrete action must be in [0, {self.discrete_action_count}).")
        return np.asarray(self._action_vectors[action], dtype=np.int64)


class FlattenedActionEnv(gym.ActionWrapper[np.ndarray, int, np.ndarray]):
    def __init__(self, env: GironaTrafficEnv) -> None:
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(env.discrete_action_count)

    def action(self, action: int | np.ndarray) -> np.ndarray:
        return self.env.decode_discrete_action(int(action))
