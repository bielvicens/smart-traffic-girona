from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "reinforcement_learning"))

from sumo_env import FlattenedActionEnv, GironaTrafficEnv, is_actionable_phase


class TrafficLightPhaseTests(unittest.TestCase):
    def test_actionable_phase_excludes_yellow_transitions(self) -> None:
        self.assertTrue(is_actionable_phase("GGrr"))
        self.assertTrue(is_actionable_phase("rrGG"))
        self.assertFalse(is_actionable_phase("yyrr"))
        self.assertFalse(is_actionable_phase("Gyyyrryyyrr"))
        self.assertFalse(is_actionable_phase("rrrr"))


class SumoEnvironmentTests(unittest.TestCase):
    def test_reset_and_step_have_expected_spaces(self) -> None:
        environment = GironaTrafficEnv(episode_duration_seconds=20, route_seeds=[42])
        try:
            observation, info = environment.reset(seed=42, options={"route_seed": 42})
            self.assertEqual(observation.shape, (70,))
            self.assertEqual(info["traffic_lights"], 14)
            self.assertEqual(environment.action_dimensions, (1, 1, 1, 3, 2, 2, 2, 2, 2, 2, 1, 2, 2, 3))
            next_observation, reward, terminated, truncated, _ = environment.step(
                np.zeros(14, dtype=np.int64)
            )
            self.assertEqual(next_observation.shape, (70,))
            self.assertLessEqual(reward, 0.0)
            self.assertFalse(truncated)
            self.assertFalse(terminated)
        finally:
            environment.close()

    def test_flattened_action_wrapper_matches_action_product(self) -> None:
        environment = FlattenedActionEnv(
            GironaTrafficEnv(episode_duration_seconds=20, route_seeds=[42])
        )
        try:
            environment.reset(seed=42, options={"route_seed": 42})
            self.assertEqual(environment.action_space.n, 2304)
            self.assertEqual(environment.action(0).shape, (14,))
        finally:
            environment.close()


if __name__ == "__main__":
    unittest.main()
