from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "simulation"))

from scenarios import available_scenarios, load_scenario


class ScenarioConfigurationTests(unittest.TestCase):
    def test_expected_scenarios_are_available(self) -> None:
        self.assertEqual(
            available_scenarios(), ["incident", "low", "normal", "peak", "variable"]
        )

    def test_demand_blocks_cover_each_scenario_duration(self) -> None:
        for name in available_scenarios():
            scenario = load_scenario(name)
            self.assertEqual(scenario.demand_blocks[0].begin_seconds, 0)
            self.assertEqual(scenario.demand_blocks[-1].end_seconds, scenario.duration_seconds)

    def test_incident_configuration_is_valid(self) -> None:
        incident = load_scenario("incident").incident
        self.assertIsNotNone(incident)
        assert incident is not None
        self.assertLess(incident.begin_seconds, incident.end_seconds)
        self.assertTrue(incident.notification_edges)


if __name__ == "__main__":
    unittest.main()
