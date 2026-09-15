from __future__ import annotations

import unittest

import numpy as np

from mcpp_compare.grid import GridMap
from mcpp_compare.metrics import evaluate_partition, evaluate_paths, path_execution_time_s
from mcpp_compare.mcca_pro import MCCAPROPlanner
from mcpp_compare.path_audit import audit_path_closure
from mcpp_compare.planners import (
    AdaptiveVoronoiMSTPlanner,
    DARPAdaptiveMSTPlanner,
    DARPBoustrophedonPlanner,
    DARPPlanner,
    DARPThreeTileMSTPlanner,
    PlanResult,
    VoronoiThreeTileMSTPlanner,
)
from mcpp_compare.sensor_planners import SCoPPGridPlanner


class PlannerSmokeTest(unittest.TestCase):
    def test_execution_time_corrected_speed_and_counter_rotation(self) -> None:
        # Two metres and one 90-degree turn: 10 s straight + wheel arc / speed.
        corner = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
        expected = 10.0 + (np.pi / 2.0) * (0.193 / 2.0) / 0.2
        self.assertAlmostEqual(path_execution_time_s(corner), expected)
        # The correction scales both translation and rotation, including U-turns.
        for path in [[], [(0.0, 0.0), (2.0, 0.0)], corner,
                     [(0.0, 0.0), (1.0, 0.0), (0.0, 0.0)]]:
            old_time = path_execution_time_s(path, 0.4, 2.0 * 0.4 / 0.193)
            self.assertAlmostEqual(path_execution_time_s(path), 2.0 * old_time)

    def test_core_planners_cover_every_free_cell(self) -> None:
        grid = GridMap.random_with_filled_disconnected(10, 10, 0.08, seed=5)
        self.assertEqual(len(grid.connected_components(grid.free_cells)), 1)
        robot_count = 3
        weights = np.ones(robot_count) / robot_count
        planners = [
            MCCAPROPlanner(seed=5, partition_iterations=3, tiling_time_limit=0.1),
            AdaptiveVoronoiMSTPlanner(max_iter=3),
            VoronoiThreeTileMSTPlanner(max_iter=3),
            DARPAdaptiveMSTPlanner(max_iter=100),
            DARPThreeTileMSTPlanner(max_iter=100),
            DARPBoustrophedonPlanner(max_iter=100),
            DARPPlanner(max_iter=100),
            SCoPPGridPlanner(),
        ]
        for planner in planners:
            with self.subTest(planner=planner.name):
                result = planner.plan(grid, robot_count, weights)
                self.assertEqual(set(result.assignments), set(grid.free_cells))
                self.assertTrue(set(result.assignments.values()).issubset(set(range(robot_count))))
                partition = evaluate_partition(grid, result, weights, "smoke", seed=5)
                paths = evaluate_paths(grid, result, "smoke", seed=5)
                closure = audit_path_closure(grid, result, "smoke", seed=5, path_style="stc")
                self.assertTrue(closure)
                self.assertTrue(all(row["explicitly_closed"] for row in closure))
                self.assertGreaterEqual(partition.component_count, robot_count)
                self.assertTrue(np.isfinite(partition.geodesic_compactness))
                self.assertTrue(np.isfinite(partition.geodesic_silhouette))
                self.assertGreater(paths.path_length_sum, 0)
                self.assertGreater(paths.max_robot_path_length, 0)
                self.assertGreater(paths.max_robot_execution_time_s, 0)
                self.assertTrue(np.isfinite(paths.path_load_balance_cv))
                self.assertGreaterEqual(paths.turn_count_sum, 0)
                self.assertGreaterEqual(paths.max_robot_turn_count, 0)
                self.assertTrue(np.isfinite(paths.turn_load_balance_cv))
                self.assertGreater(paths.planning_runtime_ms, 0)
                if result.extra.get("mst_mode") == "script_tile_mst":
                    self.assertEqual(result.extra.get("mst_mode"), "script_tile_mst")
                    self.assertGreater(result.extra.get("tile_mst_length", 0), 0)
                    self.assertAlmostEqual(paths.path_length_sum, 2.0 * result.extra["tile_mst_length"])
                if planner.name == "Your-Voronoi-Adaptive-MST-3Tiles":
                    self.assertEqual(result.extra.get("tile_layers"), ((4, 4), (2, 2), (1, 1)))
                if planner.name == "Your-DARP-Adaptive-MST-3Tiles":
                    self.assertEqual(result.extra.get("partition_mode"), "darp_style")
                    self.assertEqual(result.extra.get("tile_layers"), ((4, 4), (2, 2), (1, 1)))
                if planner.name == "DARP-Boustrophedon-W4":
                    self.assertEqual(result.extra.get("partition_mode"), "darp_style")
                    self.assertEqual(result.extra.get("swath_width"), 4)
                    self.assertTrue(result.extra.get("closed_roundtrip"))
                    self.assertEqual(result.extra.get("path_mode"), "boustrophedon_swath_4_closed_roundtrip")
                    self.assertIsNotNone(result.paths)
                    for path in result.paths or []:
                        self.assertEqual(path[0], path[-1])
                if result.extra.get("projection_kind") == "monitoring_cells":
                    self.assertIsNotNone(result.paths)
                    self.assertTrue(result.extra.get("closed_roundtrip"))
                    for rid, path in enumerate(result.paths or []):
                        self.assertEqual(path[0], path[-1])
                        for x, y in path:
                            cell = (int(round(x)), int(round(y)))
                            self.assertEqual(result.assignments.get(cell), rid)
                if planner.name == "MCCA-PRO-v2.3.2":
                    self.assertEqual(result.extra.get("mainline_file"), "mainline_tile_first_v2_3_2.py")
                    self.assertEqual(result.extra.get("mainline_version"), "2.3.2")
                    self.assertEqual(result.extra.get("path_mode"), "mcca_pro_tile_first_tsp")
                    self.assertIsNotNone(result.paths)

    def test_execution_time_uses_each_robot_path_shape(self) -> None:
        grid = GridMap(11, 2, frozenset())
        straight_longer = [(0.0, 0.0), (10.0, 0.0)]
        turning_shorter = [
            (0.0, 1.0),
            (1.0, 1.0),
            (1.0, 0.0),
            (2.0, 0.0),
            (2.0, 1.0),
            (3.0, 1.0),
            (3.0, 0.0),
            (4.0, 0.0),
            (4.0, 1.0),
            (5.0, 1.0),
        ]
        result = PlanResult(
            method="shape-test",
            assignments={(0, 0): 0, (0, 1): 1},
            roots=[(0, 0), (0, 1)],
            runtime_s=0.001,
            paths=[straight_longer, turning_shorter],
        )

        paths = evaluate_paths(grid, result, "shape-test", seed=0)

        self.assertEqual(paths.max_robot_path_length, 10.0)
        self.assertGreater(path_execution_time_s(turning_shorter), path_execution_time_s(straight_longer))
        self.assertAlmostEqual(paths.max_robot_execution_time_s, path_execution_time_s(turning_shorter))

        # A narrower track can change which robot determines the maximum time.
        result.paths[0] = [(0.0, 0.0), (10.3, 0.0)]
        old_times = [path_execution_time_s(p, 0.2, 0.4 / 0.2314) for p in result.paths]
        self.assertGreater(old_times[1], old_times[0])
        corrected = evaluate_paths(grid, result, "track-width-test", seed=0)
        self.assertLess(path_execution_time_s(turning_shorter), 51.5)
        self.assertAlmostEqual(corrected.max_robot_execution_time_s, 51.5)


if __name__ == "__main__":
    unittest.main()
