from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .grid import Coord, GridMap
from .planners import PlanResult


MAINLINE_FILENAME = "mainline_tile_first_v2_3_2.py"


@lru_cache(maxsize=1)
def _load_mainline_module() -> Any:
    """Load the repository-level MCCA-PRO implementation without copying it."""

    mainline_path = Path(__file__).resolve().parents[2] / MAINLINE_FILENAME
    if not mainline_path.is_file():
        raise FileNotFoundError(f"MCCA-PRO mainline not found: {mainline_path}")

    module_name = "_mcca_pro_mainline_tile_first_v2_3_2"
    spec = importlib.util.spec_from_file_location(module_name, mainline_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load MCCA-PRO mainline: {mainline_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class MCCAPROPlanner:
    """Thin adapter around ``mainline_tile_first_v2_3_2.TileFirstMCPP``."""

    name = "MCCA-PRO-v2.3.2"

    def __init__(
        self,
        seed: int,
        partition_iterations: int = 60,
        tiling_time_limit: float = 6.0,
        candidate_limit: int = 8,
    ) -> None:
        self.seed = int(seed)
        self.partition_iterations = int(partition_iterations)
        self.tiling_time_limit = float(tiling_time_limit)
        self.candidate_limit = int(candidate_limit)

    def plan(
        self,
        grid: GridMap,
        robot_count: int,
        weights: list[float] | np.ndarray,
        previous: dict[Coord, int] | None = None,
    ) -> PlanResult:
        del previous
        module = _load_mainline_module()
        obstacle_mask = np.zeros((grid.width, grid.height), dtype=bool)
        for x, y in grid.obstacles:
            obstacle_mask[x, y] = True

        planner = module.TileFirstMCPP(
            map_shape=(grid.width, grid.height),
            robot_num=robot_count,
            obstacle_mask=obstacle_mask,
            seed=self.seed,
            partition_iterations=self.partition_iterations,
            tiling_time_limit=self.tiling_time_limit,
            candidate_limit=self.candidate_limit,
            robot_weights=weights,
        )
        raw_result = planner.solve(method="tile_first")

        assignments = {
            (x, y): int(planner.assignments[x, y])
            for x, y in grid.free_cells
        }
        roots = [
            _root_for_region(assignments, rid, planner.centroids_xy[rid])
            for rid in range(robot_count)
        ]
        paths = [
            [tuple(map(float, point)) for point in planner.routes[rid]["path"]]
            for rid in range(robot_count)
        ]
        return PlanResult(
            method=self.name,
            assignments=assignments,
            roots=roots,
            runtime_s=float(raw_result["core_time"]),
            notes=(
                "Direct adapter to mainline_tile_first_v2_3_2.py: global tile-first "
                "coverage, connected tile partitioning, and depot-free obstacle-safe TSP routes."
            ),
            extra={
                "path_mode": "mcca_pro_tile_first_tsp",
                "path_coordinate_offset": 0.0,
                "runtime_source": "mainline_core_time_tiling_partition_routing",
                "mainline_file": MAINLINE_FILENAME,
                "mainline_version": str(module.__version__),
                "tile_count": int(raw_result["tile_count"]),
                "accepted_tile_transfers": int(raw_result["accepted_tile_transfers"]),
                "validation_time_s": float(raw_result["validation_time"]),
            },
            paths=paths,
        )


def _root_for_region(
    assignments: dict[Coord, int],
    rid: int,
    centroid: Coord | None,
) -> Coord:
    cells = [cell for cell, owner in assignments.items() if owner == rid]
    if not cells:
        raise ValueError(f"MCCA-PRO returned an empty region for robot {rid}")
    if centroid in cells:
        return centroid
    target = centroid if centroid is not None else cells[0]
    return min(cells, key=lambda cell: ((cell[0] - target[0]) ** 2 + (cell[1] - target[1]) ** 2, cell))
