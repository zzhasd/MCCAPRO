from __future__ import annotations

import numpy as np

from .grid import GridMap
from .metrics import path_distance, stc_contour_paths
from .planners import PlanResult
from .user_mst import compute_tiled_mst, tiled_mst_component_walks


def audit_path_closure(
    grid: GridMap,
    result: PlanResult,
    scenario: str,
    seed: int,
    path_style: str,
) -> list[dict[str, object]]:
    """Audit the exact path representation used by the common metrics pipeline."""

    rows = []
    for rid, component, path, path_mode in _result_path_segments(grid, result, path_style):
        first = path[0] if path else None
        last = path[-1] if path else None
        gap = (
            float(np.hypot(last[0] - first[0], last[1] - first[1]))
            if first is not None and last is not None
            else 0.0
        )
        rows.append(
            {
                "scenario": scenario,
                "seed": seed,
                "method": result.method,
                "robot": rid,
                "component": component,
                "path_mode": path_mode,
                "point_count": len(path),
                "path_length": path_distance(path),
                "first_x": first[0] if first is not None else np.nan,
                "first_y": first[1] if first is not None else np.nan,
                "last_x": last[0] if last is not None else np.nan,
                "last_y": last[1] if last is not None else np.nan,
                "closure_gap": gap,
                "explicitly_closed": bool(len(path) <= 1 or gap <= 1e-9),
            }
        )
    return rows


def _result_path_segments(
    grid: GridMap,
    result: PlanResult,
    path_style: str,
):
    robot_count = max(len(result.roots), max(result.assignments.values(), default=-1) + 1)
    if result.extra.get("mst_mode") == "script_tile_mst":
        tiled = compute_tiled_mst(
            grid,
            result.assignments,
            robot_count,
            result.extra.get("tile_layers"),
        )
        return [
            (rid, component, path, "mst_dfs_roundtrip_2x_mst")
            for rid in range(robot_count)
            for component, path in enumerate(tiled_mst_component_walks(tiled, rid))
        ]
    if result.paths is not None:
        return [
            (rid, 0, path, str(result.extra.get("path_mode", "planner_path")))
            for rid, path in enumerate(result.paths)
        ]
    if path_style == "stc":
        return [
            (rid, component, path, "stc_contour_path")
            for rid in range(robot_count)
            for component, path in enumerate(
                stc_contour_paths(
                    grid,
                    {cell for cell, owner in result.assignments.items() if owner == rid},
                    result.roots[rid] if rid < len(result.roots) else None,
                )
            )
        ]
    raise ValueError(f"Unsupported path representation: {result.method}")
