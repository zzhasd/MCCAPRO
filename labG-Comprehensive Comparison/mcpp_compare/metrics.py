from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict, deque
from statistics import mean
from time import perf_counter

import numpy as np

from .grid import Coord, GridMap, normalize_weights
from .planners import PlanResult
from .user_mst import Point, TileLayer, TiledMST, compute_tiled_mst, tiled_mst_component_walks

ROBOT_MAX_SPEED_M_S = 0.4
ROBOT_TRACK_WIDTH_M = 0.2314
ROBOT_WHEELBASE_M = 0.2844
ROBOT_COUNTER_ROTATION_MAX_ANGULAR_SPEED_RAD_S = 2.0 * ROBOT_MAX_SPEED_M_S / ROBOT_TRACK_WIDTH_M


@dataclass(frozen=True)
class PartitionMetrics:
    method: str
    scenario: str
    seed: int
    area_error_l1: float
    max_area_error: float
    load_balance_cv: float
    geodesic_compactness: float
    geodesic_silhouette: float
    disconnected_robots: int
    component_count: int
    runtime_ms: float


@dataclass(frozen=True)
class PathMetrics:
    method: str
    scenario: str
    seed: int
    path_length_sum: float
    max_robot_path_length: float
    max_robot_execution_time_s: float
    path_load_balance_cv: float
    turn_count_sum: float
    max_robot_turn_count: float
    turn_load_balance_cv: float
    planning_runtime_ms: float
    planning_runtime_source: str


def evaluate_partition(
    grid: GridMap,
    result: PlanResult,
    weights: list[float] | np.ndarray,
    scenario: str,
    seed: int,
    include_geodesic: bool = True,
) -> PartitionMetrics:
    robot_count = len(weights)
    normalized = normalize_weights(weights, robot_count)
    counts = np.array([sum(1 for rid in result.assignments.values() if rid == robot) for robot in range(robot_count)], dtype=float)
    ratios = counts / max(grid.free_count, 1)
    area_errors = np.abs(ratios - normalized)
    components = [
        grid.connected_components([cell for cell, rid in result.assignments.items() if rid == robot])
        for robot in range(robot_count)
    ]
    component_count = sum(len(items) for items in components)
    disconnected = sum(1 for items in components if len(items) > 1)
    load_balance_cv = float(np.std(counts / np.maximum(normalized, 1e-9)) / max(np.mean(counts / np.maximum(normalized, 1e-9)), 1e-9))
    geodesic_compactness, geodesic_silhouette = (
        geodesic_cluster_scores(grid, result.assignments, robot_count)
        if include_geodesic
        else (float("nan"), float("nan"))
    )
    return PartitionMetrics(
        method=result.method,
        scenario=scenario,
        seed=seed,
        area_error_l1=float(area_errors.sum()),
        max_area_error=float(area_errors.max()),
        load_balance_cv=load_balance_cv,
        geodesic_compactness=geodesic_compactness,
        geodesic_silhouette=geodesic_silhouette,
        disconnected_robots=disconnected,
        component_count=component_count,
        runtime_ms=result.runtime_s * 1000,
    )


def evaluate_paths(
    grid: GridMap,
    result: PlanResult,
    scenario: str,
    seed: int,
    path_style: str = "dfs",
) -> PathMetrics:
    robot_count = max(len(result.roots), max(result.assignments.values(), default=-1) + 1)
    path_lengths: list[float] = []
    turn_counts: list[float] = []
    execution_times: list[float] = []
    path_generation_runtime_s = 0.0
    tiled_mst = (
        compute_tiled_mst(grid, result.assignments, robot_count, _tile_layers_for_result(result))
        if result.extra.get("mst_mode") == "script_tile_mst"
        else None
    )
    if tiled_mst is not None:
        for robot in range(robot_count):
            mst_length = float(tiled_mst.lengths_by_robot.get(robot, 0.0))
            path_lengths.append(2.0 * mst_length)
            path_started = perf_counter()
            walks = tiled_mst_component_walks(tiled_mst, robot)
            path_generation_runtime_s += perf_counter() - path_started
            turn_counts.append(float(sum(count_turns(walk) for walk in walks)))
            execution_times.append(float(sum(path_execution_time_s(walk) for walk in walks)))
        return _path_metrics_from_lengths(
            result,
            scenario,
            seed,
            path_lengths,
            turn_counts,
            execution_times,
            path_generation_runtime_s,
        )

    for robot in range(robot_count):
        cells = {cell for cell, rid in result.assignments.items() if rid == robot}
        if result.paths is not None and robot < len(result.paths):
            path = result.paths[robot]
        elif path_style == "stc":
            path_started = perf_counter()
            paths = stc_contour_paths(grid, cells, result.roots[robot] if robot < len(result.roots) else None)
            path_generation_runtime_s += perf_counter() - path_started
            path_lengths.append(float(sum(path_distance(path) for path in paths)))
            turn_counts.append(float(sum(count_turns(path) for path in paths)))
            execution_times.append(float(sum(path_execution_time_s(path) for path in paths)))
            continue
        else:
            root = result.roots[robot] if robot < len(result.roots) else None
            path_started = perf_counter()
            path = dfs_coverage_path(grid, cells, root)
            path_generation_runtime_s += perf_counter() - path_started
        path_lengths.append(float(path_distance(path) if result.paths is not None else max(len(path) - 1, 0)))
        turn_counts.append(float(count_turns(path)))
        execution_times.append(path_execution_time_s(path))
    return _path_metrics_from_lengths(
        result,
        scenario,
        seed,
        path_lengths,
        turn_counts,
        execution_times,
        path_generation_runtime_s,
    )


def _path_metrics_from_lengths(
    result: PlanResult,
    scenario: str,
    seed: int,
    path_lengths: list[float],
    turn_counts: list[float],
    execution_times: list[float],
    path_generation_runtime_s: float = 0.0,
) -> PathMetrics:
    lengths = np.array(path_lengths, dtype=float)
    turns = np.array(turn_counts, dtype=float)
    times = np.array(execution_times, dtype=float)
    mean_length = float(np.mean(lengths)) if len(lengths) else 0.0
    mean_turns = float(np.mean(turns)) if len(turns) else 0.0
    load_balance_cv = float(np.std(lengths) / mean_length) if mean_length > 1e-9 else 0.0
    turn_load_balance_cv = float(np.std(turns) / mean_turns) if mean_turns > 1e-9 else 0.0
    runtime_source = str(result.extra.get("runtime_source", "planner_runtime_s"))
    if path_generation_runtime_s > 1e-12:
        runtime_source = f"{runtime_source}+generated_final_path"
    return PathMetrics(
        method=result.method,
        scenario=scenario,
        seed=seed,
        path_length_sum=float(np.sum(lengths)),
        max_robot_path_length=float(np.max(lengths) if len(lengths) else 0.0),
        max_robot_execution_time_s=float(np.max(times) if len(times) else 0.0),
        path_load_balance_cv=load_balance_cv,
        turn_count_sum=float(np.sum(turns)),
        max_robot_turn_count=float(np.max(turns) if len(turns) else 0.0),
        turn_load_balance_cv=turn_load_balance_cv,
        planning_runtime_ms=float((result.runtime_s + path_generation_runtime_s) * 1000),
        planning_runtime_source=runtime_source,
    )


def tiled_mst_turns(tiled_mst: TiledMST, robot: int) -> int:
    """Count angle changes on the DFS round trip of the robot's tile MST forest."""

    return sum(count_turns(walk) for walk in tiled_mst_component_walks(tiled_mst, robot))


def _tile_layers_for_result(result: PlanResult) -> tuple[TileLayer, ...] | None:
    value = result.extra.get("tile_layers")
    if value is None:
        return None
    return tuple((int(width), int(height)) for width, height in value)  # type: ignore[union-attr]


def geodesic_cluster_scores(
    grid: GridMap,
    assignments: dict[Coord, int],
    robot_count: int,
) -> tuple[float, float]:
    """Return intra-cluster geodesic compactness and geodesic silhouette."""

    cells_by_robot = [
        sorted(cell for cell, rid in assignments.items() if rid == robot)
        for robot in range(robot_count)
    ]
    penalty = float(max(grid.width * grid.height, grid.free_count, 1))
    same_cluster_mean: dict[Coord, float] = {}
    compactness_total = 0.0
    compactness_pairs = 0

    for cells in cells_by_robot:
        if not cells:
            continue
        if len(cells) == 1:
            same_cluster_mean[cells[0]] = 0.0
            continue
        allowed = set(cells)
        for source in cells:
            distances = grid.bfs_distances(source, allowed)
            total = 0.0
            count = 0
            for target in cells:
                if target == source:
                    continue
                total += float(distances.get(target, penalty))
                count += 1
            same_cluster_mean[source] = total / count if count else 0.0
            compactness_total += total
            compactness_pairs += count

    compactness = compactness_total / compactness_pairs if compactness_pairs else 0.0
    silhouettes: list[float] = []
    for rid, cells in enumerate(cells_by_robot):
        if not cells:
            continue
        own_cells = set(cells)
        other_clusters = [
            set(other_cells)
            for other_rid, other_cells in enumerate(cells_by_robot)
            if other_rid != rid and other_cells
        ]
        for source in cells:
            a_value = same_cluster_mean.get(source, 0.0)
            b_value = float("inf")
            for other_cells in other_clusters:
                allowed = own_cells | other_cells
                distances = grid.bfs_distances(source, allowed)
                total = sum(float(distances.get(target, penalty)) for target in other_cells)
                b_value = min(b_value, total / len(other_cells))
            if not np.isfinite(b_value):
                silhouettes.append(0.0)
                continue
            denominator = max(a_value, b_value)
            silhouettes.append(0.0 if denominator == 0 else (b_value - a_value) / denominator)

    silhouette = float(np.mean(silhouettes)) if silhouettes else 0.0
    return float(compactness), silhouette


def dfs_coverage_path(
    grid: GridMap,
    cells: set[Coord],
    root: Coord | None,
) -> list[Coord]:
    if not cells:
        return []
    components = grid.connected_components(cells)
    current = root if root in cells else min(cells)
    path: list[Coord] = []
    for component in components:
        start = current if current in component else min(component, key=lambda cell: manhattan(current, cell))
        if path and path[-1] != start:
            connector = grid.shortest_path(path[-1], start, cells)
            if len(connector) <= 1:
                connector = grid.shortest_path(path[-1], start)
            path.extend(connector[1:] if connector else [start])
        elif not path:
            path.append(start)
        tree = grid.spanning_tree_adjacency(component, start)
        visited = {start}
        _dfs_append(tree, start, visited, path)
        current = path[-1]
    return path


def stc_contour_paths(
    grid: GridMap,
    cells: set[Coord],
    root: Coord | None = None,
) -> list[list[Point]]:
    """Generate STC-style contour paths around spanning trees of grid cells."""

    if not cells:
        return []
    paths: list[list[Point]] = []
    current = root if root in cells else min(cells)
    for component in grid.connected_components(cells):
        start = current if current in component else min(component, key=lambda cell: manhattan(current, cell))
        tree = grid.spanning_tree_adjacency(component, start)
        paths.append(_stc_contour_from_tree(tree, start))
        current = start
    return paths


def _stc_contour_from_tree(tree: dict[Coord, set[Coord]], start: Coord) -> list[Point]:
    if len(tree) == 1:
        x, y = start
        return [
            (x + 0.75, y + 0.25),
            (x + 0.75, y + 0.75),
            (x + 0.25, y + 0.75),
            (x + 0.25, y + 0.25),
            (x + 0.75, y + 0.25),
        ]

    dummy_parent = min(tree[start])
    route = _spiral_route(start, dummy_parent, tree)
    trajectory: list[Point] = []
    last = dummy_parent
    for idx, current in enumerate(route):
        motion = _stc_motion_coords(last, current)
        if idx <= len(route) - 2 and last == route[idx + 1]:
            motion += _stc_round_trip_coords(last, current)
        trajectory.extend(motion)
        last = current

    idx = 0
    while idx < len(trajectory) - 1:
        dx = trajectory[idx + 1][0] - trajectory[idx][0]
        dy = trajectory[idx + 1][1] - trajectory[idx][1]
        if abs(dx) + abs(dy) == 0:
            trajectory.pop(idx)
        elif abs(dx) + abs(dy) == 1:
            if dx * dy > 0:
                trajectory.insert(idx + 1, (trajectory[idx + 1][0], trajectory[idx][1]))
            else:
                trajectory.insert(idx + 1, (trajectory[idx][0], trajectory[idx + 1][1]))
            idx += 2
        else:
            idx += 1
    return trajectory[:-1] if len(trajectory) > 1 else trajectory


def _spiral_route(start: Coord, dummy_parent: Coord, tree: dict[Coord, set[Coord]]) -> list[Coord]:
    route: list[Coord] = []
    visited_nodes = {start}
    visited_edges: dict[Coord, list[Coord]] = defaultdict(list)

    def ccw_traverse(node: Coord, parent: Coord, is_backtracking: bool) -> None:
        route.append(node)
        visited_nodes.add(node)
        if not is_backtracking and (parent, node) != (dummy_parent, start):
            visited_edges[parent].append(node)

        neighbors = deque(_ccw_neighbors(tree, node))
        neighbors.rotate(1 - _motion_dir(parent, node))
        for neighbor in neighbors:
            if neighbor is not None and neighbor not in visited_nodes:
                ccw_traverse(neighbor, node, False)

        for child in list(visited_edges[parent]):
            visited_edges[parent].remove(child)
            ccw_traverse(parent, child, True)

    ccw_traverse(start, dummy_parent, False)
    return route


def _ccw_neighbors(tree: dict[Coord, set[Coord]], node: Coord) -> list[Coord | None]:
    ordered: list[Coord | None] = [None] * 4
    for neighbor in tree[node]:
        ordered[_motion_dir(node, neighbor)] = neighbor
    return ordered


def _motion_dir(first: Coord, second: Coord) -> int:
    if second[1] < first[1]:
        return 0
    if second[0] > first[0]:
        return 1
    if second[1] > first[1]:
        return 2
    if second[0] < first[0]:
        return 3
    raise ValueError(f"identical STC nodes do not define a direction: {first}")


def _subnode(cell: Coord, direction: str) -> Point:
    x, y = cell
    if direction == "SE":
        return (x + 0.75, y + 0.25)
    if direction == "SW":
        return (x + 0.25, y + 0.25)
    if direction == "NE":
        return (x + 0.75, y + 0.75)
    if direction == "NW":
        return (x + 0.25, y + 0.75)
    raise ValueError(f"unknown subnode direction: {direction}")


def _stc_motion_coords(first: Coord, second: Coord) -> list[Point]:
    direction = _motion_dir(first, second)
    if direction == 1:
        return [_subnode(first, "SE"), _subnode(second, "SW")]
    if direction == 3:
        return [_subnode(first, "NW"), _subnode(second, "NE")]
    if direction == 0:
        return [_subnode(first, "SW"), _subnode(second, "NW")]
    if direction == 2:
        return [_subnode(first, "NE"), _subnode(second, "SE")]
    raise ValueError("unreachable STC direction")


def _stc_round_trip_coords(last: Coord, pivot: Coord) -> list[Point]:
    direction = _motion_dir(last, pivot)
    if direction == 1:
        return [_subnode(pivot, "SE"), _subnode(pivot, "NE")]
    if direction == 0:
        return [_subnode(pivot, "SW"), _subnode(pivot, "SE")]
    if direction == 3:
        return [_subnode(pivot, "NW"), _subnode(pivot, "SW")]
    if direction == 2:
        return [_subnode(pivot, "NE"), _subnode(pivot, "NW")]
    raise ValueError("unreachable STC direction")


def count_turns(path: list[tuple[float, float]]) -> int:
    if len(path) < 3:
        return 0
    directions: list[tuple[float, float]] = []
    for first, second in zip(path, path[1:]):
        dx = second[0] - first[0]
        dy = second[1] - first[1]
        length = float(np.hypot(dx, dy))
        if length > 1e-9:
            directions.append((dx / length, dy / length))
    return sum(
        1
        for first, second in zip(directions, directions[1:])
        if abs(first[0] * second[1] - first[1] * second[0]) > 1e-8
        or first[0] * second[0] + first[1] * second[1] < 1.0 - 1e-8
    )


def path_turn_angle(path: list[tuple[float, float]]) -> float:
    """Return the total absolute heading change along a path in radians."""

    if len(path) < 3:
        return 0.0
    directions: list[tuple[float, float]] = []
    for first, second in zip(path, path[1:]):
        dx = second[0] - first[0]
        dy = second[1] - first[1]
        length = float(np.hypot(dx, dy))
        if length > 1e-9:
            directions.append((dx / length, dy / length))
    total = 0.0
    for first, second in zip(directions, directions[1:]):
        cross = first[0] * second[1] - first[1] * second[0]
        dot = first[0] * second[0] + first[1] * second[1]
        total += abs(float(np.arctan2(cross, dot)))
    return total


def path_execution_time_s(
    path: list[tuple[float, float]],
    max_speed_m_s: float = ROBOT_MAX_SPEED_M_S,
    max_angular_speed_rad_s: float = ROBOT_COUNTER_ROTATION_MAX_ANGULAR_SPEED_RAD_S,
) -> float:
    if max_speed_m_s <= 0.0:
        raise ValueError("max_speed_m_s must be positive")
    if max_angular_speed_rad_s <= 0.0:
        raise ValueError("max_angular_speed_rad_s must be positive")
    return path_distance(path) / max_speed_m_s + path_turn_angle(path) / max_angular_speed_rad_s


def path_distance(path: list[tuple[float, float]]) -> float:
    if len(path) < 2:
        return 0.0
    return float(
        sum(
            float(np.hypot(second[0] - first[0], second[1] - first[1]))
            for first, second in zip(path, path[1:])
        )
    )


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    return {"mean": float(mean(values)), "std": float(np.std(values))}


def _dfs_append(
    tree: dict[Coord, set[Coord]],
    cell: Coord,
    visited: set[Coord],
    path: list[Coord],
) -> None:
    for nxt in sorted(tree[cell]):
        if nxt in visited:
            continue
        visited.add(nxt)
        path.append(nxt)
        _dfs_append(tree, nxt, visited, path)
        path.append(cell)


def manhattan(first: Coord, second: Coord) -> int:
    return abs(first[0] - second[0]) + abs(first[1] - second[1])
