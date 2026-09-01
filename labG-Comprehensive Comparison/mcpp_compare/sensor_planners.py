from __future__ import annotations

from time import perf_counter

import numpy as np

from .grid import Coord, GridMap, choose_roots, normalize_weights, target_counts
from .planners import PlanResult


Path = list[Coord]


def monitoring_cell_footprints(grid: GridMap) -> dict[Coord, frozenset[Coord]]:
    """SCoPP-style discretized cells: each task is one monitoring cell."""

    return {cell: frozenset({cell}) for cell in grid.free_cells}


class SCoPPGridPlanner:
    """Grid adaptation of SCoPP's quick load-balanced monitoring pipeline."""

    name = "SCoPP-QLB"

    def __init__(self, max_kmeans_iter: int = 12) -> None:
        self.max_kmeans_iter = max_kmeans_iter

    def plan(
        self,
        grid: GridMap,
        robot_count: int,
        weights: list[float] | np.ndarray,
        previous: dict[Coord, int] | None = None,
    ) -> PlanResult:
        started = perf_counter()
        roots = choose_roots(grid, robot_count)
        waypoints = list(monitoring_cell_footprints(grid))
        waypoint_assignment = _balanced_point_assignment(
            waypoints,
            roots,
            weights,
            robot_count,
            max_iter=self.max_kmeans_iter,
        )
        waypoint_assignment = _repair_connected_monitoring_assignments(
            grid,
            waypoint_assignment,
            roots,
            target_counts(weights, len(waypoints), robot_count),
        )
        waypoints_by_robot = [
            [point for point, rid in waypoint_assignment.items() if rid == robot]
            for robot in range(robot_count)
        ]
        paths = [
            _nearest_neighbor_route(
                grid,
                roots[robot],
                waypoints_by_robot[robot],
                allowed={cell for cell, rid in waypoint_assignment.items() if rid == robot},
            )
            for robot in range(robot_count)
        ]
        counts = [len(points) for points in waypoints_by_robot]
        return PlanResult(
            method=self.name,
            assignments=dict(waypoint_assignment),
            roots=roots,
            runtime_s=perf_counter() - started,
            notes=(
                "Clean-room grid adaptation of SCoPP/QLB: the common grid is treated as "
                "the already-discretized monitoring-cell set, then load-balanced and routed. "
                "Detached components are reassigned so paths stay inside each robot's region; "
                "each route then retraces its obstacle-safe path to close at its start."
            ),
            extra={
                "projection_kind": "monitoring_cells",
                "cell_model": "scopp_discretized_monitoring_cells",
                "path_mode": "scopp_qlb_monitoring_cells_closed_roundtrip",
                "closed_roundtrip": True,
                "runtime_source": "measured_in_tree_planner_core",
                "monitoring_waypoints": len(waypoints),
                "connectivity_repair": "detached_components_reassigned_to_adjacent_regions",
                "min_waypoints_per_robot": min(counts) if counts else 0,
                "max_waypoints_per_robot": max(counts) if counts else 0,
            },
            paths=[[(float(x), float(y)) for x, y in path] for path in paths],
        )


def _balanced_point_assignment(
    points: list[Coord],
    roots: list[Coord],
    weights: list[float] | np.ndarray,
    robot_count: int,
    max_iter: int,
) -> dict[Coord, int]:
    if not points:
        return {}
    normalized = normalize_weights(weights, robot_count)
    targets = target_counts(weights, len(points), robot_count)
    centers = np.array([[float(x), float(y)] for x, y in roots], dtype=float)
    assignments: dict[Coord, int] = {}
    for _ in range(max_iter):
        next_assignments: dict[Coord, int] = {}
        for point in points:
            scores = [
                float(np.linalg.norm(np.array(point, dtype=float) - centers[robot]) / np.sqrt(normalized[robot]))
                for robot in range(robot_count)
            ]
            next_assignments[point] = int(np.argmin(scores))
        next_assignments = _rebalance_point_counts(points, next_assignments, centers, targets, normalized)
        if next_assignments == assignments:
            break
        assignments = next_assignments
        for robot in range(robot_count):
            owned = [point for point, rid in assignments.items() if rid == robot]
            if owned:
                centers[robot] = np.mean(np.array(owned, dtype=float), axis=0)
    return assignments


def _rebalance_point_counts(
    points: list[Coord],
    assignments: dict[Coord, int],
    centers: np.ndarray,
    targets: list[int],
    weights: np.ndarray,
) -> dict[Coord, int]:
    result = dict(assignments)
    robot_count = len(targets)
    counts = [sum(1 for rid in result.values() if rid == robot) for robot in range(robot_count)]
    while True:
        over = [robot for robot in range(robot_count) if counts[robot] > targets[robot]]
        under = [robot for robot in range(robot_count) if counts[robot] < targets[robot]]
        if not over or not under:
            break
        moved = False
        for donor in sorted(over, key=lambda robot: counts[robot] - targets[robot], reverse=True):
            donor_points = [point for point in points if result.get(point) == donor]
            move_options: list[tuple[float, Coord, int]] = []
            for point in donor_points:
                donor_score = _point_score(point, centers[donor], weights[donor])
                recipient = min(under, key=lambda robot: _point_score(point, centers[robot], weights[robot]))
                penalty = _point_score(point, centers[recipient], weights[recipient]) - donor_score
                move_options.append((penalty, point, recipient))
            for _, point, recipient in sorted(move_options, key=lambda item: (item[0], item[1])):
                if counts[donor] <= targets[donor]:
                    break
                if counts[recipient] >= targets[recipient]:
                    continue
                result[point] = recipient
                counts[donor] -= 1
                counts[recipient] += 1
                moved = True
        if not moved:
            break
    return result


def _point_score(point: Coord, center: np.ndarray, weight: float) -> float:
    return float(np.linalg.norm(np.array(point, dtype=float) - center) / np.sqrt(max(weight, 1e-9)))


def _repair_connected_monitoring_assignments(
    grid: GridMap,
    assignments: dict[Coord, int],
    roots: list[Coord],
    targets: list[int],
    max_rounds: int = 20,
) -> dict[Coord, int]:
    robot_count = len(roots)
    result = dict(assignments)
    if set(result) != set(grid.free_cells):
        root_distances = [grid.bfs_distances(root) for root in roots]
        for cell in grid.free_cells:
            if cell not in result:
                result[cell] = min(range(robot_count), key=lambda rid: root_distances[rid].get(cell, 10**9))

    for rid, root in enumerate(roots):
        result[root] = rid

    for _ in range(max_rounds):
        changed = False
        counts = [sum(1 for assigned in result.values() if assigned == robot) for robot in range(robot_count)]
        for rid, root in enumerate(roots):
            cells = {cell for cell, assigned in result.items() if assigned == rid}
            components = grid.connected_components(cells)
            if len(components) <= 1:
                continue
            main = next((component for component in components if root in component), components[0])
            for component in components:
                if component is main:
                    continue
                recipient_scores: dict[int, int] = {}
                for cell in component:
                    for nbr in grid.neighbors(cell):
                        nbr_rid = result.get(nbr)
                        if nbr_rid is None or nbr_rid == rid:
                            continue
                        recipient_scores[nbr_rid] = recipient_scores.get(nbr_rid, 0) + 1
                if not recipient_scores:
                    continue
                recipient = min(
                    recipient_scores,
                    key=lambda robot: (
                        counts[robot] / max(targets[robot], 1),
                        -recipient_scores[robot],
                        robot,
                    ),
                )
                for cell in component:
                    result[cell] = recipient
                counts[rid] -= len(component)
                counts[recipient] += len(component)
                changed = True
        if not changed:
            break
    return result


def _nearest_neighbor_route(
    grid: GridMap,
    root: Coord,
    waypoints: list[Coord],
    allowed: set[Coord] | None = None,
) -> Path:
    allowed_cells = set(allowed) if allowed is not None else set(grid.free_cells)
    if root not in allowed_cells:
        allowed_cells.add(root)
    route = [root]
    current = root
    remaining = set(waypoints)
    remaining.discard(root)
    while remaining:
        distances = grid.bfs_distances(current, allowed_cells)
        nxt = min(remaining, key=lambda cell: (distances.get(cell, 10**9), cell))
        connector = grid.shortest_path(current, nxt, allowed_cells)
        route.extend(connector[1:] if connector else [nxt])
        current = nxt
        remaining.remove(nxt)
    if len(route) > 1:
        route.extend(reversed(route[:-1]))
    return route
