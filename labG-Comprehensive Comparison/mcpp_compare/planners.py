from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from time import perf_counter
from typing import Protocol

import numpy as np

from .grid import (
    Coord,
    GridMap,
    choose_roots,
    normalize_weights,
    target_counts,
    weighted_multi_source_assignment,
)
from .user_mst import THREE_TILE_LAYERS, TileLayer, compute_tiled_mst, normalize_tile_layers


@dataclass
class PlanResult:
    method: str
    assignments: dict[Coord, int]
    roots: list[Coord]
    runtime_s: float
    notes: str = ""
    extra: dict[str, object] = field(default_factory=dict)
    paths: list[list[tuple[float, float]]] | None = None


class Planner(Protocol):
    name: str

    def plan(
        self,
        grid: GridMap,
        robot_count: int,
        weights: list[float] | np.ndarray,
        previous: dict[Coord, int] | None = None,
    ) -> PlanResult:
        ...


class AdaptiveVoronoiMSTPlanner:
    """User method: weighted geodesic Voronoi partition plus adaptive MST proxy."""

    name = "Your-Voronoi-Adaptive-MST"

    def __init__(
        self,
        max_iter: int = 8,
        smoothing: float | None = None,
        tile_layers: Sequence[TileLayer] | None = None,
    ) -> None:
        self.max_iter = max_iter
        self.smoothing = smoothing
        self.tile_layers = normalize_tile_layers(tile_layers)
        self._ema_weights: np.ndarray | None = None

    def plan(
        self,
        grid: GridMap,
        robot_count: int,
        weights: list[float] | np.ndarray,
        previous: dict[Coord, int] | None = None,
    ) -> PlanResult:
        started = perf_counter()
        target = normalize_weights(weights, robot_count)
        if self.smoothing is not None:
            if self._ema_weights is None:
                self._ema_weights = np.ones(robot_count) / robot_count
            self._ema_weights = self.smoothing * target + (1 - self.smoothing) * self._ema_weights
            self._ema_weights = self._ema_weights / self._ema_weights.sum()
            target = self._ema_weights

        centroids = choose_roots(grid, robot_count)
        assignments: dict[Coord, int] = {}
        free = grid.free_cells
        free_set = set(free)

        for _ in range(self.max_iter):
            distances = [grid.bfs_distances(root, free_set) for root in centroids]
            next_assignments: dict[Coord, int] = {}
            for cell in free:
                scores = [
                    distances[rid].get(cell, 10**9) / np.sqrt(target[rid])
                    for rid in range(robot_count)
                ]
                next_assignments[cell] = int(np.argmin(scores))

            next_centroids: list[Coord] = []
            for rid in range(robot_count):
                region = [cell for cell, assigned in next_assignments.items() if assigned == rid]
                if not region:
                    next_centroids.append(centroids[rid])
                    continue
                mean_x = float(np.mean([cell[0] for cell in region]))
                mean_y = float(np.mean([cell[1] for cell in region]))
                best = min(region, key=lambda cell: (cell[0] - mean_x) ** 2 + (cell[1] - mean_y) ** 2)
                next_centroids.append(best)

            assignments = next_assignments
            if next_centroids == centroids:
                break
            centroids = next_centroids

        tiled_mst = compute_tiled_mst(grid, assignments, robot_count, self.tile_layers)
        return PlanResult(
            method=self.name,
            assignments=assignments,
            roots=centroids,
            runtime_s=perf_counter() - started,
            notes=(
                "Weighted geodesic Voronoi with adaptive rectangular tiling "
                f"({_format_tile_layers(self.tile_layers)}) + adjacent-tile MST metric."
            ),
            extra={
                "mst_mode": "script_tile_mst",
                "tile_mst_length": tiled_mst.total_length,
                "tile_count": tiled_mst.tile_count,
                "tile_layers": self.tile_layers,
                "tile_layers_label": _format_tile_layers(self.tile_layers),
                "runtime_source": "measured_in_tree_planner_core",
            },
        )


class VoronoiThreeTileMSTPlanner(AdaptiveVoronoiMSTPlanner):
    """User method with the original Voronoi partition and only 4x4/2x2/1x1 tiles."""

    name = "Your-Voronoi-Adaptive-MST-3Tiles"

    def __init__(self, max_iter: int = 8, smoothing: float | None = None) -> None:
        super().__init__(max_iter=max_iter, smoothing=smoothing, tile_layers=THREE_TILE_LAYERS)


class DARPAdaptiveMSTPlanner:
    """DARP-style partition followed by the user's adaptive tiling and tile-MST route proxy."""

    name = "Your-DARP-Adaptive-MST"

    def __init__(self, max_iter: int = 500, tile_layers: Sequence[TileLayer] | None = None) -> None:
        self.max_iter = max_iter
        self.tile_layers = normalize_tile_layers(tile_layers)

    def plan(
        self,
        grid: GridMap,
        robot_count: int,
        weights: list[float] | np.ndarray,
        previous: dict[Coord, int] | None = None,
    ) -> PlanResult:
        started = perf_counter()
        roots, assignments = _darp_partition(grid, robot_count, weights, self.max_iter)
        tiled_mst = compute_tiled_mst(grid, assignments, robot_count, self.tile_layers)
        return PlanResult(
            method=self.name,
            assignments=assignments,
            roots=roots,
            runtime_s=perf_counter() - started,
            notes=(
                "DARP-style partition with adaptive rectangular tiling "
                f"({_format_tile_layers(self.tile_layers)}) + adjacent-tile MST metric."
            ),
            extra={
                "mst_mode": "script_tile_mst",
                "partition_mode": "darp_style",
                "tile_mst_length": tiled_mst.total_length,
                "tile_count": tiled_mst.tile_count,
                "tile_layers": self.tile_layers,
                "tile_layers_label": _format_tile_layers(self.tile_layers),
                "runtime_source": "measured_in_tree_planner_core",
            },
        )


class DARPThreeTileMSTPlanner(DARPAdaptiveMSTPlanner):
    """DARP-style partition followed by 4x4/2x2/1x1 tiling and tile-MST route proxy."""

    name = "Your-DARP-Adaptive-MST-3Tiles"

    def __init__(self, max_iter: int = 500) -> None:
        super().__init__(max_iter=max_iter, tile_layers=THREE_TILE_LAYERS)


class DARPBoustrophedonPlanner:
    """DARP-style partition followed by swath-width boustrophedon coverage."""

    name = "DARP-Boustrophedon-W4"

    def __init__(self, max_iter: int = 500, swath_width: int = 4) -> None:
        if swath_width <= 0:
            raise ValueError("swath_width must be positive")
        self.max_iter = max_iter
        self.swath_width = swath_width

    def plan(
        self,
        grid: GridMap,
        robot_count: int,
        weights: list[float] | np.ndarray,
        previous: dict[Coord, int] | None = None,
    ) -> PlanResult:
        started = perf_counter()
        roots, assignments = _darp_partition(grid, robot_count, weights, self.max_iter)
        paths = [
            _boustrophedon_swath_path(
                grid,
                {cell for cell, assigned in assignments.items() if assigned == robot},
                roots[robot] if robot < len(roots) else None,
                self.swath_width,
            )
            for robot in range(robot_count)
        ]
        return PlanResult(
            method=self.name,
            assignments=assignments,
            roots=roots,
            runtime_s=perf_counter() - started,
            notes=(
                "DARP-style partition with boustrophedon swath coverage followed by "
                f"an exact reverse-path return to the start, width={self.swath_width}."
            ),
            extra={
                "partition_mode": "darp_style",
                "path_mode": f"boustrophedon_swath_{self.swath_width}_closed_roundtrip",
                "swath_width": self.swath_width,
                "closed_roundtrip": True,
                "runtime_source": "measured_in_tree_planner_core",
            },
            paths=paths,
        )


class DARPPlanner:
    """DARP-style equal-area partitioning with connectivity-preserving border moves."""

    name = "DARP"

    def __init__(self, max_iter: int = 500) -> None:
        self.max_iter = max_iter

    def plan(
        self,
        grid: GridMap,
        robot_count: int,
        weights: list[float] | np.ndarray,
        previous: dict[Coord, int] | None = None,
    ) -> PlanResult:
        started = perf_counter()
        roots, assignments = _darp_partition(grid, robot_count, weights, self.max_iter)
        return PlanResult(
            self.name,
            assignments,
            roots,
            perf_counter() - started,
            extra={"runtime_source": "measured_in_tree_partition_core"},
        )


def _darp_partition(
    grid: GridMap,
    robot_count: int,
    weights: list[float] | np.ndarray,
    max_iter: int,
) -> tuple[list[Coord], dict[Coord, int]]:
    roots = choose_roots(grid, robot_count)
    full_adj = {cell: set(grid.neighbors(cell)) for cell in grid.free_cells}
    assignments = weighted_multi_source_assignment(full_adj, roots, weights)
    assignments = rebalance_connected(grid, assignments, target_counts(weights, grid.free_count, robot_count), max_iter)
    return roots, assignments


def _format_tile_layers(tile_layers: Sequence[TileLayer]) -> str:
    return ", ".join(f"{width}x{height}" for width, height in tile_layers)


def _boustrophedon_swath_path(
    grid: GridMap,
    cells: set[Coord],
    root: Coord | None,
    swath_width: int,
) -> list[tuple[float, float]]:
    if not cells:
        return []

    route: list[Coord] = []
    current = root if root in cells else min(cells)
    for component in grid.connected_components(cells):
        start = current if current in component else min(component, key=lambda cell: _manhattan(current, cell))
        waypoints = _boustrophedon_waypoints(component, swath_width)
        if not waypoints:
            continue
        if route and route[-1] != start:
            _append_connector(route, grid.shortest_path(route[-1], start))
        if start != waypoints[0]:
            _append_connector(route, grid.shortest_path(start, waypoints[0], component))
        elif not route:
            route.append(start)

        current = waypoints[0]
        if not route:
            route.append(current)
        for waypoint in waypoints[1:]:
            if waypoint == current:
                continue
            connector = grid.shortest_path(current, waypoint, component)
            if len(connector) <= 1:
                connector = grid.shortest_path(current, waypoint)
            _append_connector(route, connector if connector else [waypoint])
            current = waypoint

    path = [(float(x), float(y)) for x, y in route]
    if len(path) > 1:
        # Return along the same obstacle-safe path.  This creates a real closed
        # trajectory whose length is exactly twice the former open route.
        path.extend(reversed(path[:-1]))
    return path


def _boustrophedon_waypoints(cells: set[Coord], swath_width: int) -> list[Coord]:
    min_y = min(y for _, y in cells)
    max_y = max(y for _, y in cells)
    waypoints: list[Coord] = []

    for band_index, band_y in enumerate(range(min_y, max_y + 1, swath_width)):
        band_cells = {cell for cell in cells if band_y <= cell[1] < band_y + swath_width}
        if not band_cells:
            continue
        y_center = band_y + (swath_width - 1) / 2.0
        columns = sorted({x for x, _ in band_cells})
        runs = _contiguous_runs(columns)
        if band_index % 2 == 1:
            runs = [list(reversed(run)) for run in reversed(runs)]

        cells_by_x: dict[int, list[Coord]] = {}
        for cell in band_cells:
            cells_by_x.setdefault(cell[0], []).append(cell)
        for run in runs:
            for x in run:
                candidates = cells_by_x[x]
                waypoint = min(candidates, key=lambda cell: (abs(cell[1] - y_center), cell[1], cell[0]))
                if not waypoints or waypoints[-1] != waypoint:
                    waypoints.append(waypoint)

    return waypoints


def _contiguous_runs(values: list[int]) -> list[list[int]]:
    if not values:
        return []
    runs = [[values[0]]]
    for value in values[1:]:
        if value == runs[-1][-1] + 1:
            runs[-1].append(value)
        else:
            runs.append([value])
    return runs


def _append_connector(route: list[Coord], connector: list[Coord]) -> None:
    if not connector:
        return
    if route and route[-1] == connector[0]:
        route.extend(connector[1:])
    else:
        route.extend(connector)


def _manhattan(first: Coord, second: Coord) -> int:
    return abs(first[0] - second[0]) + abs(first[1] - second[1])


def rebalance_connected(
    grid: GridMap,
    assignments: dict[Coord, int],
    targets: list[int],
    max_iter: int,
) -> dict[Coord, int]:
    result = dict(assignments)
    robot_count = len(targets)
    for _ in range(max_iter):
        counts = [sum(1 for rid in result.values() if rid == robot) for robot in range(robot_count)]
        over = [rid for rid in range(robot_count) if counts[rid] > targets[rid]]
        under = [rid for rid in range(robot_count) if counts[rid] < targets[rid]]
        if not over or not under:
            break
        changed = False
        for donor in sorted(over, key=lambda rid: counts[rid] - targets[rid], reverse=True):
            donor_cells = {cell for cell, rid in result.items() if rid == donor}
            candidates = [
                cell
                for cell in donor_cells
                if any(result.get(nbr) in under for nbr in grid.neighbors(cell))
            ]
            candidates.sort(key=lambda cell: _border_pressure(grid, result, cell, under))
            for cell in candidates:
                if counts[donor] <= targets[donor]:
                    break
                recipients = [result[nbr] for nbr in grid.neighbors(cell) if result.get(nbr) in under]
                recipients = sorted(set(recipients), key=lambda rid: targets[rid] - counts[rid], reverse=True)
                if not recipients:
                    continue
                if not _remains_connected(grid, donor_cells, cell):
                    continue
                recipient = recipients[0]
                result[cell] = recipient
                counts[donor] -= 1
                counts[recipient] += 1
                donor_cells.remove(cell)
                changed = True
        if not changed:
            break
    return result


def _remains_connected(grid: GridMap, cells: set[Coord], removed: Coord) -> bool:
    remaining = set(cells)
    remaining.discard(removed)
    if len(remaining) <= 1:
        return True
    return len(grid.connected_components(remaining)) == 1


def _border_pressure(
    grid: GridMap,
    assignments: dict[Coord, int],
    cell: Coord,
    target_robots: list[int],
) -> int:
    return -sum(1 for nbr in grid.neighbors(cell) if assignments.get(nbr) in target_robots)
