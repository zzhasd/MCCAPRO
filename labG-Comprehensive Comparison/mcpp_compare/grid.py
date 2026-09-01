from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush
from math import inf
from random import Random
from typing import Iterable

import numpy as np

Coord = tuple[int, int]


@dataclass(frozen=True)
class GridMap:
    """Four-neighbor 2D occupancy grid used by all planners."""

    width: int
    height: int
    obstacles: frozenset[Coord]

    @property
    def free_cells(self) -> list[Coord]:
        return [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) not in self.obstacles
        ]

    @property
    def free_count(self) -> int:
        return self.width * self.height - len(self.obstacles)

    def in_bounds(self, cell: Coord) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def is_free(self, cell: Coord) -> bool:
        return self.in_bounds(cell) and cell not in self.obstacles

    def neighbors(self, cell: Coord, allowed: set[Coord] | None = None) -> list[Coord]:
        x, y = cell
        candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        if allowed is None:
            return [candidate for candidate in candidates if self.is_free(candidate)]
        return [candidate for candidate in candidates if candidate in allowed]

    def nearest_free(self, seed: Coord, allowed: set[Coord] | None = None) -> Coord:
        """Return the closest free cell to seed by grid distance."""

        allowed_cells = allowed if allowed is not None else set(self.free_cells)
        if seed in allowed_cells:
            return seed
        queue: deque[Coord] = deque([seed])
        seen = {seed}
        while queue:
            cell = queue.popleft()
            for nxt in self._raw_neighbors(cell):
                if nxt in seen or not self.in_bounds(nxt):
                    continue
                if nxt in allowed_cells:
                    return nxt
                seen.add(nxt)
                queue.append(nxt)
        raise ValueError("grid has no reachable free cell")

    def bfs_distances(self, source: Coord, allowed: set[Coord] | None = None) -> dict[Coord, int]:
        allowed_cells = allowed if allowed is not None else set(self.free_cells)
        source = self.nearest_free(source, allowed_cells)
        queue: deque[Coord] = deque([source])
        distances = {source: 0}
        while queue:
            cell = queue.popleft()
            for nxt in self.neighbors(cell, allowed_cells):
                if nxt not in distances:
                    distances[nxt] = distances[cell] + 1
                    queue.append(nxt)
        return distances

    def shortest_path(
        self,
        start: Coord,
        goal: Coord,
        allowed: set[Coord] | None = None,
    ) -> list[Coord]:
        allowed_cells = allowed if allowed is not None else set(self.free_cells)
        if start not in allowed_cells or goal not in allowed_cells:
            return []
        queue: deque[Coord] = deque([start])
        parent: dict[Coord, Coord | None] = {start: None}
        while queue:
            cell = queue.popleft()
            if cell == goal:
                break
            for nxt in self.neighbors(cell, allowed_cells):
                if nxt not in parent:
                    parent[nxt] = cell
                    queue.append(nxt)
        if goal not in parent:
            return []
        path: list[Coord] = []
        cell: Coord | None = goal
        while cell is not None:
            path.append(cell)
            cell = parent[cell]
        return list(reversed(path))

    def connected_components(self, cells: Iterable[Coord]) -> list[set[Coord]]:
        remaining = set(cells)
        components: list[set[Coord]] = []
        while remaining:
            start = next(iter(remaining))
            queue: deque[Coord] = deque([start])
            remaining.remove(start)
            component = {start}
            while queue:
                cell = queue.popleft()
                for nxt in self.neighbors(cell, remaining):
                    remaining.remove(nxt)
                    component.add(nxt)
                    queue.append(nxt)
            components.append(component)
        components.sort(key=lambda comp: (-len(comp), min(comp) if comp else (inf, inf)))
        return components

    def spanning_tree_adjacency(
        self,
        cells: Iterable[Coord] | None = None,
        root: Coord | None = None,
    ) -> dict[Coord, set[Coord]]:
        """Build a deterministic BFS spanning forest over the requested cells."""

        allowed = set(cells) if cells is not None else set(self.free_cells)
        adjacency = {cell: set() for cell in allowed}
        if not allowed:
            return adjacency
        roots: list[Coord]
        if root is not None and root in allowed:
            roots = [root]
        else:
            roots = [min(allowed)]
        seen: set[Coord] = set()
        while len(seen) < len(allowed):
            start = roots.pop(0) if roots else min(allowed - seen)
            if start in seen:
                continue
            queue: deque[Coord] = deque([start])
            seen.add(start)
            while queue:
                cell = queue.popleft()
                for nxt in sorted(self.neighbors(cell, allowed)):
                    if nxt in seen:
                        continue
                    seen.add(nxt)
                    adjacency[cell].add(nxt)
                    adjacency[nxt].add(cell)
                    queue.append(nxt)
        return adjacency

    def _raw_neighbors(self, cell: Coord) -> list[Coord]:
        x, y = cell
        return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

    @classmethod
    def random_connected(
        cls,
        width: int,
        height: int,
        obstacle_ratio: float,
        seed: int,
        max_attempts: int = 200,
    ) -> "GridMap":
        """Generate a random obstacle grid whose free space is connected."""

        rng = np.random.default_rng(seed)
        total = width * height
        obstacle_count = int(total * obstacle_ratio)
        all_cells = [(x, y) for x in range(width) for y in range(height)]
        for _ in range(max_attempts):
            obstacle_ids = set(rng.choice(total, obstacle_count, replace=False).tolist())
            obstacles = frozenset(all_cells[index] for index in obstacle_ids)
            grid = cls(width, height, obstacles)
            free = grid.free_cells
            if not free:
                continue
            if len(grid.connected_components(free)) == 1:
                return grid
        # Fall back to a lightly structured map if random sampling is too fragmented.
        fallback_obstacles = {
            cell
            for idx, cell in enumerate(all_cells)
            if idx % max(7, int(1 / max(obstacle_ratio, 0.01))) == 0
        }
        grid = cls(width, height, frozenset(fallback_obstacles))
        largest = grid.connected_components(grid.free_cells)[0]
        return cls(width, height, frozenset(set(all_cells) - largest))

    @classmethod
    def random_with_filled_disconnected(
        cls,
        width: int,
        height: int,
        obstacle_ratio: float,
        seed: int,
    ) -> "GridMap":
        """Generate one random map, then fill non-largest free components as obstacles."""

        rng = np.random.default_rng(seed)
        total = width * height
        obstacle_count = int(total * obstacle_ratio)
        all_cells = [(x, y) for x in range(width) for y in range(height)]
        obstacle_ids = set(rng.choice(total, obstacle_count, replace=False).tolist())
        obstacles = {all_cells[index] for index in obstacle_ids}
        grid = cls(width, height, frozenset(obstacles))
        components = grid.connected_components(grid.free_cells)
        if not components:
            raise ValueError("random map has no free cells")
        largest = components[0]
        filled_obstacles = obstacles | (set(grid.free_cells) - largest)
        return cls(width, height, frozenset(filled_obstacles))


def choose_roots(grid: GridMap, robot_count: int) -> list[Coord]:
    """Place initial robot roots on a coarse lattice and snap them to free cells."""

    cols = int(np.ceil(np.sqrt(robot_count)))
    rows = int(np.ceil(robot_count / cols))
    roots: list[Coord] = []
    used: set[Coord] = set()
    for rid in range(robot_count):
        col = rid % cols
        row = rid // cols
        x = round((col + 0.5) * grid.width / cols - 0.5)
        y = round((row + 0.5) * grid.height / rows - 0.5)
        root = grid.nearest_free((int(x), int(y)), set(grid.free_cells) - used)
        roots.append(root)
        used.add(root)
    return roots


def normalize_weights(weights: Iterable[float], robot_count: int) -> np.ndarray:
    arr = np.asarray(list(weights), dtype=float)
    if arr.size != robot_count:
        raise ValueError(f"expected {robot_count} weights, got {arr.size}")
    arr = np.clip(arr, 1e-9, None)
    return arr / arr.sum()


def target_counts(weights: Iterable[float], total: int, robot_count: int) -> list[int]:
    normalized = normalize_weights(weights, robot_count)
    raw = normalized * total
    counts = np.floor(raw).astype(int)
    remainder = total - int(counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        for idx in order[:remainder]:
            counts[idx] += 1
    return counts.astype(int).tolist()


def weighted_multi_source_assignment(
    adjacency: dict[Coord, set[Coord]],
    roots: list[Coord],
    weights: Iterable[float],
) -> dict[Coord, int]:
    """Assign graph vertices to roots by weighted geodesic distance."""

    robot_count = len(roots)
    normalized = normalize_weights(weights, robot_count)
    heap: list[tuple[float, int, Coord]] = []
    assignments: dict[Coord, int] = {}
    distances: dict[tuple[Coord, int], int] = {}
    for rid, root in enumerate(roots):
        if root not in adjacency:
            continue
        heappush(heap, (0.0, rid, root))
        distances[(root, rid)] = 0
    while heap:
        _, rid, cell = heappop(heap)
        if cell in assignments:
            continue
        assignments[cell] = rid
        base_distance = distances[(cell, rid)]
        for nxt in sorted(adjacency[cell]):
            if nxt in assignments:
                continue
            next_distance = base_distance + 1
            key = (nxt, rid)
            if next_distance >= distances.get(key, 10**12):
                continue
            distances[key] = next_distance
            priority = next_distance / np.sqrt(normalized[rid])
            heappush(heap, (float(priority), rid, nxt))
    return assignments


def shuffled(items: Iterable[Coord], seed: int) -> list[Coord]:
    result = list(items)
    Random(seed).shuffle(result)
    return result
