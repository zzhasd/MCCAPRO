"""FACT-MCCA tile-first planner, v2.3.2.

Current algorithm only:
1. global free-space tiling (greedy/randomized + local exact MILP repair),
2. connected tile-graph initialization,
3. connectivity-preserving single-tile transfers accepted by closed TSP path CV,
4. depot-free obstacle-safe TSP routing,
5. v2.2-compatible portal shortcutting, then strict any-angle removal of redundant
   repeated tile-center visits without a CV-preserving balance guard.

"""
from __future__ import annotations

__version__ = "2.3.2"

from collections import deque
import copy
from dataclasses import dataclass
import heapq
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import depth_first_order, dijkstra as sparse_dijkstra, minimum_spanning_tree
from scipy.optimize import Bounds, LinearConstraint, milp

Cell = Tuple[int, int]
Tile = Tuple[int, int, int, int, int]
TILE_SHAPES = ((4, 4), (4, 2), (2, 4), (2, 2), (2, 1), (1, 2))


@dataclass
class StageTimes:
    """Top-level ``solve()`` stage timings in seconds.

    ``core_total`` is exactly the three algorithmic stages requested by Lab D:
    tiling -> partition -> routing.  Validation is reported separately.
    """

    partition: float = 0.0
    tiling: float = 0.0
    routing: float = 0.0
    validation: float = 0.0

    @property
    def core_total(self) -> float:
        return self.partition + self.tiling + self.routing

    @property
    def measured_total(self) -> float:
        return self.core_total + self.validation


def coefficient_of_variation(values: Sequence[float]) -> float:
    a = np.asarray(values, dtype=float)
    return float(np.std(a) / np.mean(a)) if a.size and np.mean(a) else 0.0


MapShape = Union[int, Sequence[int]]


def _normalize_map_shape(map_shape: MapShape) -> Tuple[int, int]:
    """Return ``(width, height)``; an integer naturally denotes a square."""
    if isinstance(map_shape, (int, np.integer)):
        width = height = int(map_shape)
    else:
        values = tuple(map_shape)
        if len(values) != 2:
            raise ValueError("map_shape must be an int or a (width, height) pair")
        width, height = map(int, values)
    if width < 2 or height < 2:
        raise ValueError("map width and height must both be >= 2")
    return width, height


def _neighbors(idx: int, width: int, height: int):
    """4-neighbors for x-major flattening: ``flat = x * height + y``."""
    x, y = divmod(idx, height)
    if x:
        yield idx - height
    if x + 1 < width:
        yield idx + height
    if y:
        yield idx - 1
    if y + 1 < height:
        yield idx + 1


def _free_components(mask: np.ndarray) -> List[List[int]]:
    width, height = mask.shape
    unseen = set(map(int, np.flatnonzero(~mask.ravel())))
    components: List[List[int]] = []
    while unseen:
        start = unseen.pop()
        comp = [start]
        q = deque([start])
        while q:
            u = q.popleft()
            for v in _neighbors(u, width, height):
                if v in unseen:
                    unseen.remove(v)
                    comp.append(v)
                    q.append(v)
        components.append(comp)
    return components


def make_random_map(map_shape: MapShape, obstacle_ratio: float, seed: int) -> np.ndarray:
    """Generate random obstacles and retain only the largest free component.

    Arrays remain x-major with shape ``(width, height)``. Passing an integer
    follows the exact v2.1.1 square-map RNG/reshape path.
    """
    width, height = _normalize_map_shape(map_shape)
    rng = np.random.default_rng(seed)
    cell_count = width * height
    obstacle_count = int(cell_count * obstacle_ratio)
    mask = np.zeros(cell_count, dtype=bool)
    mask[rng.choice(cell_count, obstacle_count, replace=False)] = True
    mask = mask.reshape((width, height))
    components = _free_components(mask)
    if len(components) > 1:
        largest = set(max(components, key=len))
        flat = mask.ravel()
        for u in np.flatnonzero(~flat):
            if int(u) not in largest:
                flat[u] = True
    return mask


class TileFirstMCPP:
    """Global tile-first, depot-free multi-robot coverage planner."""

    def __init__(
        self,
        map_shape: MapShape,
        robot_num: int,
        obstacle_mask: Optional[np.ndarray] = None,
        obstacle_ratio: float = 0.10,
        seed: int = 42,
        partition_iterations: int = 60,
        tiling_time_limit: float = 6.0,
        candidate_limit: int = 8,
        tile_shapes: Optional[Sequence[Tuple[int, int]]] = None,
        service_time_per_tile: float = 1.0,
        robot_speed: float = 1.0,
        turn_time_90: float = 0.0,
        strict_route_postprocess: bool = True,
        robot_weights: Optional[Sequence[float]] = None,
    ) -> None:
        self.width, self.height = _normalize_map_shape(map_shape)
        if robot_num < 1:
            raise ValueError("robot_num >= 1 is required")
        self.map_shape = (self.width, self.height)
        self.k = int(robot_num)

        # Robot patrol weights.  None means equal weights.  Zero means the robot
        # is offline/faulted and receives no patrol region or route.  Positive
        # weights are renormalized across active robots, so [0.6, 0.4, 0.0] stays
        # [0.6, 0.4, 0.0] while [3, 1, 0] becomes [0.75, 0.25, 0.0].
        if robot_weights is None:
            weights = np.full(self.k, 1.0 / self.k, dtype=float)
        else:
            weights = np.asarray(tuple(robot_weights), dtype=float)
            if weights.ndim != 1 or len(weights) != self.k:
                raise ValueError(f"robot_weights must contain exactly {self.k} values")
            if not np.isfinite(weights).all() or np.any(weights < 0.0):
                raise ValueError("robot_weights must contain only finite nonnegative values")
            total_weight = float(weights.sum())
            if total_weight <= 0.0:
                message = "机器人权重不能全为0，规划失败"
                print(message)
                raise ValueError(message)
            weights = weights / total_weight
        self.robot_weights = weights
        self.active_robot_ids = tuple(map(int, np.flatnonzero(self.robot_weights > 0.0)))
        self.inactive_robot_ids = tuple(map(int, np.flatnonzero(self.robot_weights == 0.0)))
        self.active_robot_count = len(self.active_robot_ids)

        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.obstacles = (
            make_random_map(self.map_shape, obstacle_ratio, seed)
            if obstacle_mask is None
            else np.asarray(obstacle_mask, dtype=bool).copy()
        )
        if self.obstacles.shape != self.map_shape:
            raise ValueError(f"obstacle_mask must have shape {self.map_shape}")
        components = _free_components(self.obstacles)
        if len(components) != 1:
            raise ValueError("The current tile-first planner requires connected free space")
        if np.count_nonzero(~self.obstacles) < self.active_robot_count:
            raise ValueError("Fewer free cells than active robots")

        self.free_flat = np.flatnonzero(~self.obstacles.ravel())
        self.partition_iterations = max(1, int(partition_iterations))
        self.tiling_time_limit = max(0.0, float(tiling_time_limit))
        self.candidate_limit = max(2, int(candidate_limit))
        if service_time_per_tile < 0 or robot_speed <= 0 or turn_time_90 < 0:
            raise ValueError("service_time_per_tile >= 0, robot_speed > 0, turn_time_90 >= 0 required")
        self.service_time_per_tile = float(service_time_per_tile)
        self.robot_speed = float(robot_speed)
        self.turn_time_90 = float(turn_time_90)
        self.strict_route_postprocess = bool(strict_route_postprocess)

        raw_shapes = TILE_SHAPES if tile_shapes is None else tuple(tile_shapes)
        normalized_shapes: List[Tuple[int, int]] = []
        for shape in raw_shapes:
            if len(shape) != 2:
                raise ValueError("Every tile shape must be a (width, height) pair")
            w, h = int(shape[0]), int(shape[1])
            if w < 1 or h < 1:
                raise ValueError("Tile widths and heights must be positive")
            if (w, h) != (1, 1) and (w, h) not in normalized_shapes:
                normalized_shapes.append((w, h))
        self.tile_shapes = tuple(normalized_shapes)
        self.action_shapes = self.tile_shapes + ((1, 1),)

        self.assignments: Optional[np.ndarray] = None
        self.global_tiles: List[Tile] = []
        self.tiles: List[Tile] = []
        self.tile_adjacency: List[set[int]] = []
        self.tile_partition_seeds: List[int] = []
        self.tile_owners = np.empty(0, dtype=np.int16)
        self.tile_partition_trace: List[dict] = []
        self.routes: Dict[int, dict] = {}
        self.centroids_xy: List[Optional[Cell]] = []

        # Geometry caches for the immutable global tiling.  Partition search only
        # changes tile owners, never rectangle geometry, so center/portal edge
        # data can be built once and reused by every exact route evaluation.
        self._global_tile_index: Dict[Tuple[int, int, int, int], int] = {}
        self._global_centers: List[np.ndarray] = []
        self._global_edge_weights: Dict[Tuple[int, int], float] = {}
        self._global_edge_portals: Dict[Tuple[int, int], np.ndarray] = {}
        self._transfer_cache_owners: Optional[np.ndarray] = None
        self._transfer_articulations: Dict[int, set[int]] = {}
        self._transfer_counts: Dict[int, int] = {}

    def _region_mask(self, rid: int) -> np.ndarray:
        return self.assignments == rid

    def _scan_greedy(self, mask: np.ndarray, rid: int, xrev: bool, yrev: bool) -> List[Tile]:
        covered = np.zeros_like(mask)
        out: List[Tile] = []
        for w, h in self.tile_shapes:
            # Compute mask-feasible placements once in C.  The subsequent Python
            # scan retains the exact original x/y order and covered-cell test.
            legal = np.lib.stride_tricks.sliding_window_view(mask, (w, h)).all(axis=(-2, -1))
            xr = range(self.width - w, -1, -1) if xrev else range(self.width - w + 1)
            yr = range(self.height - h, -1, -1) if yrev else range(self.height - h + 1)
            for x in xr:
                for y in yr:
                    if legal[x, y] and not covered[x:x+w, y:y+h].any():
                        out.append((x, y, w, h, rid))
                        covered[x:x+w, y:y+h] = True
        for x, y in np.argwhere(mask & ~covered):
            out.append((int(x), int(y), 1, 1, rid))
        return out

    def _randomized_greedy(self, mask: np.ndarray, rid: int, trials: int = 8) -> List[Tile]:
        placements = []
        for w, h in self.tile_shapes:
            legal = np.lib.stride_tricks.sliding_window_view(mask, (w, h)).all(axis=(-2, -1))
            # np.argwhere is row-major here, exactly matching the original nested
            # x-then-y loops, hence RNG jitter is attached to identical placements.
            for x, y in np.argwhere(legal):
                placements.append((int(x), int(y), w, h, rid))
        best: Optional[List[Tile]] = None
        for trial in range(trials):
            covered = np.zeros_like(mask)
            jitter = self.rng.random(len(placements))
            order = sorted(range(len(placements)), key=lambda i: (-(placements[i][2] * placements[i][3]), jitter[i]))
            chosen = []
            for i in order:
                x, y, w, h, _ = placements[i]
                if not covered[x:x+w, y:y+h].any():
                    chosen.append(placements[i])
                    covered[x:x+w, y:y+h] = True
            for x, y in np.argwhere(mask & ~covered):
                chosen.append((int(x), int(y), 1, 1, rid))
            if best is None or len(chosen) < len(best):
                best = chosen
        return best or []

    @staticmethod
    def _tile_cells(tile: Tile) -> List[Cell]:
        x, y, w, h, _ = tile
        return [(px, py) for px in range(x, x + w) for py in range(y, y + h)]

    def _exact_patch(self, patch: set, rid: int, time_limit: float) -> Optional[List[Tile]]:
        if not patch:
            return []
        candidates: List[Tile] = []
        cells = sorted(patch)
        cell_id = {c: i for i, c in enumerate(cells)}
        xs = [c[0] for c in cells]
        ys = [c[1] for c in cells]
        for w, h in self.action_shapes:
            for x in range(min(xs), max(xs) - w + 2):
                for y in range(min(ys), max(ys) - h + 2):
                    block = {(px, py) for px in range(x, x+w) for py in range(y, y+h)}
                    if block.issubset(patch):
                        candidates.append((x, y, w, h, rid))
        if not candidates:
            return None
        rows, cols = [], []
        for j, t in enumerate(candidates):
            for c in self._tile_cells(t):
                rows.append(cell_id[c])
                cols.append(j)
        A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(cells), len(candidates))).tocsr()
        result = milp(
            np.ones(len(candidates)),
            integrality=np.ones(len(candidates)),
            bounds=Bounds(0, 1),
            constraints=LinearConstraint(A, 1, 1),
            options={"time_limit": max(0.02, time_limit), "mip_rel_gap": 0.0, "presolve": True},
        )
        if result.x is None:
            return None
        return [candidates[i] for i, z in enumerate(result.x) if z > 0.5]

    def _local_milp_improve(self, tiles: List[Tile], rid: int, deadline: float) -> List[Tile]:
        # Exact global solve is cheap and valuable for small regions.
        region_cells = set(map(tuple, np.argwhere(self._region_mask(rid))))
        if len(region_cells) <= 420:
            exact = self._exact_patch(region_cells, rid, min(2.0, deadline - time.perf_counter()))
            return exact if exact is not None and len(exact) < len(tiles) else tiles
        window = 12
        candidates = []
        for t in sorted(tiles, key=lambda z: z[2] * z[3]):
            if t[2] * t[3] >= 16:
                continue
            cx, cy = t[0] + t[2] / 2, t[1] + t[3] / 2
            x0 = int(np.clip(round(cx - window / 2), 0, max(0, self.width - window)))
            y0 = int(np.clip(round(cy - window / 2), 0, max(0, self.height - window)))
            candidates.append((x0 // 2 * 2, y0 // 2 * 2))
        seen = set()
        for x0, y0 in candidates:
            if time.perf_counter() >= deadline or len(seen) >= 55:
                break
            if (x0, y0) in seen:
                continue
            seen.add((x0, y0))
            inside = [t for t in tiles if x0 <= t[0] and y0 <= t[1] and t[0] + t[2] <= x0 + window and t[1] + t[3] <= y0 + window]
            if len(inside) < 3:
                continue
            patch = {c for t in inside for c in self._tile_cells(t)}
            if len(inside) <= math.ceil(len(patch) / 16):
                continue
            exact = self._exact_patch(patch, rid, min(0.12, deadline - time.perf_counter()))
            if exact is not None and len(exact) < len(inside):
                remove = set(inside)
                tiles = [t for t in tiles if t not in remove] + exact
        return tiles

    @staticmethod
    def _two_opt(order: List[int], d: np.ndarray, passes: int = 3) -> List[int]:
        m = len(order)
        if m < 4:
            return order
        # Keep one compact integer buffer for the whole local search instead of
        # rebuilding ``np.asarray(order)`` and ``np.arange`` for every i.  The
        # scan order, gain formula, tie-breaking and accepted reversals are
        # unchanged, so this is a pure implementation-level optimization.
        arr = np.asarray(order, dtype=np.intp).copy()
        for _ in range(passes):
            improved = False
            for i in range(m - 2):
                a = int(arr[i])
                b = int(arr[i + 1])
                stop = m - (1 if i == 0 else 0)
                start = i + 2
                if start >= stop:
                    continue
                c = arr[start:stop]
                if stop < m:
                    e = arr[start + 1:stop + 1]
                else:
                    # Only i>0 reaches stop==m.  Preserve the original modulo
                    # edge for j==m-1 without allocating an index vector.
                    e = np.empty_like(c)
                    if len(c) > 1:
                        e[:-1] = arr[start + 1:m]
                    e[-1] = arr[0]
                gains = d[a, c] + d[b, e] - d[a, b] - d[c, e]
                pos = int(np.argmin(gains))
                if gains[pos] < -1e-9:
                    j = start + pos
                    arr[i + 1:j + 1] = arr[i + 1:j + 1][::-1]
                    improved = True
            if not improved:
                break
        return arr.tolist()

    @staticmethod
    def _tour_length(order: Sequence[int], d: np.ndarray) -> float:
        if len(order) <= 1:
            return 0.0
        return float(sum(d[order[i], order[(i + 1) % len(order)]] for i in range(len(order))))

    @staticmethod
    def _metric_mst_orders(d: np.ndarray, start: int) -> Tuple[List[int], List[int], float]:
        """Return MST preorder, exact doubled-tree walk, and MST weight."""
        m = len(d)
        if m <= 1:
            return ([start] if m else []), ([start] if m else []), 0.0

        # Exact low-memory Kruskal equivalent to the previous SciPy sparse-MST
        # call.  Stable ordering is essential because equal-weight metric edges
        # are common; it reproduces SciPy's edge set and DFS preorder exactly.
        ui, vi = np.triu_indices(m, k=1)
        weights = d[ui, vi]
        edge_order = np.argsort(weights, kind="stable")
        parent = np.arange(m, dtype=np.int32)
        rank = np.zeros(m, dtype=np.int8)
        selected: List[Tuple[int, int, float]] = []

        def find_root(x: int) -> int:
            root = x
            while parent[root] != root:
                root = int(parent[root])
            while parent[x] != x:
                nxt = int(parent[x])
                parent[x] = root
                x = nxt
            return root

        for edge_idx in edge_order:
            u = int(ui[edge_idx])
            v = int(vi[edge_idx])
            ru = find_root(u)
            rv = find_root(v)
            if ru == rv:
                continue
            if rank[ru] < rank[rv]:
                ru, rv = rv, ru
            parent[rv] = ru
            if rank[ru] == rank[rv]:
                rank[ru] += 1
            selected.append((u, v, float(weights[edge_idx])))
            if len(selected) == m - 1:
                break

        # CSR/COO output from the old implementation was row-major.  Sorting
        # selected edges restores that exact adjacency append and sum order.
        selected.sort(key=lambda z: (z[0], z[1]))
        adjacency = [[] for _ in range(m)]
        for u, v, _ in selected:
            adjacency[u].append(v)
            adjacency[v].append(u)
        weight = float(np.asarray([w for _, _, w in selected], dtype=float).sum())

        preorder: List[int] = []
        seen = np.zeros(m, dtype=bool)
        stack = [start]
        while stack:
            u = stack.pop()
            if seen[u]:
                continue
            seen[u] = True
            preorder.append(u)
            for v in reversed(adjacency[u]):
                if not seen[v]:
                    stack.append(v)

        walk = [start]
        stack2 = [(start, -1, iter(adjacency[start]))]
        while stack2:
            node, parent_node, children = stack2[-1]
            try:
                child = next(children)
                if child == parent_node:
                    continue
                walk.append(child)
                stack2.append((child, node, iter(adjacency[child])))
            except StopIteration:
                stack2.pop()
                if stack2:
                    walk.append(stack2[-1][0])
        return preorder, walk, weight

    def _path_turn_metrics(self, path: Sequence[Sequence[float]]) -> Tuple[float, float]:
        """Return equivalent quarter turns and rotation time for a polyline.

        The initial heading is north.  Arbitrary portal segments are handled by charging their
        smallest wrapped heading change, normalized by 90 degrees.
        """
        previous_angle = math.pi / 2.0
        quarter_turns = 0.0
        half_pi = math.pi / 2.0
        two_pi = 2.0 * math.pi
        for p, q in zip(path, path[1:]):
            dx = float(q[0]) - float(p[0])
            dy = float(q[1]) - float(p[1])
            if math.hypot(dx, dy) <= 1e-12:
                continue
            angle = math.atan2(dy, dx)
            change = abs((angle - previous_angle + math.pi) % two_pi - math.pi)
            quarter_turns += change / half_pi
            previous_angle = angle
        return float(quarter_turns), float(quarter_turns * self.turn_time_90)

    def _tile_geodesic_metric(
        self, tiles: List[Tile]
    ) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, Dict[Tuple[int, int], np.ndarray]]:
        """Shortest center-to-center metric on the obstacle-safe tile graph.

        Global rectangle geometry is immutable after tiling.  Reuse its exact
        center/portal edge data and only induce the current robot subgraph.
        This preserves the same sparse Dijkstra metric and predecessor semantics
        while eliminating repeated cell-level geometry reconstruction.
        """
        if self._global_tile_index and len(self._global_centers) == len(self.global_tiles):
            global_ids = [
                self._global_tile_index[(t[0], t[1], t[2], t[3])] for t in tiles
            ]
            local_of = {gid: i for i, gid in enumerate(global_ids)}
            centers = [self._global_centers[gid] for gid in global_ids]
            rows, cols, data = [], [], []
            edge_portals: Dict[Tuple[int, int], np.ndarray] = {}
            for gu in global_ids:
                u = local_of[gu]
                for gv in self.tile_adjacency[gu]:
                    if gv <= gu:
                        continue
                    v = local_of.get(gv)
                    if v is None:
                        continue
                    ge = (gu, gv)
                    weight = self._global_edge_weights[ge]
                    rows.extend((u, v)); cols.extend((v, u)); data.extend((weight, weight))
                    edge_portals[(min(u, v), max(u, v))] = self._global_edge_portals[ge]
            graph = sp.csr_matrix((data, (rows, cols)), shape=(len(tiles), len(tiles)))
            dist, predecessors = sparse_dijkstra(graph, directed=False, return_predecessors=True)
            dist = np.asarray(dist, dtype=float)
            if not np.isfinite(dist).all():
                raise AssertionError("Tile adjacency roadmap is disconnected")
            return centers, dist, np.asarray(predecessors, dtype=np.int32), edge_portals

        # Defensive fallback for direct/private use before global cache creation.
        centers = [np.array((t[0] + t[2] / 2, t[1] + t[3] / 2), dtype=float) for t in tiles]
        owner: Dict[Cell, int] = {}
        for i, t in enumerate(tiles):
            x, y, w, h, _ = t
            for px in range(x, x + w):
                for py in range(y, y + h):
                    owner[(px, py)] = i
        edge_weights: Dict[Tuple[int, int], float] = {}
        edge_portals: Dict[Tuple[int, int], np.ndarray] = {}
        for (x, y), i in owner.items():
            for nb in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
                j = owner.get(nb)
                if j is not None and i != j:
                    e = (min(i, j), max(i, j))
                    if nb[0] == x + 1:
                        portal = np.array((x + 1.0, y + 0.5))
                    elif nb[0] == x - 1:
                        portal = np.array((x + 0.0, y + 0.5))
                    elif nb[1] == y + 1:
                        portal = np.array((x + 0.5, y + 1.0))
                    else:
                        portal = np.array((x + 0.5, y + 0.0))
                    ci = centers[i]
                    cj = centers[j]
                    weight = float(
                        math.hypot(float(ci[0] - portal[0]), float(ci[1] - portal[1]))
                        + math.hypot(float(portal[0] - cj[0]), float(portal[1] - cj[1]))
                    )
                    if e not in edge_weights or weight < edge_weights[e]:
                        edge_weights[e] = weight
                        edge_portals[e] = portal
        rows, cols, data = [], [], []
        for (u, v), w in edge_weights.items():
            rows.extend((u, v)); cols.extend((v, u)); data.extend((w, w))
        graph = sp.csr_matrix((data, (rows, cols)), shape=(len(tiles), len(tiles)))
        dist, predecessors = sparse_dijkstra(graph, directed=False, return_predecessors=True)
        dist = np.asarray(dist, dtype=float)
        if not np.isfinite(dist).all():
            raise AssertionError("Tile adjacency roadmap is disconnected")
        return centers, dist, np.asarray(predecessors, dtype=np.int32), edge_portals

    @staticmethod
    def _expand_metric_tour(
        order: List[int], predecessors: np.ndarray, centers: List[np.ndarray],
        edge_portals: Dict[Tuple[int, int], np.ndarray],
    ) -> List[List[float]]:
        """Expand metric edges through shared-boundary portals.

        Each center-to-portal segment lies in one rectangle, so the resulting
        piecewise-linear route is collision-free by construction even when two
        adjacent rectangles share only part of an edge.
        """
        if not order:
            return []
        expanded: List[List[float]] = []
        for i, source in enumerate(order):
            target = order[(i + 1) % len(order)]
            if source == target:
                tile_segment = [source]
            else:
                reverse = [target]
                current = target
                while current != source:
                    current = int(predecessors[source, current])
                    if current < 0:
                        raise AssertionError("No predecessor for a finite tile-graph path")
                    reverse.append(current)
                tile_segment = list(reversed(reverse))
            if not expanded:
                expanded.append(centers[tile_segment[0]].tolist())
            for u, v in zip(tile_segment, tile_segment[1:]):
                expanded.append(edge_portals[(min(u, v), max(u, v))].tolist())
                expanded.append(centers[v].tolist())
        return expanded

    @staticmethod
    def _polyline_length(path: Sequence[Sequence[float]]) -> float:
        """Euclidean length of a geometric polyline."""
        if len(path) <= 1:
            return 0.0
        return float(sum(
            np.linalg.norm(np.asarray(b, dtype=float) - np.asarray(a, dtype=float))
            for a, b in zip(path, path[1:])
        ))

    @staticmethod
    def _segment_closed_cell_interval(
        a: Sequence[float],
        b: Sequence[float],
        cell_x: int,
        cell_y: int,
        tol: float = 1e-11,
        expand: float = 0.0,
    ) -> Optional[Tuple[float, float]]:
        """Clip a segment against one closed unit cell.

        Returns the parameter interval ``[t_enter, t_exit]`` for which
        ``a + t * (b-a)`` lies in the closed square.  ``None`` means no
        intersection.  ``expand`` is used only for conservative obstacle
        checks; for region-boundary classification we keep the exact cell.
        """
        p = np.asarray(a, dtype=float)
        q = np.asarray(b, dtype=float)
        d = q - p
        lo = np.array((cell_x - expand, cell_y - expand), dtype=float)
        hi = np.array((cell_x + 1.0 + expand, cell_y + 1.0 + expand), dtype=float)
        t_enter, t_exit = 0.0, 1.0
        for axis in range(2):
            if abs(float(d[axis])) <= tol:
                if float(p[axis]) < float(lo[axis]) - tol or float(p[axis]) > float(hi[axis]) + tol:
                    return None
                continue
            t0 = float((lo[axis] - p[axis]) / d[axis])
            t1 = float((hi[axis] - p[axis]) / d[axis])
            if t0 > t1:
                t0, t1 = t1, t0
            t_enter = max(t_enter, t0)
            t_exit = min(t_exit, t1)
            if t_enter > t_exit + tol:
                return None
        if t_exit < -tol or t_enter > 1.0 + tol:
            return None
        return max(0.0, t_enter), min(1.0, t_exit)

    @classmethod
    def _segment_intersects_closed_cell(
        cls,
        a: Sequence[float],
        b: Sequence[float],
        cell_x: int,
        cell_y: int,
        tol: float = 1e-11,
    ) -> bool:
        """True if a segment has ANY contact with a closed obstacle cell.

        A black obstacle is treated as the closed square
        ``[x,x+1] x [y,y+1]``.  Interior crossing, edge contact, edge overlap,
        and even a single corner touch are all collisions.
        """
        return cls._segment_closed_cell_interval(
            a, b, cell_x, cell_y, tol=tol, expand=tol
        ) is not None

    @classmethod
    def _other_region_contact_is_corner_only(
        cls,
        a: Sequence[float],
        b: Sequence[float],
        cell_x: int,
        cell_y: int,
        tol: float = 1e-10,
    ) -> bool:
        """Whether contact with another robot's cell is allowed.

        The rule is: another robot's cell may be touched only at an isolated grid corner.
        Entering its interior, touching a non-corner point on an edge, or
        overlapping an edge for a nonzero length are all forbidden.
        """
        interval = cls._segment_closed_cell_interval(
            a, b, cell_x, cell_y, tol=tol, expand=0.0
        )
        if interval is None:
            return True
        t_enter, t_exit = interval
        if t_exit - t_enter > tol:
            return False
        p = np.asarray(a, dtype=float)
        q = np.asarray(b, dtype=float)
        hit = p + 0.5 * (t_enter + t_exit) * (q - p)
        x_is_corner = (
            abs(float(hit[0]) - cell_x) <= 10.0 * tol
            or abs(float(hit[0]) - (cell_x + 1.0)) <= 10.0 * tol
        )
        y_is_corner = (
            abs(float(hit[1]) - cell_y) <= 10.0 * tol
            or abs(float(hit[1]) - (cell_y + 1.0)) <= 10.0 * tol
        )
        return bool(x_is_corner and y_is_corner)

    def _segment_is_portal_shortcut_legal(
        self,
        a: Sequence[float],
        b: Sequence[float],
        rid: int,
    ) -> bool:
        """Check one candidate center-to-center replacement segment.

        Only *new* A->C shortcut candidates are judged here.

        Rules:
          1. black obstacle cells are closed: ANY intersection is forbidden;
          2. cells assigned to another robot may be contacted only at an
             isolated corner; their interiors/edges may not be crossed;
          3. no alternative route is generated when the shortcut is illegal --
             the original ``A -> portal -> C`` geometry is kept unchanged.
        """
        if self.assignments is None:
            return False
        p = np.asarray(a, dtype=float)
        q = np.asarray(b, dtype=float)
        if p.shape != (2,) or q.shape != (2,):
            return False
        eps = 1e-10
        if (
            np.any(p < -eps) or np.any(q < -eps)
            or p[0] > self.width + eps or q[0] > self.width + eps or p[1] > self.height + eps or q[1] > self.height + eps
        ):
            return False

        min_x = max(0, int(math.floor(min(float(p[0]), float(q[0])) - 1e-9)))
        max_x = min(self.width - 1, int(math.floor(max(float(p[0]), float(q[0])) + 1e-9)))
        min_y = max(0, int(math.floor(min(float(p[1]), float(q[1])) - 1e-9)))
        max_y = min(self.height - 1, int(math.floor(max(float(p[1]), float(q[1])) + 1e-9)))

        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if self.obstacles[x, y]:
                    if self._segment_intersects_closed_cell(p, q, x, y):
                        return False
                elif int(self.assignments[x, y]) != rid:
                    if not self._other_region_contact_is_corner_only(p, q, x, y):
                        return False
        return True

    @staticmethod
    def _dedupe_polyline(path: Sequence[Sequence[float]]) -> List[List[float]]:
        out: List[List[float]] = []
        for point in path:
            p = np.asarray(point, dtype=float)
            if not out or not np.allclose(
                p, np.asarray(out[-1], dtype=float), atol=1e-12, rtol=0.0
            ):
                out.append(p.tolist())
        return out

    @staticmethod
    def _point_key(point: Sequence[float], digits: int = 10) -> Tuple[float, float]:
        p = np.asarray(point, dtype=float)
        return (round(float(p[0]), digits), round(float(p[1]), digits))

    def postprocess_routes_strict(self) -> Dict[int, dict]:
        """Strict unconstrained post-process with a v2.2-dominating repeated-center stage.

        v2.3.2 deliberately applies every locally accepted shortening opportunity.
        No CV/fairness rollback is performed after the partition and TSP order are frozen.
        """
        if not self.strict_route_postprocess:
            return self.routes
        for rid in range(self.k):
            self._postprocess_one_route_strict(rid)
        return self.routes

    def _partition_sizes(self) -> np.ndarray:
        return np.array([np.count_nonzero(self.assignments == rid) for rid in range(self.k)])

    def _point_in_region_closure(self, point: np.ndarray, rid: int) -> bool:
        """Whether a geometric point lies in the closure of a robot's cells."""
        x, y = map(float, point)
        eps = 1e-8
        xs = {math.floor(x - eps), math.floor(x + eps)}
        ys = {math.floor(y - eps), math.floor(y + eps)}
        return any(
            0 <= px < self.width and 0 <= py < self.height and self.assignments[px, py] == rid
            for px in xs for py in ys
        )

    def _validate_strict_obstacle_routes(self) -> bool:
        """Check that no final route segment even touches a black cell."""
        for rid in range(self.k):
            path = self.routes.get(rid, {}).get("path", [])
            for a, b in zip(path, path[1:]):
                p = np.asarray(a, dtype=float)
                q = np.asarray(b, dtype=float)
                min_x = max(0, int(math.floor(min(float(p[0]), float(q[0])) - 1e-9)))
                max_x = min(self.width - 1, int(math.floor(max(float(p[0]), float(q[0])) + 1e-9)))
                min_y = max(0, int(math.floor(min(float(p[1]), float(q[1])) - 1e-9)))
                max_y = min(self.height - 1, int(math.floor(max(float(p[1]), float(q[1])) + 1e-9)))
                for x in range(min_x, max_x + 1):
                    for y in range(min_y, max_y + 1):
                        if self.obstacles[x, y] and self._segment_intersects_closed_cell(p, q, x, y):
                            return False
        return True

    def tile_global_hybrid(self) -> List[Tile]:
        """Tile the entire free workspace before any robot partition exists."""
        mask = ~self.obstacles
        old_assignments = self.assignments
        temporary = np.full(self.map_shape, -1, dtype=np.int16)
        temporary[mask] = 0
        self.assignments = temporary
        try:
            schemes = [
                self._scan_greedy(mask, 0, xr, yr)
                for xr, yr in ((False, False), (True, True), (False, True), (True, False))
            ]
            schemes.append(self._randomized_greedy(mask, 0, trials=10))
            tiles = min(schemes, key=len)
            deadline = time.perf_counter() + self.tiling_time_limit
            tiles = self._local_milp_improve(tiles, 0, deadline)
        finally:
            self.assignments = old_assignments

        self.global_tiles = [(x, y, w, h, -1) for x, y, w, h, _ in tiles]
        self.tiles = list(self.global_tiles)
        self._build_tile_adjacency()
        return self.tiles

    def _build_tile_adjacency(self) -> None:
        m = len(self.global_tiles)
        if m < self.active_robot_count:
            raise ValueError(
                f"Global tiling produced only {m} tiles for {self.active_robot_count} active robots; "
                "a nonempty connected partition is impossible."
            )
        cell_owner = np.full(self.width * self.height, -1, dtype=np.int32)
        for tid, tile in enumerate(self.global_tiles):
            for x, y in self._tile_cells(tile):
                u = x * self.height + y
                if cell_owner[u] >= 0:
                    raise AssertionError("Global tiling overlaps before partitioning")
                cell_owner[u] = tid
        if np.any(cell_owner[self.free_flat] < 0):
            raise AssertionError("Global tiling does not exactly cover all free cells")
        if np.any(cell_owner[np.flatnonzero(self.obstacles.ravel())] >= 0):
            raise AssertionError("Global tiling covers an obstacle")

        adjacency = [set() for _ in range(m)]
        for u in self.free_flat:
            a = int(cell_owner[u])
            for v in _neighbors(int(u), self.width, self.height):
                b = int(cell_owner[v])
                if b >= 0 and b != a:
                    adjacency[a].add(b)
                    adjacency[b].add(a)
        self.tile_adjacency = adjacency

        # Build immutable center/portal edge geometry once.  Iterate tiles/cells
        # in the same order as _tile_geodesic_metric historically did, preserving
        # strict '<' portal tie behavior exactly.
        self._global_tile_index = {
            (t[0], t[1], t[2], t[3]): i for i, t in enumerate(self.global_tiles)
        }
        self._global_centers = [
            np.array((t[0] + t[2] / 2, t[1] + t[3] / 2), dtype=float)
            for t in self.global_tiles
        ]
        owner_dict: Dict[Cell, int] = {}
        for i, t in enumerate(self.global_tiles):
            x, y, w, h, _ = t
            for px in range(x, x + w):
                for py in range(y, y + h):
                    owner_dict[(px, py)] = i
        edge_weights: Dict[Tuple[int, int], float] = {}
        edge_portals: Dict[Tuple[int, int], np.ndarray] = {}
        for (x, y), i in owner_dict.items():
            for nb in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
                j = owner_dict.get(nb)
                if j is None or i == j:
                    continue
                e = (min(i, j), max(i, j))
                if nb[0] == x + 1:
                    portal = np.array((x + 1.0, y + 0.5))
                elif nb[0] == x - 1:
                    portal = np.array((x + 0.0, y + 0.5))
                elif nb[1] == y + 1:
                    portal = np.array((x + 0.5, y + 1.0))
                else:
                    portal = np.array((x + 0.5, y + 0.0))
                ci, cj = self._global_centers[i], self._global_centers[j]
                weight = float(
                    math.hypot(float(ci[0] - portal[0]), float(ci[1] - portal[1]))
                    + math.hypot(float(portal[0] - cj[0]), float(portal[1] - cj[1]))
                )
                if e not in edge_weights or weight < edge_weights[e]:
                    edge_weights[e] = weight
                    edge_portals[e] = portal
        self._global_edge_weights = edge_weights
        self._global_edge_portals = edge_portals

        # Exact cover of connected free space must induce a connected tile graph.
        seen = {0}
        q = deque([0])
        while q:
            u = q.popleft()
            for v in adjacency[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        if len(seen) != m:
            raise AssertionError("Global tile adjacency graph is disconnected")

    def _tile_graph_matrix(self) -> sp.csr_matrix:
        rows, cols, data = [], [], []
        centers = [
            np.array((t[0] + t[2] / 2.0, t[1] + t[3] / 2.0), dtype=float)
            for t in self.global_tiles
        ]
        for u, nbs in enumerate(self.tile_adjacency):
            for v in nbs:
                if u < v:
                    dx = float(centers[u][0] - centers[v][0])
                    dy = float(centers[u][1] - centers[v][1])
                    w = float(math.hypot(dx, dy))
                    rows.extend((u, v)); cols.extend((v, u)); data.extend((w, w))
        return sp.csr_matrix((data, (rows, cols)), shape=(len(centers), len(centers)))

    def _choose_tile_partition_seeds(self) -> List[int]:
        """Farthest-point graph seeds for active robots only.

        The returned list is indexed by robot id; inactive robots carry seed -1.
        """
        m = len(self.global_tiles)
        centers = np.asarray([
            (t[0] + t[2] / 2.0, t[1] + t[3] / 2.0) for t in self.global_tiles
        ], dtype=float)
        areas = np.asarray([t[2] * t[3] for t in self.global_tiles], dtype=float)
        workspace_center = np.average(centers, axis=0, weights=areas)
        first = int(np.argmin(np.linalg.norm(centers - workspace_center, axis=1)))

        seeds = [-1] * self.k
        first_rid = self.active_robot_ids[0]
        seeds[first_rid] = first
        chosen = [first]
        graph = self._tile_graph_matrix()
        min_dist = np.full(m, np.inf)
        for rid in self.active_robot_ids[1:]:
            d = np.asarray(sparse_dijkstra(graph, indices=chosen[-1]), dtype=float)
            min_dist = np.minimum(min_dist, d)
            min_dist[chosen] = -np.inf
            nxt = int(np.argmax(min_dist))
            if not np.isfinite(min_dist[nxt]):
                raise AssertionError("Cannot place connected partition seeds")
            seeds[rid] = nxt
            chosen.append(nxt)
        self.tile_partition_seeds = seeds
        return seeds

    def _balanced_connected_growth(self, seeds: Sequence[int]) -> np.ndarray:
        """Compact connected growth with covered area proportional to robot weights.

        This is only an initializer for the true weighted TSP-CV local search.
        Each robot targets ``total_free_area * robot_weight``.  Growth remains
        adjacent to the robot's current region, so connectivity is preserved by
        construction.  Equal weights reduce to the original equal-area behavior.
        """
        m = len(self.global_tiles)
        centers = np.asarray([
            (t[0] + t[2] / 2.0, t[1] + t[3] / 2.0) for t in self.global_tiles
        ], dtype=float)
        areas = np.asarray([t[2] * t[3] for t in self.global_tiles], dtype=float)
        targets = areas.sum() * self.robot_weights
        owners = np.full(m, -1, dtype=np.int16)
        loads = np.zeros(self.k, dtype=float)
        best_cost = np.full((self.k, m), np.inf)
        frontiers: List[List[Tuple[float, int]]] = [[] for _ in range(self.k)]

        for rid in self.active_robot_ids:
            seed = int(seeds[rid])
            owners[seed] = rid
            loads[rid] = areas[seed]
            best_cost[rid, seed] = 0.0
        for rid in self.active_robot_ids:
            seed = int(seeds[rid])
            for v in self.tile_adjacency[seed]:
                if owners[v] < 0:
                    dx = float(centers[seed, 0] - centers[v, 0])
                    dy = float(centers[seed, 1] - centers[v, 1])
                    cost = float(math.hypot(dx, dy))
                    if cost < best_cost[rid, v]:
                        best_cost[rid, v] = cost
                        heapq.heappush(frontiers[rid], (cost, int(v)))

        remaining = int(np.count_nonzero(owners < 0))
        while remaining:
            active = []
            for rid in self.active_robot_ids:
                while frontiers[rid] and owners[frontiers[rid][0][1]] >= 0:
                    heapq.heappop(frontiers[rid])
                if frontiers[rid]:
                    active.append((loads[rid] / max(float(targets[rid]), 1e-12), rid))
            if not active:
                raise AssertionError("Connected balanced growth cannot reach all global tiles")
            _, rid = min(active)
            cost, u = heapq.heappop(frontiers[rid])
            if owners[u] >= 0:
                continue
            owners[u] = rid
            loads[rid] += areas[u]
            remaining -= 1
            for v in self.tile_adjacency[u]:
                if owners[v] >= 0:
                    continue
                dx = float(centers[u, 0] - centers[v, 0])
                dy = float(centers[u, 1] - centers[v, 1])
                new_cost = cost + float(math.hypot(dx, dy))
                if new_cost + 1e-12 < best_cost[rid, v]:
                    best_cost[rid, v] = new_cost
                    heapq.heappush(frontiers[rid], (new_cost, int(v)))
        return owners

    def _tile_owner_connected(self, owners: np.ndarray, rid: int) -> bool:
        nodes = np.flatnonzero(owners == rid)
        if not len(nodes):
            return False
        start = int(nodes[0])
        seen = {start}
        q = deque([start])
        while q:
            u = q.popleft()
            for v in self.tile_adjacency[u]:
                if owners[v] == rid and v not in seen:
                    seen.add(v)
                    q.append(v)
        return len(seen) == len(nodes)

    def _can_transfer_tile(self, owners: np.ndarray, tid: int, receiver: int) -> bool:
        donor = int(owners[tid])
        if donor == receiver:
            return False
        if not any(owners[v] == receiver for v in self.tile_adjacency[tid]):
            return False

        # During one partition iteration the owner vector is immutable while many
        # candidate tiles are queried.  A donor remains connected after deleting
        # tid iff tid is not an articulation vertex of that donor's induced graph.
        # Cache the exact articulation set once per donor/owner-vector instead of
        # repeating a full BFS for every receiver candidate.
        if self._transfer_cache_owners is not owners:
            self._transfer_cache_owners = owners
            self._transfer_articulations = {}
            self._transfer_counts = {}

        if donor not in self._transfer_articulations:
            donor_nodes = np.flatnonzero(owners == donor)
            count = int(len(donor_nodes))
            self._transfer_counts[donor] = count
            if count <= 2:
                self._transfer_articulations[donor] = set()
            else:
                m = len(owners)
                disc = np.full(m, -1, dtype=np.int32)
                low = np.empty(m, dtype=np.int32)
                parent = np.full(m, -1, dtype=np.int32)
                child_count = np.zeros(m, dtype=np.int16)
                articulation: set[int] = set()
                tick = 0
                donor_mask = owners == donor
                for root_value in donor_nodes:
                    root = int(root_value)
                    if disc[root] >= 0:
                        continue
                    disc[root] = low[root] = tick
                    tick += 1
                    stack = [(root, iter(self.tile_adjacency[root]))]
                    while stack:
                        u, children = stack[-1]
                        try:
                            v = int(next(children))
                        except StopIteration:
                            stack.pop()
                            p = int(parent[u])
                            if p < 0:
                                if child_count[u] > 1:
                                    articulation.add(u)
                            else:
                                if low[u] < low[p]:
                                    low[p] = low[u]
                                if parent[p] >= 0 and low[u] >= disc[p]:
                                    articulation.add(p)
                            continue
                        if not donor_mask[v]:
                            continue
                        if disc[v] < 0:
                            parent[v] = u
                            child_count[u] += 1
                            disc[v] = low[v] = tick
                            tick += 1
                            stack.append((v, iter(self.tile_adjacency[v])))
                        elif v != parent[u] and disc[v] < low[u]:
                            low[u] = disc[v]
                self._transfer_articulations[donor] = articulation

        if self._transfer_counts[donor] <= 1:
            return False
        return tid not in self._transfer_articulations[donor]

    def _apply_tile_owners(self, owners: np.ndarray) -> None:
        self.tile_owners = np.asarray(owners, dtype=np.int16).copy()
        self.tiles = [
            (t[0], t[1], t[2], t[3], int(self.tile_owners[i]))
            for i, t in enumerate(self.global_tiles)
        ]
        a = np.full(self.map_shape, -1, dtype=np.int16)
        for tile in self.tiles:
            x, y, w, h, rid = tile
            a[x:x+w, y:y+h] = rid
        if np.any(a[~self.obstacles] < 0) or np.any(a[self.obstacles] >= 0):
            raise AssertionError("Tile-owner assignment failed to reconstruct free-space partition")
        self.assignments = a
        # Visualization-only region representatives, never route depots/roots.
        self.centroids_xy = []
        for rid in range(self.k):
            cells = np.argwhere(a == rid)
            if not len(cells):
                self.centroids_xy.append(None)
                continue
            c = np.mean(cells, axis=0)
            self.centroids_xy.append((int(round(c[0])), int(round(c[1]))))

    def _tile_geodesic_dist_only(self, tiles: List[Tile]) -> Tuple[List[np.ndarray], np.ndarray]:
        """Distance-only twin of _tile_geodesic_metric for partition trial routes."""
        if self._global_tile_index and len(self._global_centers) == len(self.global_tiles):
            global_ids = [self._global_tile_index[(t[0], t[1], t[2], t[3])] for t in tiles]
            local_of = {gid: i for i, gid in enumerate(global_ids)}
            centers = [self._global_centers[gid] for gid in global_ids]
            rows, cols, data = [], [], []
            for gu in global_ids:
                u = local_of[gu]
                for gv in self.tile_adjacency[gu]:
                    if gv <= gu:
                        continue
                    v = local_of.get(gv)
                    if v is None:
                        continue
                    weight = self._global_edge_weights[(gu, gv)]
                    rows.extend((u, v)); cols.extend((v, u)); data.extend((weight, weight))
            graph = sp.csr_matrix((data, (rows, cols)), shape=(len(tiles), len(tiles)))
            dist = np.asarray(sparse_dijkstra(graph, directed=False, return_predecessors=False), dtype=float)
            if not np.isfinite(dist).all():
                raise AssertionError("Tile adjacency roadmap is disconnected")
            return centers, dist
        centers, dist, _, _ = self._tile_geodesic_metric(tiles)
        return centers, dist

    def _route_trial_for_tiles(self, tiles: List[Tile]) -> dict:
        """Exact partition-trial route objective without final geometric expansion."""
        if len(tiles) <= 1:
            return self._route_cycle_for_tiles(tiles)
        centers, d = self._tile_geodesic_dist_only(tiles)
        center_arr = np.asarray(centers)
        start = int(np.argmin(np.linalg.norm(center_arr - center_arr.mean(axis=0), axis=1)))

        nn_order = [start]
        unused = np.ones(len(tiles), dtype=bool)
        unused[start] = False
        while unused.any():
            costs = d[nn_order[-1]].copy()
            costs[~unused] = np.inf
            nxt = int(np.argmin(costs))
            nn_order.append(nxt)
            unused[nxt] = False
        preorder, _, mst_weight = self._metric_mst_orders(d, start)
        passes = 2 if len(tiles) > 1000 else 3
        order_a = self._two_opt(nn_order.copy(), d, passes=passes)
        order_b = self._two_opt(preorder.copy(), d, passes=passes)
        length_a = self._tour_length(order_a, d)
        length_b = self._tour_length(order_b, d)
        if length_a == length_b:
            # Preserve the original secondary mission-time tie-break exactly.
            return self._route_cycle_for_tiles(tiles)
        if length_a < length_b:
            order, length = order_a, length_a
        else:
            order, length = order_b, length_b

        marginal_costs: Dict[Tile, float] = {}
        for pos, tile_index in enumerate(order):
            previous = order[(pos - 1) % len(order)]
            following = order[(pos + 1) % len(order)]
            detour = d[previous, tile_index] + d[tile_index, following] - d[previous, following]
            marginal_costs[tiles[tile_index]] = float(max(0.0, detour))
        return {
            "order": order,
            "centers": [centers[i] for i in order],
            "path": [],
            "length": float(length),
            "quarter_turns": 0.0,
            "turn_time": 0.0,
            "mission_time": float(self.service_time_per_tile * len(tiles) + length / self.robot_speed),
            "double_tree_bound": float(2.0 * mst_weight),
            "tile_marginal_costs": marginal_costs,
            "routing_mode": "depot_free_metric_trial",
        }

    def _route_cycle_for_tiles(self, tiles: List[Tile]) -> dict:
        if not tiles:
            return {
                "order": [], "centers": [], "path": [], "length": 0.0,
                "quarter_turns": 0.0, "turn_time": 0.0, "mission_time": 0.0,
                "double_tree_bound": 0.0, "tile_marginal_costs": {},
                "routing_mode": "depot_free_metric_expansion",
            }
        if len(tiles) == 1:
            c = np.array((tiles[0][0] + tiles[0][2] / 2.0,
                          tiles[0][1] + tiles[0][3] / 2.0), dtype=float)
            return {
                "order": [0], "centers": [c], "path": [c.tolist()], "length": 0.0,
                "quarter_turns": 0.0, "turn_time": 0.0,
                "mission_time": float(self.service_time_per_tile),
                "double_tree_bound": 0.0, "tile_marginal_costs": {tiles[0]: 0.0},
                "routing_mode": "depot_free_singleton",
            }

        centers, d, predecessors, edge_portals = self._tile_geodesic_metric(tiles)
        center_arr = np.asarray(centers)
        start = int(np.argmin(np.linalg.norm(center_arr - center_arr.mean(axis=0), axis=1)))

        nn_order = [start]
        unused = np.ones(len(tiles), dtype=bool)
        unused[start] = False
        while unused.any():
            costs = d[nn_order[-1]].copy()
            costs[~unused] = np.inf
            nxt = int(np.argmin(costs))
            nn_order.append(nxt)
            unused[nxt] = False
        preorder, _, mst_weight = self._metric_mst_orders(d, start)
        passes = 2 if len(tiles) > 1000 else 3
        candidates = [
            self._two_opt(nn_order.copy(), d, passes=passes),
            self._two_opt(preorder.copy(), d, passes=passes),
        ]
        evaluated = []
        for order in candidates:
            length = self._tour_length(order, d)
            path = self._expand_metric_tour(order, predecessors, centers, edge_portals)
            quarter_turns, turn_time = self._path_turn_metrics(path)
            mission = (
                self.service_time_per_tile * len(tiles)
                + length / self.robot_speed
                + turn_time
            )
            evaluated.append((mission, length, order, path, quarter_turns, turn_time))
        mission, length, order, path, quarter_turns, turn_time = min(
            evaluated, key=lambda z: (z[1], z[0])
        )

        marginal_costs: Dict[Tile, float] = {}
        for pos, tile_index in enumerate(order):
            previous = order[(pos - 1) % len(order)]
            following = order[(pos + 1) % len(order)]
            detour = d[previous, tile_index] + d[tile_index, following] - d[previous, following]
            marginal_costs[tiles[tile_index]] = float(max(0.0, detour))
        return {
            "order": order,
            "centers": [centers[i] for i in order],
            "path": path,
            "length": float(length),
            "quarter_turns": float(quarter_turns),
            "turn_time": float(turn_time),
            "mission_time": float(mission),
            "double_tree_bound": float(2.0 * mst_weight),
            "tile_marginal_costs": marginal_costs,
            "routing_mode": "depot_free_metric_expansion",
        }

    def _legacy_portal_shortcut_path(
        self,
        raw_path: Sequence[Sequence[float]],
        rid: int,
        center_keys: set,
    ) -> Tuple[List[List[float]], int, int, List[Tuple[List[float], List[float]]]]:
        """Exactly reproduce v2.2.0 immutable center->portal->center deletion."""
        raw = self._dedupe_polyline(raw_path)
        remove_portals: set[int] = set()
        segments: List[Tuple[List[float], List[float]]] = []
        tested = 0
        for i in range(1, len(raw) - 1):
            prev_key = self._point_key(raw[i - 1])
            mid_key = self._point_key(raw[i])
            next_key = self._point_key(raw[i + 1])
            if prev_key not in center_keys or next_key not in center_keys:
                continue
            if mid_key in center_keys:
                continue
            tested += 1
            a, c = raw[i - 1], raw[i + 1]
            if self._segment_is_portal_shortcut_legal(a, c, rid):
                remove_portals.add(i)
                segments.append((list(a), list(c)))
        out = [p for i, p in enumerate(raw) if i not in remove_portals]
        return self._dedupe_polyline(out), tested, len(remove_portals), segments

    def _shortcut_redundant_route_points(
        self,
        path: Sequence[Sequence[float]],
        rid: int,
        center_keys: set,
        min_final_length: float = -math.inf,
    ) -> Tuple[List[List[float]], int, int, List[Tuple[List[float], List[float]]]]:
        """Geometry-constrained metric shortcutting of portals *and repeated centers*.

        A tile center is mandatory only until one occurrence remains.  Every other
        occurrence is a metric-TSP style repeated-vertex visit and may be removed
        when the newly exposed segment passes the exact closed-obstacle / region
        legality predicate.  Portal points are always optional.

        The stack formulation lets accepted deletions cascade, which is the key
        difference from v2.2.0's immutable center->portal->center test.
        """
        pts = self._dedupe_polyline(path)
        if len(pts) < 3:
            return pts, 0, 0, []

        center_count: Dict[Tuple[float, float], int] = {}
        for p in pts:
            key = self._point_key(p)
            if key in center_keys:
                center_count[key] = center_count.get(key, 0) + 1

        stack: List[List[float]] = []
        current_length = self._polyline_length(pts)
        tested = 0
        accepted = 0
        shortcut_segments: List[Tuple[List[float], List[float]]] = []
        for point in pts:
            stack.append(list(map(float, point)))
            while len(stack) >= 3:
                a, mid, c = stack[-3], stack[-2], stack[-1]
                mid_key = self._point_key(mid)
                # Portals are optional; a center is optional only when another
                # occurrence remains somewhere in the closed tour.
                if mid_key in center_keys and center_count.get(mid_key, 0) <= 1:
                    break
                old_len = (
                    float(np.linalg.norm(np.asarray(mid) - np.asarray(a)))
                    + float(np.linalg.norm(np.asarray(c) - np.asarray(mid)))
                )
                new_len = float(np.linalg.norm(np.asarray(c) - np.asarray(a)))
                if new_len > old_len + 1e-12:
                    break
                gain = old_len - new_len
                if current_length - gain < min_final_length - 1e-12:
                    break
                tested += 1
                if not self._segment_is_portal_shortcut_legal(a, c, rid):
                    break
                stack.pop(-2)
                current_length -= gain
                if mid_key in center_keys:
                    center_count[mid_key] -= 1
                accepted += 1
                shortcut_segments.append((list(a), list(c)))

        return self._dedupe_polyline(stack), tested, accepted, shortcut_segments

    def _postprocess_one_route_strict(self, rid: int) -> None:
        """Strict any-angle shortcutting while preserving every tile center visit.

        This implements two safe generalizations of the v2.2.0 rule:
        1. diagonal/any-angle edges are allowed whenever the exact segment test
           proves they do not touch a closed obstacle and stay in the robot region;
        2. repeated tile-center visits are treated like repeated vertices in metric
           TSP shortcutting and can be deleted while at least one visit remains.
        """
        route = self.routes.get(rid)
        if not route or route.get("strict_postprocessed", False):
            return

        raw_path = self._dedupe_polyline(route.get("path", []))
        raw_length = float(route.get("length", self._polyline_length(raw_path)))
        raw_mission = float(route.get("mission_time", 0.0))
        tiles = [t for t in self.tiles if t[4] == rid]
        center_keys = {
            self._point_key((t[0] + t[2] / 2.0, t[1] + t[3] / 2.0))
            for t in tiles
        }

        if len(raw_path) < 3:
            new_path, tested, accepted, shortcut_segments = raw_path, 0, 0, []
        else:
            # Stage 1 is bit-for-bit the v2.2.0 immutable portal rule.  Stage 2
            # operates only on that already-shortened route, so v2.3.2 can never
            # be longer than v2.2.0 for the same partition/TSP order.
            legacy_path, legacy_tested, legacy_accepted, legacy_segments = (
                self._legacy_portal_shortcut_path(raw_path, rid, center_keys)
            )
            forward = self._shortcut_redundant_route_points(legacy_path, rid, center_keys)
            reversed_path = list(reversed(legacy_path))
            backward = self._shortcut_redundant_route_points(reversed_path, rid, center_keys)
            backward_path = list(reversed(backward[0]))
            forward_len = self._polyline_length(forward[0])
            backward_len = self._polyline_length(backward_path)
            if backward_len + 1e-12 < forward_len:
                new_path = backward_path
                extra_tested, extra_accepted, extra_segments = backward[1], backward[2], backward[3]
            else:
                new_path = forward[0]
                extra_tested, extra_accepted, extra_segments = forward[1], forward[2], forward[3]
            tested = legacy_tested + extra_tested
            accepted = legacy_accepted + extra_accepted
            shortcut_segments = legacy_segments + extra_segments
            if self._polyline_length(new_path) > self._polyline_length(legacy_path) + 1e-9:
                raise AssertionError("Repeated-center stage regressed v2.2.0 shortcut length")

            route["legacy_path"] = legacy_path
            route["legacy_length"] = float(self._polyline_length(legacy_path))
            route["legacy_shortcuts_tested"] = int(legacy_tested)
            route["legacy_shortcuts_accepted"] = int(legacy_accepted)
            route["legacy_shortcut_segments"] = legacy_segments
            route["unconstrained_path"] = self._dedupe_polyline(new_path)
            route["unconstrained_length"] = float(self._polyline_length(new_path))

        new_path = self._dedupe_polyline(new_path)
        if "legacy_path" not in route:
            route["legacy_path"] = self._dedupe_polyline(raw_path)
            route["legacy_length"] = float(self._polyline_length(raw_path))
            route["legacy_shortcuts_tested"] = 0
            route["legacy_shortcuts_accepted"] = 0
            route["legacy_shortcut_segments"] = []
            route["unconstrained_path"] = self._dedupe_polyline(new_path)
            route["unconstrained_length"] = float(self._polyline_length(new_path))
        # Hard coverage invariant: every assigned tile center must still appear.
        visited_keys = {self._point_key(p) for p in new_path}
        missing = center_keys - visited_keys
        if missing:
            raise AssertionError(f"Strict shortcut removed mandatory tile centers: {len(missing)}")

        # Exact re-check of every *final* segment, not just accepted shortcuts.
        for a, c in zip(new_path, new_path[1:]):
            if not self._segment_is_portal_shortcut_legal(a, c, rid):
                raise AssertionError("Final strict any-angle segment is illegal")

        new_length = self._polyline_length(new_path)
        if new_length > raw_length + 1e-9:
            raise AssertionError("Strict shortcutting increased route length")
        quarter_turns, turn_time = self._path_turn_metrics(new_path)
        mission_time = (
            self.service_time_per_tile * len(tiles)
            + new_length / self.robot_speed
            + turn_time
        )

        route["raw_path"] = raw_path
        route["raw_length"] = raw_length
        route["raw_mission_time"] = raw_mission
        route["path"] = new_path
        route["length"] = float(new_length)
        route["quarter_turns"] = float(quarter_turns)
        route["turn_time"] = float(turn_time)
        route["mission_time"] = float(mission_time)
        route["shortcut_gain"] = float(raw_length - new_length)
        route["shortcut_gain_pct"] = (
            100.0 * (raw_length - new_length) / raw_length
            if raw_length > 1e-12 else 0.0
        )
        route["portal_shortcuts_tested"] = int(tested)
        route["portal_shortcuts_accepted"] = int(accepted)
        route["shortcut_segments"] = shortcut_segments
        route["strict_postprocessed"] = True
        route["routing_mode"] = (
            str(route.get("routing_mode", "route")) + "+strict_repeated_center_shortcut"
        )

    def route_tsp(self, robot_ids: Optional[Sequence[int]] = None) -> Dict[int, dict]:
        ids = list(range(self.k)) if robot_ids is None else list(map(int, robot_ids))
        routes = {} if robot_ids is None else dict(self.routes)
        for rid in ids:
            tiles = [t for t in self.tiles if t[4] == rid]
            routes[rid] = self._route_cycle_for_tiles(tiles)
        self.routes = routes
        return routes

    def _normalized_path_loads(self, lengths: Sequence[float]) -> np.ndarray:
        """Return length/weight for active robots and 0 for inactive robots."""
        a = np.asarray(lengths, dtype=float)
        out = np.zeros(self.k, dtype=float)
        active = np.asarray(self.active_robot_ids, dtype=int)
        out[active] = a[active] / self.robot_weights[active]
        return out

    def _weighted_path_cv(self, lengths: Sequence[float]) -> float:
        normalized = self._normalized_path_loads(lengths)
        return coefficient_of_variation(normalized[list(self.active_robot_ids)])

    def _active_cv(self, values: Sequence[float]) -> float:
        a = np.asarray(values, dtype=float)
        return coefficient_of_variation(a[list(self.active_robot_ids)])

    def _candidate_transfer_proxy(
        self, tid: int, donor: int, receiver: int, lengths: np.ndarray
    ) -> float:
        tile = self.tiles[tid]
        donor_marginal = float(
            self.routes[donor].get("tile_marginal_costs", {}).get(tile, 0.0)
        )
        tc = np.array((tile[0] + tile[2] / 2.0, tile[1] + tile[3] / 2.0))
        receiver_neighbors = [
            v for v in self.tile_adjacency[tid] if self.tile_owners[v] == receiver
        ]
        insertion = min(
            2.0 * float(np.linalg.norm(
                tc - np.array((self.global_tiles[v][0] + self.global_tiles[v][2] / 2.0,
                               self.global_tiles[v][1] + self.global_tiles[v][3] / 2.0))
            ))
            for v in receiver_neighbors
        )
        approximate = lengths.copy()
        approximate[donor] = max(0.0, approximate[donor] - donor_marginal)
        approximate[receiver] += insertion
        # Minimize dispersion of patrol length per unit robot weight.  Therefore
        # the optimum is length_i proportional to weight_i.
        return self._weighted_path_cv(approximate)

    def partition_tiles_min_tsp_cv(self) -> np.ndarray:
        """Connected partition minimizing CV of TSP length divided by robot weight."""
        seeds = self._choose_tile_partition_seeds()
        owners = self._balanced_connected_growth(seeds)
        self._apply_tile_owners(owners)
        self.route_tsp()
        lengths = np.asarray([self.routes[r]["length"] for r in range(self.k)], dtype=float)
        normalized_loads = self._normalized_path_loads(lengths)
        current_cv = self._weighted_path_cv(lengths)
        # Keep the legacy raw path CV for reporting, and separately track the
        # weighted objective used by partition optimization.
        self.initial_path_cv = float(self._active_cv(lengths))
        self.initial_weighted_path_cv = float(current_cv)
        self.initial_path_lengths = lengths.tolist()
        self.tile_partition_trace = []

        max_iterations = max(1, min(80, int(self.partition_iterations)))
        candidate_limit = max(2, int(self.candidate_limit))
        exact_limit = min(3, candidate_limit)

        for iteration in range(max_iterations):
            boundary_candidates = []
            normalized_loads = self._normalized_path_loads(lengths)
            mean_normalized_load = float(normalized_loads[list(self.active_robot_ids)].mean())
            for tid, donor in enumerate(owners):
                donor = int(donor)
                # A transfer is promising when it moves work from a robot with a
                # larger length/weight ratio to an adjacent robot with a smaller
                # ratio.  Exact weighted-CV acceptance below remains authoritative.
                if normalized_loads[donor] + 1e-9 < mean_normalized_load:
                    continue
                receivers = sorted({
                    int(owners[v]) for v in self.tile_adjacency[tid]
                    if owners[v] != donor
                    and normalized_loads[int(owners[v])] < normalized_loads[donor] - 1e-9
                })
                for receiver in receivers:
                    if not self._can_transfer_tile(owners, tid, receiver):
                        continue
                    proxy = self._candidate_transfer_proxy(tid, donor, receiver, lengths)
                    boundary_candidates.append((proxy, tid, donor, receiver))
            if not boundary_candidates:
                break
            boundary_candidates.sort(key=lambda z: z[0])
            shortlisted = boundary_candidates[:candidate_limit]

            best = None
            for _, tid, donor, receiver in shortlisted[:exact_limit]:
                trial = owners.copy()
                trial[tid] = receiver
                donor_tiles = [
                    (t[0], t[1], t[2], t[3], donor)
                    for i, t in enumerate(self.global_tiles) if trial[i] == donor
                ]
                receiver_tiles = [
                    (t[0], t[1], t[2], t[3], receiver)
                    for i, t in enumerate(self.global_tiles) if trial[i] == receiver
                ]
                donor_route = self._route_trial_for_tiles(donor_tiles)
                receiver_route = self._route_trial_for_tiles(receiver_tiles)
                trial_lengths = lengths.copy()
                trial_lengths[donor] = donor_route["length"]
                trial_lengths[receiver] = receiver_route["length"]
                trial_normalized_loads = self._normalized_path_loads(trial_lengths)
                trial_cv = self._weighted_path_cv(trial_lengths)
                key = (
                    trial_cv,
                    float(trial_normalized_loads[list(self.active_robot_ids)].max(initial=0.0)),
                    float(trial_lengths.sum()),
                )
                if best is None or key < best[0]:
                    best = (key, tid, donor, receiver, trial, trial_lengths, donor_route, receiver_route)

            if best is None or best[0][0] >= current_cv - 1e-9:
                break

            _, tid, donor, receiver, owners, lengths, donor_route, receiver_route = best
            self._apply_tile_owners(owners)
            self.routes[donor] = donor_route
            self.routes[receiver] = receiver_route
            current_cv = self._weighted_path_cv(lengths)
            self.tile_partition_trace.append({
                "iteration": iteration + 1,
                "moved_tile": int(tid),
                "donor": int(donor),
                "receiver": int(receiver),
                "weighted_path_cv": float(current_cv),
                "path_cv": float(self._active_cv(lengths)),
                "path_lengths": lengths.tolist(),
                "normalized_path_loads": self._normalized_path_loads(lengths).tolist(),
            })

        # Rebuild once to guarantee routes and immutable tile tuples exactly match
        # the final owner vector after all accepted moves.
        self._apply_tile_owners(owners)
        self.route_tsp()
        return self.assignments

    def _validate_depot_free_routes(self) -> Tuple[bool, bool]:
        collision_free = True
        cycle_closed = True
        for rid in range(self.k):
            if rid in self.inactive_robot_ids:
                continue
            route = self.routes.get(rid, {})
            path = [np.asarray(p, dtype=float) for p in route.get("path", [])]
            if not path or not np.allclose(path[0], path[-1]):
                # A singleton tour is represented by one center and is trivially closed.
                if len(path) != 1:
                    cycle_closed = False
            for a, b in zip(path, path[1:]):
                steps = max(1, int(math.ceil(float(np.linalg.norm(b - a)) * 10.0)))
                for alpha in np.linspace(0.0, 1.0, steps + 1):
                    if not self._point_in_region_closure((1.0 - alpha) * a + alpha * b, rid):
                        collision_free = False
                        break
                if not collision_free:
                    break
        return collision_free, cycle_closed

    def validate(self) -> dict:
        covered = np.zeros(self.map_shape, dtype=np.int16)
        for x, y, w, h, rid in self.tiles:
            if not (self.assignments[x:x+w, y:y+h] == rid).all():
                raise AssertionError("A global tile crosses its final partition boundary")
            covered[x:x+w, y:y+h] += 1
        if not np.all(covered[~self.obstacles] == 1) or np.any(covered[self.obstacles]):
            raise AssertionError("Global tiling is not an exact cover")
        connected = [self._tile_owner_connected(self.tile_owners, rid) for rid in self.active_robot_ids]
        if not all(connected):
            raise AssertionError("A final active-robot tile partition is disconnected")
        if any(np.any(self.tile_owners == rid) for rid in self.inactive_robot_ids):
            raise AssertionError("An inactive robot received patrol tiles")
        route_collision_free, route_closed = self._validate_depot_free_routes()
        if not route_collision_free:
            raise AssertionError("A TSP route leaves its assigned connected region")
        if not route_closed:
            raise AssertionError("A depot-free patrol route is not a closed cycle")

        # Do not rely on sampling for newly-created shortcut segments: re-run
        # the exact closed-cell/corner-only predicate used for acceptance.
        shortcut_legal = all(
            self._segment_is_portal_shortcut_legal(a, b, rid)
            for rid in range(self.k)
            for a, b in self.routes.get(rid, {}).get("shortcut_segments", [])
        )
        if not shortcut_legal:
            raise AssertionError("A stored center-to-center portal shortcut is illegal")

        # Audit the final polyline against closed obstacle cells.
        strict_obstacle_clear = self._validate_strict_obstacle_routes()
        if self.strict_route_postprocess and not strict_obstacle_clear:
            raise AssertionError("A final route segment touches a closed obstacle cell")

        all_centers_visited = True
        for rid in self.active_robot_ids:
            required = {
                self._point_key((t[0] + t[2] / 2.0, t[1] + t[3] / 2.0))
                for t in self.tiles if t[4] == rid
            }
            visited = {
                self._point_key(p) for p in self.routes.get(rid, {}).get("path", [])
            }
            if not required.issubset(visited):
                all_centers_visited = False
                break
        if not all_centers_visited:
            raise AssertionError("A final route does not visit every assigned tile center")
        return {
            "exact_cover": True,
            "connected_regions": True,
            "inactive_robots_unassigned": True,
            "route_collision_free": True,
            "route_closed": True,
            "shortcut_segments_legal": bool(shortcut_legal),
            "strict_obstacle_no_touch": bool(strict_obstacle_clear),
            "all_tile_centers_visited": bool(all_centers_visited),
        }

    def solve(self, method: str = "tile_first") -> dict:
        if method != "tile_first":
            raise ValueError("Only method='tile_first' exists in v2.3.2")
        solve_started = time.perf_counter()
        times = StageTimes()

        t = time.perf_counter()
        self.tile_global_hybrid()
        times.tiling = time.perf_counter() - t

        t = time.perf_counter()
        self.partition_tiles_min_tsp_cv()
        times.partition = time.perf_counter() - t

        t = time.perf_counter()
        # partition_tiles_min_tsp_cv already leaves exact final raw routes; reroute
        # once here so timing/outputs have the same clear three-stage semantics.
        self.route_tsp()
        # Final post-process.  Partition, global tiling and TSP order are all
        # frozen before this point.  Only legal raw center->portal->center points
        # are removed; no robot start/depot participates in the test.
        if self.strict_route_postprocess:
            self.postprocess_routes_strict()
        times.routing = time.perf_counter() - t

        t = time.perf_counter()
        valid = self.validate()
        times.validation = time.perf_counter() - t

        sizes = self._partition_sizes()
        tile_loads = np.array([sum(t[4] == rid for t in self.tiles) for rid in range(self.k)])
        path_loads = np.array([self.routes[rid]["length"] for rid in range(self.k)], dtype=float)
        raw_path_loads = np.array([
            self.routes[rid].get("raw_length", self.routes[rid]["length"])
            for rid in range(self.k)
        ], dtype=float)
        legacy_path_loads = np.array([
            self.routes[rid].get("legacy_length", self.routes[rid]["length"])
            for rid in range(self.k)
        ], dtype=float)
        unconstrained_path_loads = np.array([
            self.routes[rid].get("unconstrained_length", self.routes[rid]["length"])
            for rid in range(self.k)
        ], dtype=float)
        turn_times = np.array([self.routes[rid]["turn_time"] for rid in range(self.k)], dtype=float)
        mission_times = np.array([self.routes[rid]["mission_time"] for rid in range(self.k)], dtype=float)
        normalized_path_loads = self._normalized_path_loads(path_loads)
        target_partition_sizes = len(self.free_flat) * self.robot_weights
        actual_path_shares = (
            path_loads / path_loads.sum()
            if float(path_loads.sum()) > 1e-12
            else np.zeros(self.k, dtype=float)
        )
        result = {
            "method": method,
            "map_shape": [self.width, self.height],
            "map_width": self.width,
            "map_height": self.height,
            "robots": self.k,
            "active_robots": self.active_robot_count,
            "active_robot_ids": list(self.active_robot_ids),
            "inactive_robot_ids": list(self.inactive_robot_ids),
            "robot_weights": self.robot_weights.tolist(),
            "seed": self.seed,
            "free_cells": int(len(self.free_flat)),
            "target_partition_sizes": target_partition_sizes.tolist(),
            "partition_sizes": sizes.tolist(),
            "partition_cv": self._active_cv(sizes),
            "tile_count": len(self.tiles),
            "tile_counts": tile_loads.tolist(),
            "tile_cv": self._active_cv(tile_loads),
            "path_length": float(path_loads.sum()),
            "path_lengths": path_loads.tolist(),
            "path_shares": actual_path_shares.tolist(),
            "path_cv": self._active_cv(path_loads),
            "normalized_path_loads": normalized_path_loads.tolist(),
            "weighted_path_cv": self._weighted_path_cv(path_loads),
            "initial_path_cv": float(getattr(self, "initial_path_cv", coefficient_of_variation(path_loads))),
            "initial_weighted_path_cv": float(getattr(
                self, "initial_weighted_path_cv", self._weighted_path_cv(path_loads)
            )),
            "initial_path_lengths": list(getattr(self, "initial_path_lengths", path_loads.tolist())),
            "turn_times": turn_times.tolist(),
            "turn_time_total": float(turn_times.sum()),
            "mission_times": mission_times.tolist(),
            "makespan": float(mission_times.max(initial=0.0)),
            "initial_makespan": float(max(self.initial_path_lengths, default=0.0)),
            "mission_time_total": float(mission_times.sum()),
            "mission_cv": self._active_cv(mission_times),
            "service_time_per_tile": self.service_time_per_tile,
            "tile_shapes": self.tile_shapes,
            "robot_speed": self.robot_speed,
            "turn_time_90": self.turn_time_90,
            "strict_route_postprocess": self.strict_route_postprocess,
            "raw_path_lengths": raw_path_loads.tolist(),
            "raw_path_cv": self._active_cv(raw_path_loads),
            "legacy_final_path_lengths": legacy_path_loads.tolist(),
            "legacy_final_path_length": float(legacy_path_loads.sum()),
            "legacy_final_path_cv": self._active_cv(legacy_path_loads),
            "unconstrained_path_lengths": unconstrained_path_loads.tolist(),
            "unconstrained_path_length": float(unconstrained_path_loads.sum()),
            "unconstrained_path_cv": self._active_cv(unconstrained_path_loads),
            # Kept for v2.3.0 result-schema compatibility.  v2.3.2 never rolls
            # back legal shortcuts for CV/fairness reasons.
            "balance_guard_applied": False,
            "shortcut_policy": "unconstrained_strict",
            "shortcut_gains": [
                float(self.routes[rid].get("shortcut_gain", 0.0))
                for rid in range(self.k)
            ],
            "shortcut_gain_pcts": [
                float(self.routes[rid].get("shortcut_gain_pct", 0.0))
                for rid in range(self.k)
            ],
            "portal_shortcuts_tested": [
                int(self.routes[rid].get("portal_shortcuts_tested", 0))
                for rid in range(self.k)
            ],
            "portal_shortcuts_accepted": [
                int(self.routes[rid].get("portal_shortcuts_accepted", 0))
                for rid in range(self.k)
            ],
            "accepted_tile_transfers": len(self.tile_partition_trace),
            "tile_partition_seeds": list(self.tile_partition_seeds),
            "tile_partition_trace": copy.deepcopy(self.tile_partition_trace),
            # Stable scalar timing interface.  The first three values are the
            # algorithm's core top-level stages; validation is intentionally
            # separate. ``core_time`` is their exact sum.
            "partition_time": float(times.partition),
            "tiling_time": float(times.tiling),
            "routing_time": float(times.routing),
            "validation_time": float(times.validation),
            "core_time": float(times.core_total),
            "measured_stage_time": float(times.measured_total),
            **valid,
        }
        # Measure the whole solve call through validation and result construction.
        # This is intentionally wall-clock time rather than merely the sum of the
        # three core stages, so callers get a true end-to-end runtime.
        total_time = time.perf_counter() - solve_started
        result["stage_times"] = {
            "tiling": float(times.tiling),
            "partition": float(times.partition),
            "routing": float(times.routing),
            "validation": float(times.validation),
            "core": float(times.core_total),
            "total": 0.0,
        }
        # Refresh once after the timing dictionary itself has been assembled so
        # total_time is as close as possible to the full solve() wall time.
        total_time = time.perf_counter() - solve_started
        result["total_time"] = float(total_time)
        result["stage_times"]["total"] = float(total_time)
        result["timing_overhead"] = float(max(0.0, total_time - times.measured_total))
        return result


FACTMCCA = TileFirstMCPP
