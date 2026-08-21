"""Scalable three-stage multi-robot coverage planner.

The module contains a faithful dependency-free baseline for the uploaded
algorithm and an upgraded planner:

1. capacity-balanced connected geodesic power partition + boundary repair;
2. multi-start anisotropic tiling + exact, time-limited MILP neighborhoods;
3. obstacle-safe all-pairs grid distances + NN/2-opt TSP tour.

Coordinates follow the uploaded program: a cell is (x, y), and a tile is
(x, y, width, height, robot_id).  The public ``solve`` method returns metrics
and all geometry required for plotting or downstream robot execution.
"""

from __future__ import annotations

from collections import deque
import copy
from dataclasses import dataclass
import heapq
import itertools
import math
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import (
    depth_first_order,
    dijkstra as sparse_dijkstra,
    minimum_spanning_tree,
)
from scipy.optimize import Bounds, LinearConstraint, linprog, milp


Cell = Tuple[int, int]
Tile = Tuple[int, int, int, int, int]
TILE_SHAPES = ((4, 4), (4, 2), (2, 4), (2, 2), (2, 1), (1, 2))


@dataclass
class StageTimes:
    partition: float = 0.0
    tiling: float = 0.0
    routing: float = 0.0

    @property
    def total(self) -> float:
        return self.partition + self.tiling + self.routing


def coefficient_of_variation(values: Sequence[float]) -> float:
    a = np.asarray(values, dtype=float)
    return float(np.std(a) / np.mean(a)) if a.size and np.mean(a) else 0.0


def _neighbors(idx: int, n: int) -> Iterable[int]:
    x, y = divmod(idx, n)
    if x:
        yield idx - n
    if x + 1 < n:
        yield idx + n
    if y:
        yield idx - 1
    if y + 1 < n:
        yield idx + 1


def _largest_free_component(mask: np.ndarray) -> int:
    """Return the free-cell count when free space is connected, else -1."""
    n = mask.shape[0]
    free = np.flatnonzero(~mask.ravel())
    if not len(free):
        return 0
    seen = np.zeros(n * n, dtype=bool)
    q = deque([int(free[0])])
    seen[free[0]] = True
    count = 0
    while q:
        u = q.popleft()
        count += 1
        for v in _neighbors(u, n):
            if not mask.ravel()[v] and not seen[v]:
                seen[v] = True
                q.append(v)
    return count if count == len(free) else -1


def _free_components(mask: np.ndarray) -> List[List[int]]:
    n = mask.shape[0]
    unseen = set(map(int, np.flatnonzero(~mask.ravel())))
    components: List[List[int]] = []
    while unseen:
        start = unseen.pop()
        comp = [start]
        q = deque([start])
        while q:
            u = q.popleft()
            for v in _neighbors(u, n):
                if v in unseen:
                    unseen.remove(v)
                    comp.append(v)
                    q.append(v)
        components.append(comp)
    return components


def make_random_map(n: int, obstacle_ratio: float, seed: int) -> np.ndarray:
    """Generate random-cell obstacles and retain a connected workspace.

    The uploaded generator can leave isolated free pockets that no seeded BFS
    can ever assign.  We preserve its random obstacle model, then classify
    unreachable free pockets as obstacles.  At 10% density this changes only a
    small number of cells and makes complete-coverage claims well-defined.
    """
    rng = np.random.default_rng(seed)
    obstacle_count = int(n * n * obstacle_ratio)
    mask = np.zeros(n * n, dtype=bool)
    mask[rng.choice(n * n, obstacle_count, replace=False)] = True
    mask = mask.reshape((n, n))
    components = _free_components(mask)
    if len(components) > 1:
        largest = set(max(components, key=len))
        flat = mask.ravel()
        for u in np.flatnonzero(~flat):
            if int(u) not in largest:
                flat[u] = True
    return mask


class HybridMCPP:
    def __init__(
        self,
        map_size: int,
        robot_num: int,
        obstacle_mask: Optional[np.ndarray] = None,
        obstacle_ratio: float = 0.10,
        seed: int = 42,
        partition_iterations: int = 120,
        tiling_time_limit: float = 8.0,
        balance_tolerance: float = 0.01,
        robot_starts: Optional[Sequence[Cell]] = None,
        service_time_per_tile: float = 1.0,
        robot_speed: float = 1.0,
        refinement_time_limit: float = 6.0,
        dual_guidance: bool = False,
        tile_shapes: Optional[Sequence[Tuple[int, int]]] = None,
        turn_time_90: float = 0.0,
        orientation_lifted_routing: bool = False,
        orientation_order_evaluations: int = 8,
        enable_orientation_beam: bool = False,
        enable_contracted_fact_beam: bool = False,
        refinement_max_iterations: int = 4,
        refinement_candidate_limit: int = 8,
    ) -> None:
        if map_size < 2 or robot_num < 1:
            raise ValueError("map_size >= 2 and robot_num >= 1 are required")
        self.n = int(map_size)
        self.k = int(robot_num)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.obstacles = (
            make_random_map(self.n, obstacle_ratio, seed)
            if obstacle_mask is None
            else np.asarray(obstacle_mask, dtype=bool).copy()
        )
        if self.obstacles.shape != (self.n, self.n):
            raise ValueError("obstacle_mask must have shape (map_size, map_size)")
        if _largest_free_component(self.obstacles) < 0:
            raise ValueError("The current implementation requires connected free space")
        if np.count_nonzero(~self.obstacles) < self.k:
            raise ValueError("Fewer free cells than robots")
        self.free_flat = np.flatnonzero(~self.obstacles.ravel())
        self.partition_iterations = partition_iterations
        self.tiling_time_limit = float(tiling_time_limit)
        self.balance_tolerance = float(balance_tolerance)
        if service_time_per_tile < 0 or robot_speed <= 0 or turn_time_90 < 0:
            raise ValueError(
                "service_time_per_tile >= 0, robot_speed > 0, and turn_time_90 >= 0 are required"
            )
        self.service_time_per_tile = float(service_time_per_tile)
        self.robot_speed = float(robot_speed)
        self.turn_time_90 = float(turn_time_90)
        self.orientation_lifted_routing = bool(orientation_lifted_routing)
        self.orientation_order_evaluations = max(0, int(orientation_order_evaluations))
        self.enable_orientation_beam = bool(enable_orientation_beam)
        self.enable_contracted_fact_beam = bool(enable_contracted_fact_beam)
        self.refinement_max_iterations = max(0, int(refinement_max_iterations))
        self.refinement_candidate_limit = max(1, int(refinement_candidate_limit))
        self.refinement_time_limit = max(0.0, float(refinement_time_limit))
        self.dual_guidance = bool(dual_guidance)
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
        self._provided_start_flats: Optional[List[int]] = None
        if robot_starts is not None:
            if len(robot_starts) != self.k:
                raise ValueError("robot_starts must contain one cell per robot")
            starts = []
            for x, y in robot_starts:
                if not (0 <= x < self.n and 0 <= y < self.n) or self.obstacles[x, y]:
                    raise ValueError("Every robot start must be a free in-map cell")
                starts.append(int(x) * self.n + int(y))
            if len(set(starts)) != len(starts):
                raise ValueError("Robot starts must be distinct")
            self._provided_start_flats = starts
        self.robot_start_flats = self._initial_seeds()
        self.robot_starts_xy = [divmod(s, self.n) for s in self.robot_start_flats]
        self.assignments: Optional[np.ndarray] = None
        self.tiles: List[Tile] = []
        self.routes: Dict[int, dict] = {}
        self.centroids_xy: List[Cell] = []
        self.baseline_root_repaired = False
        self.refinement_trace: List[dict] = []
        self.refinement_beams: List[dict] = []
        self.selected_refinement_beam = ""

    # ------------------------------- partition ----------------------------
    def _initial_seeds(self) -> List[int]:
        """Grid seeds compatible with the uploaded program, snapped to free cells."""
        if hasattr(self, "robot_start_flats"):
            return list(self.robot_start_flats)
        if self._provided_start_flats is not None:
            return list(self._provided_start_flats)
        n, k = self.n, self.k
        cols = int(math.ceil(math.sqrt(k)))
        step = n / math.sqrt(k) if k > 1 else n / 2
        seeds: List[int] = []
        used = set()
        for i in range(k):
            x = int(np.clip(round((i % cols) * step + step / 2 - 0.5), 0, n - 1))
            y = int(np.clip(round((i // cols) * step + step / 2 - 0.5), 0, n - 1))
            start = x * n + y
            if self.obstacles.ravel()[start] or start in used:
                q = deque([start])
                seen = {start}
                while q:
                    u = q.popleft()
                    if not self.obstacles.ravel()[u] and u not in used:
                        start = u
                        break
                    for v in _neighbors(u, n):
                        if v not in seen:
                            seen.add(v)
                            q.append(v)
            seeds.append(start)
            used.add(start)
        return seeds

    def _bfs_distance(self, source: int) -> np.ndarray:
        dist = np.full(self.n * self.n, np.inf)
        dist[source] = 0.0
        q = deque([source])
        flat_obs = self.obstacles.ravel()
        while q:
            u = q.popleft()
            nd = dist[u] + 1.0
            for v in _neighbors(u, self.n):
                if not flat_obs[v] and not np.isfinite(dist[v]):
                    dist[v] = nd
                    q.append(v)
        return dist

    def _seed_distances(self, seeds: Sequence[int]) -> np.ndarray:
        return np.vstack([self._bfs_distance(s)[self.free_flat] for s in seeds])

    def _rooted_power_assignment(self, seeds: Sequence[int], weights: np.ndarray) -> np.ndarray:
        """Return a rooted, connected weighted geodesic partition.

        Ordinary additive power diagrams may assign a seed cell to a different
        robot when its additive weight is sufficiently favorable.  A protected
        multi-source Dijkstra keeps every root fixed and grows each label only
        through cells already reached by that label.  Consequently every
        non-root cell has a same-label predecessor and each nonempty region is
        connected to its own root by construction.
        """
        root_owner = {int(s): rid for rid, s in enumerate(seeds)}
        distance = np.full(self.n * self.n, np.inf)
        owner = np.full(self.n * self.n, -1, dtype=np.int16)
        heap: List[Tuple[float, int, int]] = []
        for rid, source in enumerate(seeds):
            distance[source] = float(weights[rid])
            owner[source] = rid
            heapq.heappush(heap, (float(weights[rid]), rid, int(source)))
        flat_obs = self.obstacles.ravel()
        while heap:
            cost, rid, u = heapq.heappop(heap)
            if owner[u] != rid or cost > distance[u] + 1e-12:
                continue
            for v in _neighbors(u, self.n):
                if flat_obs[v]:
                    continue
                protected = root_owner.get(v)
                if protected is not None and protected != rid:
                    continue
                candidate = cost + 1.0
                if candidate + 1e-12 < distance[v] or (
                    abs(candidate - distance[v]) <= 1e-12 and rid < owner[v]
                ):
                    distance[v] = candidate
                    owner[v] = rid
                    heapq.heappush(heap, (candidate, rid, v))
        if np.any(owner[self.free_flat] < 0):
            raise AssertionError("Rooted power growth did not cover connected free space")
        return owner[self.free_flat]

    def partition_baseline(self, max_iter: int = 20) -> np.ndarray:
        """Reproduce multi-source FIFO BFS + Lloyd centroid relocation."""
        n, k = self.n, self.k
        seeds = self._initial_seeds()
        history: List[Tuple[int, ...]] = []
        flat_obs = self.obstacles.ravel()
        for _ in range(max_iter):
            a = np.full(n * n, -2, dtype=np.int16)
            a[flat_obs] = -1
            q = deque()
            for rid, s in enumerate(seeds):
                a[s] = rid
                q.append((s, rid))
            while q:
                u, rid = q.popleft()
                for v in _neighbors(u, n):
                    if a[v] == -2:
                        a[v] = rid
                        q.append((v, rid))
            new_seeds = []
            for rid in range(k):
                cells = np.flatnonzero(a == rid)
                if not len(cells):
                    new_seeds.append(seeds[rid])
                    continue
                xs, ys = cells // n, cells % n
                candidate = int(np.clip(round(xs.mean()), 0, n - 1)) * n + int(
                    np.clip(round(ys.mean()), 0, n - 1)
                )
                if flat_obs[candidate]:
                    candidate = int(cells[np.argmin(np.abs(xs - xs.mean()) + np.abs(ys - ys.mean()))])
                new_seeds.append(candidate)
            state = tuple(new_seeds)
            if state == tuple(seeds) or state in history:
                break
            history = (history + [state])[-3:]
            seeds = new_seeds
        # Lloyd relocation can move the generating sites away from the robots
        # and assign an original depot to another region.  Preserve the legacy
        # result when it is rooted; otherwise fall back to a root-fixed
        # geodesic Voronoi so downstream route comparisons remain executable.
        if any(a[root] != rid for rid, root in enumerate(self.robot_start_flats)):
            local = self._rooted_power_assignment(self.robot_start_flats, np.zeros(self.k))
            a = np.full(n * n, -1, dtype=np.int16)
            a[self.free_flat] = local
            seeds = list(self.robot_start_flats)
            self.baseline_root_repaired = True
        self.assignments = a.reshape((n, n))
        self.centroids_xy = [divmod(s, n) for s in seeds]
        return self.assignments

    def _locally_safe_removal(self, flat_a: np.ndarray, cell: int, rid: int) -> bool:
        same = [v for v in _neighbors(cell, self.n) if flat_a[v] == rid]
        if len(same) <= 1:
            return True
        # Sufficient digital-topology test: donor neighbors remain connected in
        # the 3x3 neighborhood after removing the boundary cell.
        x0, y0 = divmod(cell, self.n)
        allowed = set()
        for x in range(max(0, x0 - 1), min(self.n, x0 + 2)):
            for y in range(max(0, y0 - 1), min(self.n, y0 + 2)):
                u = x * self.n + y
                if u != cell and flat_a[u] == rid:
                    allowed.add(u)
        reached = {same[0]}
        q = deque([same[0]])
        while q:
            u = q.popleft()
            for v in _neighbors(u, self.n):
                if v in allowed and v not in reached:
                    reached.add(v)
                    q.append(v)
        return all(v in reached for v in same)

    def _boundary_balance(self, a: np.ndarray, targets: np.ndarray, max_passes: int = 40) -> np.ndarray:
        flat = a.ravel()
        protected_roots = set(self.robot_start_flats)
        sizes = np.bincount(flat[self.free_flat], minlength=self.k).astype(int)
        for _ in range(max_passes):
            changed = 0
            boundary = self.free_flat.copy()
            self.rng.shuffle(boundary)
            for u in boundary:
                if int(u) in protected_roots:
                    continue
                donor = int(flat[u])
                if sizes[donor] <= targets[donor]:
                    continue
                receivers = {
                    int(flat[v]) for v in _neighbors(int(u), self.n)
                    if flat[v] >= 0 and flat[v] != donor and sizes[int(flat[v])] < targets[int(flat[v])]
                }
                if not receivers or not self._locally_safe_removal(flat, int(u), donor):
                    continue
                receiver = min(receivers, key=lambda r: sizes[r] / targets[r])
                flat[u] = receiver
                sizes[donor] -= 1
                sizes[receiver] += 1
                changed += 1
            if changed == 0 or np.array_equal(sizes, targets):
                break
        return flat.reshape((self.n, self.n))

    def partition_balanced(self) -> np.ndarray:
        """Connected power partition with an epsilon-balance/shape objective.

        Among partitions meeting the requested CV tolerance, the algorithm
        keeps the one with the shortest inter-region boundary.  This avoids
        destroying 2x2/4x4 tile structure merely to transfer the final few
        cells, an important coupling between partitioning and downstream
        anisotropic tiling.
        """
        seeds = self._initial_seeds()
        m = len(self.free_flat)
        base, remainder = divmod(m, self.k)
        targets = np.full(self.k, base, dtype=int)
        targets[:remainder] += 1
        lam = np.zeros(self.k)
        best_assign = None
        best_score = float("inf")
        feasible_assign = None
        feasible_boundary = float("inf")
        feasible_pool: List[Tuple[int, float, np.ndarray]] = []
        stagnation = 0
        for it in range(self.partition_iterations):
            local = self._rooted_power_assignment(seeds, lam)
            sizes = np.bincount(local, minlength=self.k)
            score = coefficient_of_variation(sizes)
            if score + 1e-12 < best_score:
                best_score, best_assign = score, local.copy()
                stagnation = 0
            else:
                stagnation += 1
            if score <= max(self.balance_tolerance, 0.15 if self.n <= 30 else self.balance_tolerance):
                full = np.full(self.n * self.n, -1, dtype=np.int16)
                full[self.free_flat] = local
                grid = full.reshape((self.n, self.n))
                vertical = (~self.obstacles[:-1]) & (~self.obstacles[1:]) & (grid[:-1] != grid[1:])
                horizontal = (~self.obstacles[:, :-1]) & (~self.obstacles[:, 1:]) & (grid[:, :-1] != grid[:, 1:])
                boundary = int(vertical.sum() + horizontal.sum())
                if score <= self.balance_tolerance and boundary < feasible_boundary:
                    feasible_boundary = boundary
                    feasible_assign = local.copy()
                if self.n <= 30:
                    feasible_pool.append((boundary, score, local.copy()))
            if np.max(np.abs(sizes - targets)) <= 1:
                best_assign = local
                break
            step = max(0.35, 0.55 * self.n / math.sqrt(it + 1))
            lam += step * (sizes - targets) / np.maximum(targets, 1)
            lam -= lam.mean()
            if stagnation >= 18:
                lam *= 0.92
                stagnation = 0
        a = np.full(self.n * self.n, -1, dtype=np.int16)
        chosen = feasible_assign if feasible_assign is not None else best_assign
        # On tiny maps, discrete cell transfers strongly affect rectangle
        # alignment.  Jointly score a short list of epsilon-balanced partitions
        # by their *exact* downstream tile optimum.  The same strategy would be
        # too costly globally, hence the scale-dependent hybrid design.
        if self.n <= 30 and feasible_pool:
            proxy_ranked = []
            unique = set()
            for boundary, cv, candidate in feasible_pool:
                key = candidate.tobytes()
                if key in unique:
                    continue
                unique.add(key)
                temp = np.full(self.n * self.n, -1, dtype=np.int16)
                temp[self.free_flat] = candidate
                temp = temp.reshape((self.n, self.n))
                old = self.assignments
                self.assignments = temp
                proxy = 0
                for rid in range(self.k):
                    mask = temp == rid
                    proxy += min(len(self._scan_greedy(mask, rid, xr, yr)) for xr, yr in ((False, False), (True, True), (False, True), (True, False)))
                self.assignments = old
                proxy_ranked.append((proxy, cv, boundary, candidate))
            short = sorted(proxy_ranked, key=lambda z: (z[0], z[1], z[2]))[:10]
            best_joint = None
            for _, cv, boundary, candidate in short:
                temp = np.full(self.n * self.n, -1, dtype=np.int16)
                temp[self.free_flat] = candidate
                temp = temp.reshape((self.n, self.n))
                tile_total = 0
                exact_ok = True
                for rid in range(self.k):
                    patch = set(map(tuple, np.argwhere(temp == rid)))
                    exact = self._exact_patch(patch, rid, 0.12)
                    if exact is None:
                        exact_ok = False
                        break
                    tile_total += len(exact)
                score_tuple = (tile_total if exact_ok else 10**9, cv, boundary)
                if best_joint is None or score_tuple < best_joint[0]:
                    best_joint = (score_tuple, candidate)
            if best_joint is not None and best_joint[0][0] < 10**9:
                chosen = best_joint[1]
        a[self.free_flat] = chosen
        # Give ceil targets first to regions that are already largest, reducing
        # the number of topology-preserving boundary transfers.
        sizes = np.bincount(chosen, minlength=self.k)
        order = np.argsort(-sizes)
        targets = np.full(self.k, base, dtype=int)
        targets[order[:remainder]] += 1
        a = a.reshape((self.n, self.n))
        if coefficient_of_variation(sizes) > self.balance_tolerance:
            a = self._boundary_balance(a, targets)
        self.assignments = a
        self.centroids_xy = [divmod(s, self.n) for s in seeds]
        return a

    # -------------------------------- tiling -------------------------------
    def _region_mask(self, rid: int) -> np.ndarray:
        return self.assignments == rid

    def _scan_greedy(self, mask: np.ndarray, rid: int, xrev: bool, yrev: bool) -> List[Tile]:
        covered = np.zeros_like(mask)
        out: List[Tile] = []
        xr = range(self.n - 1, -1, -1) if xrev else range(self.n)
        yr_values = list(range(self.n - 1, -1, -1) if yrev else range(self.n))
        for w, h in self.tile_shapes:
            for x in xr:
                for y in yr_values:
                    if x + w <= self.n and y + h <= self.n and mask[x:x+w, y:y+h].all() and not covered[x:x+w, y:y+h].any():
                        out.append((x, y, w, h, rid))
                        covered[x:x+w, y:y+h] = True
        for x, y in np.argwhere(mask & ~covered):
            out.append((int(x), int(y), 1, 1, rid))
        return out

    def _randomized_greedy(self, mask: np.ndarray, rid: int, trials: int = 8) -> List[Tile]:
        placements = []
        for w, h in self.tile_shapes:
            for x in range(self.n - w + 1):
                for y in range(self.n - h + 1):
                    if mask[x:x+w, y:y+h].all():
                        placements.append((x, y, w, h, rid))
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
            x0 = int(np.clip(round(cx - window / 2), 0, max(0, self.n - window)))
            y0 = int(np.clip(round(cy - window / 2), 0, max(0, self.n - window)))
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

    def tile_baseline(self) -> List[Tile]:
        all_tiles = []
        for rid in range(self.k):
            mask = self._region_mask(rid)
            schemes = [self._scan_greedy(mask, rid, xr, yr) for xr, yr in ((False, False), (True, True), (False, True), (True, False))]
            all_tiles.extend(min(schemes, key=len))
        self.tiles = all_tiles
        return all_tiles

    def tile_hybrid(self) -> List[Tile]:
        all_tiles = []
        per_region = self.tiling_time_limit / self.k
        for rid in range(self.k):
            start = time.perf_counter()
            mask = self._region_mask(rid)
            schemes = [self._scan_greedy(mask, rid, xr, yr) for xr, yr in ((False, False), (True, True), (False, True), (True, False))]
            schemes.append(self._randomized_greedy(mask, rid, trials=6))
            tiles = min(schemes, key=len)
            tiles = self._local_milp_improve(tiles, rid, start + per_region)
            all_tiles.extend(tiles)
        self.tiles = all_tiles
        return all_tiles

    # -------------------------------- routing ------------------------------
    @staticmethod
    def _center_anchor(tile: Tile) -> Tuple[np.ndarray, Cell, float]:
        x, y, w, h, _ = tile
        center = np.array((x + w / 2.0, y + h / 2.0), dtype=float)
        ax, ay = x + (w - 1) // 2, y + (h - 1) // 2
        anchor_center = np.array((ax + 0.5, ay + 0.5))
        return center, (ax, ay), float(np.linalg.norm(center - anchor_center))

    def _all_pairs_region_distances(self, rid: int, anchors: List[Cell]) -> np.ndarray:
        n, m = self.n, len(anchors)
        allowed = (self.assignments.ravel() == rid)
        anchor_flat = np.array([x * n + y for x, y in anchors], dtype=int)
        anchor_lookup = {int(u): i for i, u in enumerate(anchor_flat)}
        out = np.full((m, m), np.inf)
        for i, source in enumerate(anchor_flat):
            dist = np.full(n * n, -1, dtype=np.int32)
            dist[source] = 0
            q = deque([int(source)])
            remaining = m - 1
            while q and remaining:
                u = q.popleft()
                j = anchor_lookup.get(u)
                if j is not None and j != i and not np.isfinite(out[i, j]):
                    out[i, j] = dist[u]
                    remaining -= 1
                for v in _neighbors(u, n):
                    if allowed[v] and dist[v] < 0:
                        dist[v] = dist[u] + 1
                        q.append(v)
            out[i, i] = 0.0
        return out

    @staticmethod
    def _two_opt(order: List[int], d: np.ndarray, passes: int = 3) -> List[int]:
        m = len(order)
        if m < 4:
            return order
        for _ in range(passes):
            improved = False
            for i in range(m - 2):
                a, b = order[i], order[(i + 1) % m]
                stop = m - (1 if i == 0 else 0)
                js = np.arange(i + 2, stop)
                if not len(js):
                    continue
                arr = np.asarray(order, dtype=int)
                c = arr[js]
                e = arr[(js + 1) % m]
                gains = d[a, c] + d[b, e] - d[a, b] - d[c, e]
                p = int(np.argmin(gains))
                if gains[p] < -1e-9:
                    j = int(js[p])
                    order[i+1:j+1] = reversed(order[i+1:j+1])
                    improved = True
            if not improved:
                break
        return order

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
        upper = minimum_spanning_tree(sp.csr_matrix(d))
        weight = float(upper.data.sum())
        tree = (upper + upper.T).tocsr()
        preorder = list(map(int, depth_first_order(tree, start, directed=False, return_predecessors=False)))
        adjacency = [[] for _ in range(m)]
        coo = upper.tocoo()
        for u, v in zip(coo.row, coo.col):
            adjacency[int(u)].append(int(v))
            adjacency[int(v)].append(int(u))
        walk = [start]
        stack = [(start, -1, iter(adjacency[start]))]
        while stack:
            node, parent, children = stack[-1]
            try:
                child = next(children)
                if child == parent:
                    continue
                walk.append(child)
                stack.append((child, node, iter(adjacency[child])))
            except StopIteration:
                stack.pop()
                if stack:
                    walk.append(stack[-1][0])
        return preorder, walk, weight

    def _root_tile_index(self, rid: int, tiles: Sequence[Tile]) -> int:
        root = self.robot_starts_xy[rid]
        for i, tile in enumerate(tiles):
            if root in self._tile_cells(tile):
                return i
        raise AssertionError(f"Robot {rid} root is not covered by its own tile set")

    def _attach_depot(self, rid: int, path: List[List[float]], root_center: np.ndarray) -> Tuple[List[List[float]], float]:
        depot = np.asarray(self.robot_starts_xy[rid], dtype=float) + 0.5
        leg = float(np.linalg.norm(depot - root_center))
        if not path:
            path = [root_center.tolist()]
        if leg > 1e-12:
            path = [depot.tolist()] + path + [depot.tolist()]
        return path, 2.0 * leg

    def _path_turn_metrics(self, path: Sequence[Sequence[float]]) -> Tuple[float, float]:
        """Return equivalent quarter turns and rotation time for a polyline.

        The initial heading is north, matching LS-MCPP's public benchmark
        convention.  Arbitrary portal segments are handled by charging their
        smallest wrapped heading change, normalized by 90 degrees.
        """
        previous_angle = math.pi / 2.0
        quarter_turns = 0.0
        for p, q in zip(path, path[1:]):
            delta = np.asarray(q, dtype=float) - np.asarray(p, dtype=float)
            if float(np.linalg.norm(delta)) <= 1e-12:
                continue
            angle = math.atan2(float(delta[1]), float(delta[0]))
            change = abs((angle - previous_angle + math.pi) % (2.0 * math.pi) - math.pi)
            quarter_turns += change / (math.pi / 2.0)
            previous_angle = angle
        return float(quarter_turns), float(quarter_turns * self.turn_time_90)

    def _oriented_grid_tour(
        self,
        tiles: Sequence[Tile],
        order: Sequence[int],
        cache: Optional[Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> dict:
        """Exactly realize one singleton visit order in heading-state space.

        For a fixed cyclic target order, each leg is solved on the product
        graph ``(cell, arrival heading)``.  A four-state dynamic program then
        couples the arrival heading of every leg to the departure heading of
        the next one.  Thus equal-length grid paths are no longer expanded by
        arbitrary Dijkstra tie breaking, and a longer translation is selected
        when it genuinely lowers total rotation time.

        This is exact only for the supplied visit order, not for the joint
        ordering problem.  It currently applies to the fixed 1x1 reduction;
        rectangular portal actions require directed portal states.
        """
        if not order:
            return {
                "path": [], "length": 0.0, "quarter_turns": 0.0,
                "turn_time": 0.0, "motion_time": 0.0,
            }
        if any(tile[2:4] != (1, 1) for tile in tiles):
            raise ValueError("Heading-state grid realization requires singleton actions")
        if len(order) == 1:
            x, y, _, _, _ = tiles[order[0]]
            return {
                "path": [[x + 0.5, y + 0.5]], "length": 0.0,
                "quarter_turns": 0.0, "turn_time": 0.0, "motion_time": 0.0,
            }

        local_cache = {} if cache is None else cache
        cells = [(tile[0], tile[1]) for tile in tiles]
        cell_to_node = {cell: node for node, cell in enumerate(cells)}
        # Match LS-MCPP's public Heading enum: E=0, N=1, W=2, S=3.
        moves = ((1, 0), (0, 1), (-1, 0), (0, -1))
        node_neighbors: List[List[Tuple[int, int]]] = [[] for _ in cells]
        for node, (x, y) in enumerate(cells):
            for heading, (dx, dy) in enumerate(moves):
                neighbor = cell_to_node.get((x + dx, y + dy))
                if neighbor is not None:
                    node_neighbors[node].append((neighbor, heading))

        def shortest_states(source: int, target: int, start_heading: int) -> Tuple[np.ndarray, np.ndarray]:
            key = (source, target, start_heading)
            if key in local_cache:
                return local_cache[key]
            state_count = 4 * len(cells)
            dist = np.full(state_count, np.inf, dtype=float)
            predecessor = np.full(state_count, -1, dtype=np.int32)
            start_state = 4 * source + start_heading
            dist[start_state] = 0.0
            queue = [(0.0, start_state)]
            while queue:
                cost, state = heapq.heappop(queue)
                if cost > dist[state] + 1e-12:
                    continue
                node, heading = divmod(state, 4)
                for neighbor, move_heading in node_neighbors[node]:
                    quarter = min((move_heading - heading) % 4, (heading - move_heading) % 4)
                    next_cost = (
                        cost + 1.0 / self.robot_speed
                        + quarter * self.turn_time_90
                    )
                    next_state = 4 * neighbor + move_heading
                    if next_cost + 1e-12 < dist[next_state]:
                        dist[next_state] = next_cost
                        predecessor[next_state] = state
                        heapq.heappush(queue, (next_cost, next_state))
            if not np.isfinite(dist[4 * target:4 * target + 4]).any():
                raise AssertionError("Singleton region is disconnected in heading-state graph")
            local_cache[key] = (dist, predecessor)
            return dist, predecessor

        legs = [
            (int(order[i]), int(order[(i + 1) % len(order)]))
            for i in range(len(order))
        ]
        transition_tables: List[np.ndarray] = []
        for source, target in legs:
            table = np.full((4, 4), np.inf, dtype=float)
            for start_heading in range(4):
                dist, _ = shortest_states(source, target, start_heading)
                table[start_heading] = dist[4 * target:4 * target + 4]
            transition_tables.append(table)

        costs = np.full(4, np.inf, dtype=float)
        costs[1] = 0.0  # initial heading north
        choices: List[np.ndarray] = []
        for table in transition_tables:
            next_costs = np.full(4, np.inf, dtype=float)
            choice = np.full(4, -1, dtype=np.int8)
            for end_heading in range(4):
                values = costs + table[:, end_heading]
                start_heading = int(np.argmin(values))
                next_costs[end_heading] = values[start_heading]
                choice[end_heading] = start_heading
            costs = next_costs
            choices.append(choice)

        end_heading = int(np.argmin(costs))
        heading_pairs: List[Tuple[int, int]] = [(-1, -1)] * len(legs)
        for leg_index in range(len(legs) - 1, -1, -1):
            start_heading = int(choices[leg_index][end_heading])
            heading_pairs[leg_index] = (start_heading, end_heading)
            end_heading = start_heading
        if end_heading != 1:
            raise AssertionError("Heading dynamic program lost the fixed initial heading")

        node_path: List[int] = []
        for (source, target), (start_heading, target_heading) in zip(legs, heading_pairs):
            _, predecessor = shortest_states(source, target, start_heading)
            start_state = 4 * source + start_heading
            state = 4 * target + target_heading
            reversed_nodes = [target]
            while state != start_state:
                state = int(predecessor[state])
                if state < 0:
                    raise AssertionError("Missing predecessor in heading-state path")
                reversed_nodes.append(state // 4)
            segment = list(reversed(reversed_nodes))
            if node_path:
                segment = segment[1:]
            node_path.extend(segment)

        path = [[cells[node][0] + 0.5, cells[node][1] + 0.5] for node in node_path]
        length = float(max(0, len(node_path) - 1))
        quarter_turns, turn_time = self._path_turn_metrics(path)
        motion_time = length / self.robot_speed + turn_time
        if abs(motion_time - float(np.min(costs))) > 1e-7:
            raise AssertionError("Heading-state DP cost does not match expanded path")
        return {
            "path": path,
            "length": length,
            "quarter_turns": quarter_turns,
            "turn_time": turn_time,
            "motion_time": motion_time,
        }

    def _oriented_relocate_search(
        self,
        tiles: Sequence[Tile],
        order: Sequence[int],
        distance: np.ndarray,
        cache: Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray]],
        max_evaluations: int,
        passes: int = 2,
    ) -> Tuple[List[int], dict]:
        """Improve target order using exact heading-state relocate evaluations.

        Relocating one non-root action changes only a constant number of order
        arcs, so cached orientation-conditioned shortest paths are reused.  A
        cheap translation delta ranks the neighborhood, but acceptance uses
        the exact heading-state motion time.
        """
        current = list(order)
        current_realization = self._oriented_grid_tour(tiles, current, cache)
        if len(current) < 4 or max_evaluations <= 0:
            return current, current_realization
        for _ in range(max(1, passes)):
            base_length = self._tour_length(current, distance)
            ranked = []
            for source_index in range(1, len(current)):
                for destination_index in range(1, len(current)):
                    if destination_index in (source_index, source_index + 1):
                        continue
                    candidate = list(current)
                    moved = candidate.pop(source_index)
                    insert_at = destination_index
                    if destination_index > source_index:
                        insert_at -= 1
                    candidate.insert(insert_at, moved)
                    translation_delta = self._tour_length(candidate, distance) - base_length
                    ranked.append((translation_delta, source_index, destination_index, candidate))
            ranked.sort(key=lambda item: (item[0], item[1], item[2]))
            best_order = current
            best_realization = current_realization
            for _, _, _, candidate in ranked[:max_evaluations]:
                realization = self._oriented_grid_tour(tiles, candidate, cache)
                if realization["motion_time"] + 1e-9 < best_realization["motion_time"]:
                    best_order = candidate
                    best_realization = realization
            if best_order == current:
                break
            current, current_realization = best_order, best_realization
        return current, current_realization

    def _tile_geodesic_metric(
        self, tiles: List[Tile]
    ) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, Dict[Tuple[int, int], np.ndarray]]:
        """Shortest center-to-center metric on the obstacle-safe tile graph.

        A segment between centers of two edge-adjacent rectangles lies inside
        their union.  Therefore every graph path is collision-free, while
        sparse Dijkstra provides the exact shortest path in this geometric
        roadmap without the artificial center-to-cell-anchor detour.
        """
        centers = [np.array((t[0] + t[2] / 2, t[1] + t[3] / 2), dtype=float) for t in tiles]
        owner: Dict[Cell, int] = {}
        for i, t in enumerate(tiles):
            for c in self._tile_cells(t):
                owner[c] = i
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
                    weight = float(np.linalg.norm(centers[i] - portal) + np.linalg.norm(portal - centers[j]))
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

    def route_tsp(self, robot_ids: Optional[Sequence[int]] = None) -> Dict[int, dict]:
        ids = list(range(self.k)) if robot_ids is None else list(map(int, robot_ids))
        routes = {} if robot_ids is None else dict(self.routes)
        for rid in ids:
            tiles = [t for t in self.tiles if t[4] == rid]
            if not tiles:
                routes[rid] = {"order": [], "centers": [], "path": [], "length": 0.0, "mission_time": 0.0}
                continue
            centers, d, predecessors, edge_portals = self._tile_geodesic_metric(tiles)
            start = self._root_tile_index(rid, tiles)
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
                self._two_opt(nn_order, d, passes=passes),
                self._two_opt(preorder, d, passes=passes),
            ]
            evaluated = []
            oriented_cache: Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray]] = {}
            for candidate in candidates:
                # Always retain the ordinary metric expansion as a conservative
                # route candidate.  The heading-state realization is an
                # additional exact subproblem, never a replacement that can
                # alter the search trajectory without an incumbent fallback.
                candidate_length = self._tour_length(candidate, d)
                candidate_path = self._expand_metric_tour(
                    candidate, predecessors, centers, edge_portals
                )
                candidate_path, candidate_depot_length = self._attach_depot(
                    rid, candidate_path, centers[start]
                )
                candidate_total_length = candidate_length + candidate_depot_length
                candidate_quarter_turns, candidate_turn_time = self._path_turn_metrics(
                    candidate_path
                )
                candidate_mission = (
                    self.service_time_per_tile * len(tiles)
                    + candidate_total_length / self.robot_speed
                    + candidate_turn_time
                )
                evaluated.append((
                    candidate_mission,
                    candidate_total_length,
                    candidate,
                    candidate_path,
                    candidate_depot_length,
                    candidate_quarter_turns,
                    candidate_turn_time,
                    "metric_expansion",
                ))

                use_oriented_grid = (
                    self.orientation_lifted_routing
                    and
                    self.turn_time_90 > 0
                    and all(tile[2:4] == (1, 1) for tile in tiles)
                )
                if use_oriented_grid:
                    realization = self._oriented_grid_tour(
                        tiles, candidate, cache=oriented_cache
                    )
                    candidate_total_length = realization["length"]
                    candidate_path = realization["path"]
                    candidate_depot_length = 0.0
                    candidate_quarter_turns = realization["quarter_turns"]
                    candidate_turn_time = realization["turn_time"]
                    candidate_mission = (
                        self.service_time_per_tile * len(tiles)
                        + candidate_total_length / self.robot_speed
                        + candidate_turn_time
                    )
                    evaluated.append((
                        candidate_mission,
                        candidate_total_length,
                        candidate,
                        candidate_path,
                        candidate_depot_length,
                        candidate_quarter_turns,
                        candidate_turn_time,
                        "orientation_lifted",
                    ))
            if self.orientation_lifted_routing and self.turn_time_90 > 0:
                oriented_candidates = [
                    item for item in evaluated if item[7] == "orientation_lifted"
                ]
                if oriented_candidates:
                    oriented_seed = min(oriented_candidates, key=lambda item: (item[0], item[1]))
                    improved_order, realization = self._oriented_relocate_search(
                        tiles,
                        oriented_seed[2],
                        d,
                        oriented_cache,
                        max_evaluations=self.orientation_order_evaluations,
                    )
                    improved_mission = (
                        self.service_time_per_tile * len(tiles)
                        + realization["motion_time"]
                    )
                    evaluated.append((
                        improved_mission,
                        realization["length"],
                        improved_order,
                        realization["path"],
                        0.0,
                        realization["quarter_turns"],
                        realization["turn_time"],
                        "orientation_lifted_relocate",
                    ))
            (
                mission_time,
                length,
                order,
                path,
                depot_length,
                quarter_turns,
                turn_time,
                routing_mode,
            ) = min(evaluated, key=lambda item: (item[0], item[1]))
            marginal_costs: Dict[Tile, float] = {}
            if len(order) > 1:
                for pos, tile_index in enumerate(order):
                    previous = order[(pos - 1) % len(order)]
                    following = order[(pos + 1) % len(order)]
                    detour = d[previous, tile_index] + d[tile_index, following] - d[previous, following]
                    marginal_costs[tiles[tile_index]] = float(max(0.0, detour) / self.robot_speed)
            else:
                marginal_costs[tiles[0]] = 0.0
            routes[rid] = {
                "order": order,
                "centers": [centers[i] for i in order],
                "path": path,
                "length": float(length),
                "quarter_turns": float(quarter_turns),
                "turn_time": float(turn_time),
                "routing_mode": routing_mode,
                "mission_time": float(mission_time),
                "double_tree_bound": float(2.0 * mst_weight + depot_length),
                "tile_marginal_costs": marginal_costs,
            }
        self.routes = routes
        return routes

    def _region_is_rooted_connected(self, assignments: np.ndarray, rid: int) -> bool:
        root = self.robot_start_flats[rid]
        flat = assignments.ravel()
        if flat[root] != rid:
            return False
        cells = np.flatnonzero(flat == rid)
        seen = {int(root)}
        q = deque([int(root)])
        while q:
            for v in _neighbors(q.popleft(), self.n):
                if flat[v] == rid and v not in seen:
                    seen.add(v)
                    q.append(v)
        return len(seen) == len(cells)

    @staticmethod
    def _tile_intersects_box(tile: Tile, box: Tuple[int, int, int, int]) -> bool:
        x0, y0, x1, y1 = box
        x, y, w, h, _ = tile
        return x < x1 and x + w > x0 and y < y1 and y + h > y0

    def _retile_boundary_patch_variants(
        self,
        assignments: np.ndarray,
        moved_tile: Tile,
        donor: int,
        receiver: int,
        deadline: float,
        padding: int = 4,
        max_variants: int = 3,
    ) -> List[List[Tile]]:
        """Generate diverse exact-cover tilings for both sides of a move."""
        x, y, w, h, _ = moved_tile
        box = (
            max(0, x - padding), max(0, y - padding),
            min(self.n, x + w + padding), min(self.n, y + h + padding),
        )
        affected = [
            t for t in self.tiles
            if t[4] in (donor, receiver) and self._tile_intersects_box(t, box)
        ]
        if moved_tile not in affected:
            return []
        patch_cells = {c for tile in affected for c in self._tile_cells(tile)}
        affected_set = set(affected)
        outside = [t for t in self.tiles if t not in affected_set]
        per_robot_options: List[List[List[Tile]]] = []
        old_assignments = self.assignments
        self.assignments = assignments
        try:
            for rid in (donor, receiver):
                cells = {c for c in patch_cells if assignments[c] == rid}
                if not cells:
                    per_robot_options.append([[]])
                    continue
                mask = np.zeros((self.n, self.n), dtype=bool)
                for cell in cells:
                    mask[cell] = True
                schemes = [
                    self._scan_greedy(mask, rid, xr, yr)
                    for xr, yr in ((False, False), (True, True), (False, True), (True, False))
                ]
                remaining = deadline - time.perf_counter()
                if remaining > 0.03 and len(cells) <= 220:
                    exact = self._exact_patch(cells, rid, min(0.18, remaining))
                    if exact is not None:
                        schemes.append(exact)
                unique: Dict[Tuple[Tile, ...], List[Tile]] = {}
                for scheme in schemes:
                    key = tuple(sorted(scheme))
                    unique.setdefault(key, scheme)
                # Tile count is only a ranking heuristic.  Multiple geometries
                # survive and are later judged by the true routed makespan.
                options = sorted(unique.values(), key=lambda option: (len(option), tuple(sorted(option))))
                per_robot_options.append(options[:max_variants])
        finally:
            self.assignments = old_assignments
        variants = []
        for choices in itertools.product(*per_robot_options):
            variants.append(outside + [tile for option in choices for tile in option])
        variants.sort(key=lambda option: (len(option), tuple(sorted(option))))
        return variants[:max_variants]

    def _boundary_transfer_candidates(self, bottleneck: int, limit: int) -> List[Tuple[Tile, int]]:
        owner: Dict[Cell, Tile] = {}
        for tile in self.tiles:
            if tile[4] == bottleneck:
                for cell in self._tile_cells(tile):
                    owner[cell] = tile
        scored: Dict[Tuple[Tile, int], Tuple[int, int]] = {}
        a = self.assignments
        root = self.robot_starts_xy[bottleneck]
        for cell, tile in owner.items():
            if root in self._tile_cells(tile):
                continue
            x, y = cell
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if not (0 <= nx < self.n and 0 <= ny < self.n):
                    continue
                receiver = int(a[nx, ny])
                if receiver < 0 or receiver == bottleneck:
                    continue
                key = (tile, receiver)
                boundary_contacts, area = scored.get(key, (0, tile[2] * tile[3]))
                scored[key] = (boundary_contacts + 1, area)
        geometry_ranked = sorted(
            scored, key=lambda key: (-scored[key][0], -scored[key][1], key[1], key[0])
        )
        if self.dual_guidance and scored:
            duals = {bottleneck: self._tiling_lp_duals(bottleneck)}
            for _, receiver in scored:
                if receiver not in duals:
                    duals[receiver] = self._tiling_lp_duals(receiver)

            def predicted_gain(key: Tuple[Tile, int]) -> float:
                tile, receiver = key
                cells = self._tile_cells(tile)
                donor_relief = sum(duals[bottleneck].get(c, 1.0 / 16.0) for c in cells)
                receiver_boundary_prices = []
                for x, y in cells:
                    for nb in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
                        if duals[receiver].get(nb) is not None:
                            receiver_boundary_prices.append(duals[receiver][nb])
                density = (
                    float(np.mean(receiver_boundary_prices))
                    if receiver_boundary_prices else 1.0 / 16.0
                )
                receiver_cost = density * len(cells)
                donor_route_relief = float(
                    self.routes.get(bottleneck, {}).get("tile_marginal_costs", {}).get(tile, 0.0)
                )
                center = np.array(
                    (tile[0] + tile[2] / 2.0, tile[1] + tile[3] / 2.0), dtype=float
                )
                receiver_centers = self.routes.get(receiver, {}).get("centers", [])
                receiver_insertion = (
                    2.0 * min(float(np.linalg.norm(center - np.asarray(c))) for c in receiver_centers)
                    / self.robot_speed
                    if receiver_centers else 0.0
                )
                contacts, area = scored[key]
                return (
                    donor_relief + donor_route_relief
                    - receiver_cost - receiver_insertion
                    + 1e-3 * contacts + 1e-5 * area
                )

            dual_ranked = sorted(
                scored,
                key=lambda key: (-predicted_gain(key), -scored[key][0], key[1], key[0]),
            )
            ranked = dual_ranked
        else:
            # Geometry-only ablation.
            ranked = geometry_ranked
        return [(tile, receiver) for tile, receiver in ranked[:limit]]

    def _tiling_lp_duals(self, rid: int) -> Dict[Cell, float]:
        """Exact-cover LP prices used as downstream workload subgradients."""
        cells = list(map(tuple, np.argwhere(self.assignments == rid)))
        if not cells:
            return {}
        cell_id = {cell: i for i, cell in enumerate(cells)}
        candidates: List[Tile] = []
        for w, h in self.action_shapes:
            for x in range(self.n - w + 1):
                for y in range(self.n - h + 1):
                    if (self.assignments[x:x+w, y:y+h] == rid).all():
                        candidates.append((x, y, w, h, rid))
        rows, cols = [], []
        for pid, tile in enumerate(candidates):
            for cell in self._tile_cells(tile):
                rows.append(cell_id[cell]); cols.append(pid)
        matrix = sp.coo_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(len(cells), len(candidates))
        ).tocsr()
        result = linprog(
            np.full(len(candidates), self.service_time_per_tile),
            A_eq=matrix,
            b_eq=np.ones(len(cells)),
            bounds=(0.0, 1.0),
            method="highs",
            options={"presolve": True},
        )
        if not result.success or result.eqlin.marginals is None:
            return {cell: self.service_time_per_tile / 16.0 for cell in cells}
        prices = np.maximum(np.asarray(result.eqlin.marginals, dtype=float), 0.0)
        return {cell: float(prices[i]) for i, cell in enumerate(cells)}

    def _joint_fact_patch_variant(
        self,
        proposed_assignments: np.ndarray,
        moved_tile: Tile,
        donor: int,
        receiver: int,
        deadline: float,
        max_side: int = 8,
    ) -> Optional[Tuple[np.ndarray, List[Tile]]]:
        """Solve a frozen-anchor FACT tree model around one boundary action.

        Unlike scan/MILP retiling, this local oracle jointly chooses cell
        ownership, exact-cover rectangles, and rooted portal trees.  The rest
        of both robot regions remains frozen and is reattached through one
        protected anchor per side.
        """
        remaining = deadline - time.perf_counter()
        if remaining < 0.15:
            return None
        moved_cells = set(self._tile_cells(moved_tile))
        halo = set(moved_cells)
        for x, y in list(moved_cells):
            halo.update((nx, ny) for nx, ny in ((x-1, y), (x+1, y), (x, y-1), (x, y+1))
                        if 0 <= nx < self.n and 0 <= ny < self.n)
        affected = [
            tile for tile in self.tiles
            if tile[4] in (donor, receiver)
            and any(cell in halo for cell in self._tile_cells(tile))
        ]
        if moved_tile not in affected:
            return None
        patch = {cell for tile in affected for cell in self._tile_cells(tile)}
        xs, ys = [c[0] for c in patch], [c[1] for c in patch]
        x0, y0 = min(xs), min(ys)
        width, height = max(xs) - x0 + 1, max(ys) - y0 + 1
        side = max(width, height)
        if side > max_side:
            return None

        anchors: List[Cell] = []
        for rid in (donor, receiver):
            candidates = []
            for cell in patch:
                if proposed_assignments[cell] != rid:
                    continue
                flat = cell[0] * self.n + cell[1]
                if any(
                    divmod(v, self.n) not in patch and proposed_assignments.ravel()[v] == rid
                    for v in _neighbors(flat, self.n)
                ):
                    candidates.append(cell)
            root = self.robot_starts_xy[rid]
            if root in patch and proposed_assignments[root] == rid:
                candidates.append(root)
            if not candidates:
                return None
            anchors.append(min(candidates, key=lambda c: abs(c[0] - moved_tile[0]) + abs(c[1] - moved_tile[1])))
        if anchors[0] == anchors[1]:
            return None

        local_mask = np.ones((side, side), dtype=bool)
        for x, y in patch:
            local_mask[x - x0, y - y0] = False
        local_starts = [(x - x0, y - y0) for x, y in anchors]
        try:
            # Delayed import avoids a module cycle: fact_mcpp reuses this
            # class for executable-route validation.
            # Self-contained mainline: exact FACT oracle is defined below.
            fixed_costs = []
            for rid in (donor, receiver):
                affected_count = sum(tile[4] == rid for tile in affected)
                fixed_costs.append(max(
                    0.0,
                    float(self.routes[rid]["mission_time"])
                    - self.service_time_per_tile * affected_count,
                ))
            answer = solve_exact_fact(
                local_mask,
                local_starts,
                service_time_per_tile=self.service_time_per_tile,
                robot_speed=self.robot_speed,
                time_limit=min(0.8, max(0.15, remaining)),
                mip_rel_gap=0.15,
                max_placements=700,
                max_edges=12000,
                fixed_costs=fixed_costs,
                strengthen_connectivity=False,
                tile_shapes=self.tile_shapes,
            )
        except (AssertionError, RuntimeError, ValueError):
            return None

        candidate_assignments = self.assignments.copy()
        local_assignment = answer["assignment"]
        rid_map = (donor, receiver)
        for x, y in patch:
            local_rid = int(local_assignment[x - x0, y - y0])
            if local_rid not in (0, 1):
                return None
            candidate_assignments[x, y] = rid_map[local_rid]
        if not self._region_is_rooted_connected(candidate_assignments, donor):
            return None
        if not self._region_is_rooted_connected(candidate_assignments, receiver):
            return None
        affected_set = set(affected)
        candidate_tiles = [tile for tile in self.tiles if tile not in affected_set]
        for x, y, w, h, local_rid in answer["tiles"]:
            candidate_tiles.append((x + x0, y + y0, w, h, rid_map[local_rid]))
        return candidate_assignments, candidate_tiles

    def _contracted_fact_band_variant(
        self,
        moved_tile: Tile,
        donor: int,
        receiver: int,
        deadline: float,
        band_radius: int = 1,
        max_patch_cells: int = 24,
    ) -> Optional[Tuple[np.ndarray, List[Tile]]]:
        """Jointly reoptimize a band while exactly contracting frozen outside trees."""
        remaining = deadline - time.perf_counter()
        if remaining < 0.2:
            return None
        pair = {donor, receiver}
        seed_cells = set(self._tile_cells(moved_tile))
        roots = {self.robot_starts_xy[donor], self.robot_starts_xy[receiver]}
        boundary_cells = set()
        for x, y in map(tuple, np.argwhere(np.isin(self.assignments, list(pair)))):
            owner = int(self.assignments[x, y])
            if any(
                0 <= nx < self.n and 0 <= ny < self.n
                and int(self.assignments[nx, ny]) in pair
                and int(self.assignments[nx, ny]) != owner
                for nx, ny in ((x-1, y), (x+1, y), (x, y-1), (x, y+1))
            ):
                boundary_cells.add((x, y))
        if not boundary_cells:
            return None

        def cell_distance(cell: Cell, targets: set[Cell]) -> int:
            return min(abs(cell[0] - x) + abs(cell[1] - y) for x, y in targets)

        affected = []
        for tile in self.tiles:
            if tile[4] not in pair or roots.intersection(self._tile_cells(tile)):
                continue
            cells = set(self._tile_cells(tile))
            if min(cell_distance(cell, boundary_cells) for cell in cells) > 1:
                continue
            distance = min(cell_distance(cell, seed_cells) for cell in cells)
            if distance <= band_radius:
                affected.append((distance, tile, cells))
        if not any(tile == moved_tile for _, tile, _ in affected):
            moved_cells = set(self._tile_cells(moved_tile))
            if not roots.intersection(moved_cells):
                affected.append((0, moved_tile, moved_cells))
        affected.sort(key=lambda item: (item[0], item[1]))
        patch: set[Cell] = set()
        owners = set()
        for _, tile, cells in affected:
            if len(patch.union(cells)) > max_patch_cells:
                continue
            patch.update(cells)
            owners.add(tile[4])
        if owners != pair or not seed_cells.issubset(patch):
            return None
        try:
            # Self-contained mainline: contracted FACT oracle is defined below.
            other_makespan = max(
                (float(self.routes[r]["mission_time"]) for r in range(self.k) if r not in pair),
                default=0.0,
            )
            answer = solve_contracted_fact_band(
                self.obstacles,
                self.assignments,
                self.tiles,
                self.robot_starts_xy,
                donor,
                receiver,
                sorted(patch),
                tile_shapes=self.tile_shapes,
                service_time_per_tile=self.service_time_per_tile,
                robot_speed=self.robot_speed,
                tree_multiplier=2.0,
                other_robot_makespan=other_makespan,
                time_limit=min(2.0, max(0.2, remaining)),
                mip_rel_gap=0.10,
            )
        except (AssertionError, RuntimeError, ValueError):
            return None
        candidate_assignments = answer["assignment"]
        if not self._region_is_rooted_connected(candidate_assignments, donor):
            return None
        if not self._region_is_rooted_connected(candidate_assignments, receiver):
            return None
        return candidate_assignments, answer["tiles"]

    def refine_coupled(
        self,
        time_limit: Optional[float] = None,
        max_iterations: int = 4,
        candidate_limit: int = 8,
        joint_retile: bool = True,
        joint_fact: bool = True,
        contracted_fact: bool = False,
    ) -> List[dict]:
        """Bottleneck-guided joint partition/tiling/routing neighborhood.

        Every accepted move strictly lowers the true depot-aware makespan.
        Feasibility is preserved by rooted connectivity checks, local exact
        cover reconstruction, and the common safe portal router.
        """
        budget = self.refinement_time_limit if time_limit is None else max(0.0, float(time_limit))
        deadline = time.perf_counter() + budget
        trace: List[dict] = []
        if not self.routes:
            self.route_tsp()
        initial = max(float(self.routes[r]["mission_time"]) for r in range(self.k))
        for iteration in range(max_iterations):
            if time.perf_counter() >= deadline:
                break
            mission = np.array([self.routes[r]["mission_time"] for r in range(self.k)], dtype=float)
            bottleneck = int(np.argmax(mission))
            current_makespan = float(mission.max())
            base_assignments, base_tiles, base_routes = self.assignments, self.tiles, self.routes
            best_after = current_makespan
            best_state = None
            best_info = None
            feasible = []

            def consider(
                candidate_assignments: np.ndarray,
                candidate_tiles: List[Tile],
                moved_tile: Tile,
                receiver: int,
                tiling_mode: str,
            ) -> float:
                nonlocal best_after, best_state, best_info
                self.assignments, self.tiles, self.routes = (
                    candidate_assignments, candidate_tiles, dict(base_routes)
                )
                after = float("inf")
                try:
                    self.route_tsp([bottleneck, receiver])
                    after = max(float(self.routes[r]["mission_time"]) for r in range(self.k))
                    if after + 1e-8 < best_after:
                        self.validate()
                        best_after = after
                        best_state = (candidate_assignments, candidate_tiles, dict(self.routes))
                        best_info = (moved_tile, receiver, tiling_mode)
                except (AssertionError, ValueError):
                    pass
                finally:
                    self.assignments, self.tiles, self.routes = base_assignments, base_tiles, base_routes
                return after

            # Phase 1 evaluates the assignment-only feasible member for every
            # candidate.  The joint neighborhood therefore contains, rather
            # than replaces, the non-coupled ablation.
            for moved_tile, receiver in self._boundary_transfer_candidates(bottleneck, candidate_limit):
                if time.perf_counter() >= deadline:
                    break
                candidate_assignments = self.assignments.copy()
                moved_cells = self._tile_cells(moved_tile)
                for cell in moved_cells:
                    candidate_assignments[cell] = receiver
                if not self._region_is_rooted_connected(candidate_assignments, bottleneck):
                    continue
                if not self._region_is_rooted_connected(candidate_assignments, receiver):
                    continue
                mx, my, mw, mh, _ = moved_tile
                replacement = (mx, my, mw, mh, receiver)
                transfer_tiles = [replacement if t == moved_tile else t for t in base_tiles]
                transfer_after = consider(
                    candidate_assignments, transfer_tiles, moved_tile, receiver, "assignment_only"
                )
                feasible.append(
                    (moved_tile, receiver, candidate_assignments, transfer_tiles, transfer_after)
                )

            # Phase 2 adds alternative exact covers and judges them only by the
            # routed makespan, never by tile count alone.
            if joint_retile:
                # Spend the exact-tree budget on the most promising transfer
                # found in Phase 1.  The local FACT oracle is the genuinely
                # coupled neighborhood; scan variants below are inexpensive
                # diversity fallbacks.
                if joint_fact:
                    exact_targets = sorted(feasible, key=lambda item: item[4])[:2]
                    for exact_target in exact_targets:
                        if time.perf_counter() >= deadline:
                            break
                        moved_tile, receiver, candidate_assignments, _, _ = exact_target
                        exact_variant = self._joint_fact_patch_variant(
                            candidate_assignments, moved_tile, bottleneck, receiver, deadline
                        )
                        if exact_variant is not None:
                            exact_assignments, exact_tiles = exact_variant
                            consider(
                                exact_assignments, exact_tiles, moved_tile, receiver,
                                "joint_fact_tree",
                            )
                if contracted_fact:
                    contracted_targets = sorted(feasible, key=lambda item: item[4])[:2]
                    for moved_tile, receiver, _, _, _ in contracted_targets:
                        if time.perf_counter() >= deadline:
                            break
                        contracted_variant = self._contracted_fact_band_variant(
                            moved_tile, bottleneck, receiver, deadline
                        )
                        if contracted_variant is not None:
                            band_assignments, band_tiles = contracted_variant
                            consider(
                                band_assignments,
                                band_tiles,
                                moved_tile,
                                receiver,
                                "contracted_fact_band",
                            )
                for moved_tile, receiver, candidate_assignments, transfer_tiles, _ in feasible:
                    if time.perf_counter() >= deadline:
                        break
                    variants = self._retile_boundary_patch_variants(
                        candidate_assignments, moved_tile, bottleneck, receiver, deadline
                    )
                    transfer_key = tuple(sorted(transfer_tiles))
                    for variant in variants:
                        if time.perf_counter() >= deadline:
                            break
                        if tuple(sorted(variant)) == transfer_key:
                            continue
                        consider(candidate_assignments, variant, moved_tile, receiver, "route_aware_retile")

            if best_state is None or best_info is None:
                break
            self.assignments, self.tiles, self.routes = best_state
            moved_tile, receiver, tiling_mode = best_info
            trace.append({
                "iteration": iteration,
                "donor": bottleneck,
                "receiver": receiver,
                "moved_tile": moved_tile[:4],
                "makespan_before": current_makespan,
                "makespan_after": best_after,
                "tile_count_after": len(self.tiles),
                "joint_retile": joint_retile,
                "tiling_mode": tiling_mode,
            })
        final = max(float(self.routes[r]["mission_time"]) for r in range(self.k))
        self.refinement_trace = trace
        if final > initial + 1e-8:
            raise AssertionError("Coupled refinement increased makespan")
        return trace

    def refine_coupled_beam(
        self,
        time_limit_per_beam: Optional[float] = None,
        max_iterations: int = 4,
        candidate_limit: int = 8,
    ) -> List[dict]:
        """Keep assignment, route-aware, and FACT-tree incumbents."""
        budget = (
            self.refinement_time_limit
            if time_limit_per_beam is None else max(0.0, float(time_limit_per_beam))
        )
        initial_state = (
            self.assignments.copy(), list(self.tiles), dict(self.routes),
            copy.deepcopy(self.rng.bit_generator.state), self.dual_guidance,
            self.orientation_lifted_routing,
        )
        beams = []
        if not self.tile_shapes:
            # In the strict 1x1 reduction there is no alternative exact cover,
            # so retiling and FACT-patch beams duplicate the same feasible
            # decisions while consuming three times the budget.
            beam_configs = [(False, False, False, False, "assignment_geometry")]
        else:
            beam_configs = [
                (False, False, False, False, "assignment_geometry"),
                (True, False, False, False, "route_aware_geometry"),
                (True, True, True, False, "dual_guided_fact"),
            ]
            if self.enable_contracted_fact_beam:
                beam_configs.append((True, False, False, True, "contracted_fact_lns"))
        if (
            self.enable_orientation_beam
            and self.turn_time_90 > 0
            and all(tile[2:4] == (1, 1) for tile in self.tiles)
        ):
            beam_configs.append((False, False, False, False, "orientation_lifted"))
        for joint_retile, use_fact, use_dual, use_contracted, beam_name in beam_configs:
            self.assignments = initial_state[0].copy()
            self.tiles = list(initial_state[1])
            self.routes = dict(initial_state[2])
            self.rng.bit_generator.state = copy.deepcopy(initial_state[3])
            self.dual_guidance = use_dual
            self.orientation_lifted_routing = beam_name == "orientation_lifted"
            if self.orientation_lifted_routing:
                self.route_tsp()
            beam_started = time.perf_counter()
            trace = self.refine_coupled(
                time_limit=budget,
                max_iterations=max_iterations,
                candidate_limit=candidate_limit,
                joint_retile=joint_retile,
                joint_fact=use_fact,
                contracted_fact=use_contracted,
            )
            beam_runtime = time.perf_counter() - beam_started
            makespan = max(float(self.routes[r]["mission_time"]) for r in range(self.k))
            beams.append((
                makespan, self.assignments, self.tiles, self.routes, list(trace), beam_name,
                beam_runtime,
            ))
        winner = min(beams, key=lambda beam: (beam[0], len(beam[2])))
        _, self.assignments, self.tiles, self.routes, trace, selected_beam, _ = winner
        self.selected_refinement_beam = selected_beam
        self.dual_guidance = initial_state[4]
        self.orientation_lifted_routing = initial_state[5]
        self.refinement_trace = trace
        self.refinement_beams = [
            {
                "makespan": beam[0],
                "tile_count": len(beam[2]),
                "beam": beam[5],
                "runtime": beam[6],
                "iterations": len(beam[4]),
                "accepted_modes": {
                    mode: sum(step.get("tiling_mode") == mode for step in beam[4])
                    for mode in sorted({step.get("tiling_mode", "") for step in beam[4]})
                    if mode
                },
            }
            for beam in beams
        ]
        for step in self.refinement_trace:
            step["selected_beam"] = selected_beam
        return self.refinement_trace

    def route_mst_walk_length(self) -> float:
        """Build the uploaded method's closed depth-first walk around its MST.

        Besides returning ``2 * MST weight``, this stores the actual center
        polyline in ``self.routes[rid]['path']`` so the baseline can be plotted
        and checked using the same geometry convention as the hybrid route.
        """
        total = 0.0
        routes: Dict[int, dict] = {}
        for rid in range(self.k):
            tiles = [t for t in self.tiles if t[4] == rid]
            if not tiles:
                routes[rid] = {"order": [], "centers": [], "path": [], "length": 0.0, "mission_time": 0.0}
                continue
            centers, d, predecessors, edge_portals = self._tile_geodesic_metric(tiles)
            start = self._root_tile_index(rid, tiles)
            _, walk, weight = self._metric_mst_orders(d, start)
            path = self._expand_metric_tour(walk, predecessors, centers, edge_portals)
            path, depot_length = self._attach_depot(rid, path, centers[start])
            robot_length = 2.0 * weight + depot_length
            quarter_turns, turn_time = self._path_turn_metrics(path)
            mission_time = (
                self.service_time_per_tile * len(tiles)
                + robot_length / self.robot_speed
                + turn_time
            )
            center_lists = [c.tolist() for c in centers]
            routes[rid] = {
                "order": walk,
                "centers": center_lists,
                "path": path,
                "length": robot_length,
                "quarter_turns": quarter_turns,
                "turn_time": turn_time,
                "mission_time": float(mission_time),
                "double_tree_bound": robot_length,
            }
            total += robot_length
        self.routes = routes
        return total

    # -------------------------------- public -------------------------------
    def _partition_sizes(self) -> np.ndarray:
        return np.array([np.count_nonzero(self.assignments == rid) for rid in range(self.k)])

    def _point_in_region_closure(self, point: np.ndarray, rid: int) -> bool:
        """Whether a geometric point lies in the closure of a robot's cells."""
        x, y = map(float, point)
        eps = 1e-8
        xs = {math.floor(x - eps), math.floor(x + eps)}
        ys = {math.floor(y - eps), math.floor(y + eps)}
        return any(
            0 <= px < self.n and 0 <= py < self.n and self.assignments[px, py] == rid
            for px in xs for py in ys
        )

    def _validate_routes(self) -> Tuple[bool, bool]:
        collision_free = True
        depot_closed = True
        for rid in range(self.k):
            route = self.routes.get(rid, {})
            path = [np.asarray(p, dtype=float) for p in route.get("path", [])]
            depot = np.asarray(self.robot_starts_xy[rid], dtype=float) + 0.5
            if not path or not np.allclose(path[0], depot) or not np.allclose(path[-1], depot):
                depot_closed = False
            for a, b in zip(path, path[1:]):
                steps = max(1, int(math.ceil(float(np.linalg.norm(b - a)) * 10.0)))
                for alpha in np.linspace(0.0, 1.0, steps + 1):
                    if not self._point_in_region_closure((1.0 - alpha) * a + alpha * b, rid):
                        collision_free = False
                        break
                if not collision_free:
                    break
        return collision_free, depot_closed

    def validate(self) -> dict:
        covered = np.zeros((self.n, self.n), dtype=np.int16)
        for x, y, w, h, rid in self.tiles:
            if not (self.assignments[x:x+w, y:y+h] == rid).all():
                raise AssertionError("Tile crosses an obstacle or partition boundary")
            covered[x:x+w, y:y+h] += 1
        if not np.all(covered[~self.obstacles] == 1) or np.any(covered[self.obstacles]):
            raise AssertionError("Tiling is not an exact cover")
        connected = []
        rooted = []
        for rid in range(self.k):
            cells = np.flatnonzero(self.assignments.ravel() == rid)
            seen = {int(cells[0])}
            q = deque(seen)
            while q:
                for v in _neighbors(q.popleft(), self.n):
                    if self.assignments.ravel()[v] == rid and v not in seen:
                        seen.add(v); q.append(v)
            connected.append(len(seen) == len(cells))
            rooted.append(self.assignments.ravel()[self.robot_start_flats[rid]] == rid)
        route_collision_free, depot_closed = self._validate_routes()
        if not route_collision_free:
            raise AssertionError("A route segment leaves its assigned free-space region")
        if not depot_closed:
            raise AssertionError("A route does not start and end at its robot depot")
        return {
            "exact_cover": True,
            "connected_regions": all(connected),
            "rooted_regions": all(rooted),
            "route_collision_free": route_collision_free,
            "depot_closed": depot_closed,
        }

    def solve(self, method: str = "hybrid") -> dict:
        times = StageTimes()
        refinement_time = 0.0
        initial_makespan = 0.0
        t = time.perf_counter()
        if method == "baseline":
            self.partition_baseline()
        elif method in ("hybrid", "coupled"):
            self.partition_balanced()
        else:
            raise ValueError("method must be 'baseline', 'hybrid', or 'coupled'")
        times.partition = time.perf_counter() - t
        t = time.perf_counter()
        self.tile_baseline() if method == "baseline" else self.tile_hybrid()
        times.tiling = time.perf_counter() - t
        t = time.perf_counter()
        if method == "baseline":
            path_length = self.route_mst_walk_length()
        else:
            self.route_tsp()
            initial_makespan = max(r["mission_time"] for r in self.routes.values())
            if method == "coupled":
                refine_start = time.perf_counter()
                self.refine_coupled_beam(
                    max_iterations=self.refinement_max_iterations,
                    candidate_limit=self.refinement_candidate_limit,
                )
                refinement_time = time.perf_counter() - refine_start
            path_length = sum(r["length"] for r in self.routes.values())
        times.routing = time.perf_counter() - t
        valid = self.validate()
        sizes = self._partition_sizes()
        tile_loads = np.array([sum(t[4] == rid for t in self.tiles) for rid in range(self.k)])
        path_loads = np.array([self.routes.get(rid, {}).get("length", 0.0) for rid in range(self.k)])
        turn_times = np.array([self.routes.get(rid, {}).get("turn_time", 0.0) for rid in range(self.k)])
        mission_times = np.array([self.routes.get(rid, {}).get("mission_time", 0.0) for rid in range(self.k)])
        return {
            "method": method,
            "map_size": self.n,
            "robots": self.k,
            "seed": self.seed,
            "free_cells": int(len(self.free_flat)),
            "partition_sizes": sizes.tolist(),
            "partition_cv": coefficient_of_variation(sizes),
            "tile_count": len(self.tiles),
            "tile_counts": tile_loads.tolist(),
            "tile_cv": coefficient_of_variation(tile_loads),
            "path_length": float(path_length),
            "path_lengths": path_loads.tolist(),
            "path_cv": coefficient_of_variation(path_loads),
            "turn_times": turn_times.tolist(),
            "turn_time_total": float(turn_times.sum()),
            "mission_times": mission_times.tolist(),
            "makespan": float(mission_times.max(initial=0.0)),
            "mission_time_total": float(mission_times.sum()),
            "mission_cv": coefficient_of_variation(mission_times),
            "service_time_per_tile": self.service_time_per_tile,
            "tile_shapes": self.tile_shapes,
            "robot_speed": self.robot_speed,
            "turn_time_90": self.turn_time_90,
            "orientation_lifted_routing": self.orientation_lifted_routing,
            "baseline_root_repaired": self.baseline_root_repaired,
            "initial_makespan": float(initial_makespan),
            "refinement_iterations": len(self.refinement_trace),
            "refinement_time": refinement_time,
            "makespan_refinement_pct": (
                100.0 * (initial_makespan - float(mission_times.max(initial=0.0))) / initial_makespan
                if initial_makespan > 0 else 0.0
            ),
            "selected_refinement_beam": (
                self.selected_refinement_beam
            ),
            "refinement_beams": copy.deepcopy(self.refinement_beams),
            "partition_time": times.partition,
            "tiling_time": times.tiling,
            "routing_time": times.routing,
            "total_time": times.total,
            **valid,
        }


# ============================================================================
# Exact FACT model and certificate oracle


# ============================================================================
"""Certified small-instance models for footprint-aware coupled MCPP.

The model jointly selects non-overlapping rectangular coverage actions,
assigns them to rooted robot regions, and connects every robot's selected
actions with a minimum-cost portal tree.  The one-tree model is a lower bound
on any closed walk in the same portal graph; the doubled-tree model gives an
executable upper bound.  Solving both therefore yields an auditable bound on
the original route objective, rather than only a MIP gap for a surrogate.

This implementation is intentionally an exact oracle for small instances.  It
is not the scalable solver; its role is to provide optima/lower bounds and to
validate the joint neighborhood method used on large maps.
"""


from dataclasses import dataclass
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp



@dataclass(frozen=True)
class Placement:
    x: int
    y: int
    w: int
    h: int
    cells: frozenset[Cell]

    @property
    def center(self) -> np.ndarray:
        return np.array((self.x + self.w / 2.0, self.y + self.h / 2.0), dtype=float)


def _placements(
    mask: np.ndarray,
    tile_shapes: Sequence[Tuple[int, int]] = TILE_SHAPES,
) -> Tuple[List[Placement], List[List[int]]]:
    n = mask.shape[0]
    out: List[Placement] = []
    by_cell: List[List[int]] = [[] for _ in range(n * n)]
    shapes = tuple(dict.fromkeys(tuple(tile_shapes) + ((1, 1),)))
    for w, h in shapes:
        for x in range(n - w + 1):
            for y in range(n - h + 1):
                if mask[x:x + w, y:y + h].any():
                    continue
                cells = frozenset((px, py) for px in range(x, x + w) for py in range(y, y + h))
                pid = len(out)
                out.append(Placement(x, y, w, h, cells))
                for px, py in cells:
                    by_cell[px * n + py].append(pid)
    return out, by_cell


def _compatibility_edges(
    mask: np.ndarray, placements: Sequence[Placement], by_cell: Sequence[Sequence[int]],
) -> List[Tuple[int, int, float]]:
    """Generate disjoint placement pairs sharing a free cell boundary."""
    n = mask.shape[0]
    weights: Dict[Tuple[int, int], float] = {}
    for x in range(n):
        for y in range(n):
            if mask[x, y]:
                continue
            u = x * n + y
            for nx, ny, portal in (
                (x + 1, y, np.array((x + 1.0, y + 0.5))),
                (x, y + 1, np.array((x + 0.5, y + 1.0))),
            ):
                if nx >= n or ny >= n or mask[nx, ny]:
                    continue
                v = nx * n + ny
                for p in by_cell[u]:
                    for q in by_cell[v]:
                        if p == q or not placements[p].cells.isdisjoint(placements[q].cells):
                            continue
                        edge = (min(p, q), max(p, q))
                        cost = float(
                            np.linalg.norm(placements[p].center - portal)
                            + np.linalg.norm(portal - placements[q].center)
                        )
                        if edge not in weights or cost < weights[edge]:
                            weights[edge] = cost
    return [(p, q, w) for (p, q), w in weights.items()]


class _Rows:
    def __init__(self) -> None:
        self.rows: List[int] = []
        self.cols: List[int] = []
        self.data: List[float] = []
        self.lb: List[float] = []
        self.ub: List[float] = []

    def add(self, coeffs: Dict[int, float], lb: float, ub: float) -> None:
        row = len(self.lb)
        for col, value in coeffs.items():
            if abs(value) > 0:
                self.rows.append(row)
                self.cols.append(col)
                self.data.append(float(value))
        self.lb.append(float(lb))
        self.ub.append(float(ub))

    def matrix(self, variables: int) -> sp.csr_matrix:
        return sp.coo_matrix(
            (self.data, (self.rows, self.cols)), shape=(len(self.lb), variables)
        ).tocsr()


def solve_exact_fact(
    obstacle_mask: np.ndarray,
    robot_starts: Sequence[Cell],
    *,
    service_time_per_tile: float = 1.0,
    robot_speed: float = 1.0,
    time_limit: float = 60.0,
    mip_rel_gap: float = 0.0,
    max_placements: int = 1200,
    max_edges: int = 30000,
    fixed_costs: Optional[Sequence[float]] = None,
    tree_multiplier: float = 2.0,
    strengthen_connectivity: bool = False,
    separate_connectivity_cuts: bool = False,
    cut_rounds: int = 4,
    cuts_per_round: int = 20,
    compute_root_lp_bound: bool = False,
    tile_shapes: Sequence[Tuple[int, int]] = TILE_SHAPES,
) -> dict:
    """Solve one joint rooted exact-cover tree model on a small square grid.

    ``tree_multiplier=1`` minimizes the MST relaxation and gives a lower
    bound on a closed portal-graph coverage walk.  ``tree_multiplier=2``
    minimizes the doubled-tree executable upper surrogate used by the
    approximation algorithm.
    """
    mask = np.asarray(obstacle_mask, dtype=bool)
    if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError("obstacle_mask must be square")
    if service_time_per_tile < 0 or robot_speed <= 0 or tree_multiplier <= 0:
        raise ValueError("Invalid service time or speed")
    n, k = mask.shape[0], len(robot_starts)
    if k < 1:
        raise ValueError("At least one robot is required")
    starts = [(int(x), int(y)) for x, y in robot_starts]
    if len(set(starts)) != k or any(not (0 <= x < n and 0 <= y < n) or mask[x, y] for x, y in starts):
        raise ValueError("Robot starts must be distinct free cells")
    fixed = np.zeros(k, dtype=float) if fixed_costs is None else np.asarray(fixed_costs, dtype=float)
    if fixed.shape != (k,) or np.any(fixed < 0):
        raise ValueError("fixed_costs must be one nonnegative value per robot")

    build_start = time.perf_counter()
    placements, by_cell = _placements(mask, tile_shapes)
    if len(placements) > max_placements:
        raise ValueError(f"Exact oracle has {len(placements)} placements; limit is {max_placements}")
    edges = _compatibility_edges(mask, placements, by_cell)
    if len(edges) > max_edges:
        raise ValueError(f"Exact oracle has {len(edges)} compatibility edges; limit is {max_edges}")
    p_count, e_count = len(placements), len(edges)
    root_candidates = [by_cell[x * n + y] for x, y in starts]

    cursor = 0
    z = np.arange(cursor, cursor + k * p_count).reshape(k, p_count); cursor += k * p_count
    yvar = np.arange(cursor, cursor + k * e_count).reshape(k, e_count); cursor += k * e_count
    flow = np.arange(cursor, cursor + k * e_count * 2).reshape(k, e_count, 2); cursor += k * e_count * 2
    root_flow: List[Dict[int, int]] = []
    for rid in range(k):
        mapping = {}
        for pid in root_candidates[rid]:
            mapping[pid] = cursor
            cursor += 1
        root_flow.append(mapping)
    makespan_var = cursor; cursor += 1

    lower = np.zeros(cursor)
    upper = np.full(cursor, np.inf)
    integrality = np.zeros(cursor, dtype=np.int8)
    upper[z.ravel()] = 1.0; integrality[z.ravel()] = 1
    upper[yvar.ravel()] = 1.0; integrality[yvar.ravel()] = 1
    # A robot cannot own an action covering another robot's protected depot.
    for rid in range(k):
        foreign_roots = set(starts) - {starts[rid]}
        for pid, placement in enumerate(placements):
            if not placement.cells.isdisjoint(foreign_roots):
                upper[z[rid, pid]] = 0.0
    big_m = float(p_count)
    upper[flow.ravel()] = big_m
    for mapping in root_flow:
        upper[list(mapping.values())] = big_m

    rows = _Rows()
    # Every free cell is covered exactly once across all robots and actions.
    for x in range(n):
        for yy in range(n):
            if mask[x, yy]:
                continue
            pids = by_cell[x * n + yy]
            rows.add({int(z[rid, pid]): 1.0 for rid in range(k) for pid in pids}, 1.0, 1.0)
    # Every robot owns the unique action covering its depot.
    for rid in range(k):
        rows.add({int(z[rid, pid]): 1.0 for pid in root_candidates[rid]}, 1.0, 1.0)
        # Together with rooted flow, |E_r|=|P_r|-1 makes the selected
        # connected subgraph a tree and substantially tightens the relaxation.
        tree_count = {int(yvar[rid, eid]): 1.0 for eid in range(e_count)}
        tree_count.update({int(z[rid, pid]): -1.0 for pid in range(p_count)})
        rows.add(tree_count, -1.0, -1.0)

    incident: List[List[Tuple[int, int]]] = [[] for _ in range(p_count)]
    for eid, (p, q, _) in enumerate(edges):
        incident[p].append((eid, 0))  # direction 0 is p -> q
        incident[q].append((eid, 1))  # direction 1 is q -> p
        for rid in range(k):
            edge_y = int(yvar[rid, eid])
            rows.add({edge_y: 1.0, int(z[rid, p]): -1.0}, -np.inf, 0.0)
            rows.add({edge_y: 1.0, int(z[rid, q]): -1.0}, -np.inf, 0.0)
            rows.add({int(flow[rid, eid, 0]): 1.0, edge_y: -big_m}, -np.inf, 0.0)
            rows.add({int(flow[rid, eid, 1]): 1.0, edge_y: -big_m}, -np.inf, 0.0)

    connectivity_cut_count = 0
    if strengthen_connectivity:
        # Static small-set members of the rooted connectivity-cut family.
        # These do not change an integer feasible tree.  They strengthen the
        # single-commodity-flow relaxation by preventing a selected non-root
        # placement, or a selected two-placement island, from satisfying its
        # degree solely through edges internal to that island.
        root_sets = [set(candidates) for candidates in root_candidates]
        for rid in range(k):
            roots = root_sets[rid]
            for pid in range(p_count):
                if pid in roots:
                    continue
                coeffs = {int(z[rid, pid]): -1.0}
                for edge_id, _ in incident[pid]:
                    coeffs[int(yvar[rid, edge_id])] = 1.0
                rows.add(coeffs, 0.0, np.inf)
                connectivity_cut_count += 1

            for internal_eid, (p, q, _) in enumerate(edges):
                pair = {p, q}
                boundary = {
                    edge_id
                    for node in pair
                    for edge_id, _ in incident[node]
                    if edge_id != internal_eid
                    and len(pair.intersection(edges[edge_id][:2])) == 1
                }
                for anchor in (p, q):
                    coeffs = {
                        int(yvar[rid, edge_id]): 1.0 for edge_id in boundary
                    }
                    anchor_col = int(z[rid, anchor])
                    coeffs[anchor_col] = coeffs.get(anchor_col, 0.0) - 1.0
                    for root_pid in pair.intersection(roots):
                        root_col = int(z[rid, root_pid])
                        coeffs[root_col] = coeffs.get(root_col, 0.0) + 1.0
                    rows.add(coeffs, 0.0, np.inf)
                    connectivity_cut_count += 1

    # One unit of rooted commodity is consumed by every selected placement.
    for rid in range(k):
        for pid in range(p_count):
            coeffs = {int(z[rid, pid]): -1.0}
            if pid in root_flow[rid]:
                rf = root_flow[rid][pid]
                coeffs[rf] = 1.0
                rows.add({rf: 1.0, int(z[rid, pid]): -big_m}, -np.inf, 0.0)
            for eid, outward_dir in incident[pid]:
                coeffs[int(flow[rid, eid, outward_dir])] = -1.0
                coeffs[int(flow[rid, eid, 1 - outward_dir])] = 1.0
            rows.add(coeffs, 0.0, 0.0)

    # Min-max service plus a configurable rooted-tree multiplier.  One tree
    # is a lower relaxation of a closed portal walk; two trees are directly
    # executable by edge doubling and shortcutting in the metric closure.
    for rid in range(k):
        coeffs: Dict[int, float] = {makespan_var: -1.0}
        for pid in range(p_count):
            coeffs[int(z[rid, pid])] = service_time_per_tile
        for eid, (_, _, edge_cost) in enumerate(edges):
            coeffs[int(yvar[rid, eid])] = tree_multiplier * edge_cost / robot_speed
        depot = np.asarray(starts[rid], dtype=float) + 0.5
        for pid in root_candidates[rid]:
            coeffs[int(z[rid, pid])] += tree_multiplier * float(
                np.linalg.norm(depot - placements[pid].center)
            ) / robot_speed
        rows.add(coeffs, -np.inf, -float(fixed[rid]))

    objective = np.zeros(cursor)
    objective[makespan_var] = 1.0
    root_lp_bound = math.nan
    root_lp_time = 0.0
    if compute_root_lp_bound:
        root_lp_start = time.perf_counter()
        root_relaxation = milp(
            objective,
            integrality=np.zeros(cursor, dtype=np.int8),
            bounds=Bounds(lower, upper),
            constraints=LinearConstraint(
                rows.matrix(cursor), np.asarray(rows.lb), np.asarray(rows.ub)
            ),
            options={"time_limit": min(3.0, 0.2 * max(0.1, float(time_limit))), "presolve": True},
        )
        root_lp_time = time.perf_counter() - root_lp_start
        if root_relaxation.x is not None:
            root_lp_bound = float(root_relaxation.fun)
    separated_cut_count = 0
    separation_rounds = 0
    separation_start = time.perf_counter()
    if separate_connectivity_cuts:
        # Root-LP separation of the full depot connectivity-cut family:
        #   y(delta(S)) + z(root candidates in S) >= z_p,
        # for every selected placement p in a set S not containing the
        # artificial depot source.  The minimum source-p cut gives the most
        # violated member for p.  Unlike the static singleton/pair family,
        # this adds only cuts violated by the current fractional solution.
        signatures = set()
        separation_budget = min(3.0, 0.25 * max(0.1, float(time_limit)))
        for round_id in range(max(0, int(cut_rounds))):
            elapsed = time.perf_counter() - separation_start
            if elapsed >= separation_budget:
                break
            relaxation_constraint = LinearConstraint(
                rows.matrix(cursor), np.asarray(rows.lb), np.asarray(rows.ub)
            )
            relaxation = milp(
                objective,
                integrality=np.zeros(cursor, dtype=np.int8),
                bounds=Bounds(lower, upper),
                constraints=relaxation_constraint,
                options={
                    "time_limit": max(0.1, separation_budget - elapsed),
                    "presolve": True,
                },
            )
            if relaxation.x is None:
                break
            new_cuts = []
            source = p_count
            for rid in range(k):
                graph = nx.DiGraph()
                graph.add_nodes_from(range(p_count + 1))
                for eid, (p, q, _) in enumerate(edges):
                    capacity = max(0.0, float(relaxation.x[yvar[rid, eid]]))
                    graph.add_edge(p, q, capacity=capacity)
                    graph.add_edge(q, p, capacity=capacity)
                for root_pid in root_candidates[rid]:
                    graph.add_edge(
                        source,
                        root_pid,
                        capacity=max(0.0, float(relaxation.x[z[rid, root_pid]])),
                    )
                roots = set(root_candidates[rid])
                targets = sorted(
                    ((
                        float(relaxation.x[z[rid, pid]]), pid
                    )
                        for pid in range(p_count)
                        if pid not in roots and relaxation.x[z[rid, pid]] > 1e-7
                    ),
                    reverse=True,
                )
                for selected_value, target in targets:
                    cut_value, partition = nx.minimum_cut(
                        graph, source, target, capacity="capacity"
                    )
                    if cut_value + 1e-7 >= selected_value:
                        continue
                    target_side = frozenset(partition[1] - {source})
                    signature = (rid, target_side, target)
                    if signature in signatures:
                        continue
                    signatures.add(signature)
                    coeffs: Dict[int, float] = {int(z[rid, target]): -1.0}
                    for root_pid in roots.intersection(target_side):
                        root_col = int(z[rid, root_pid])
                        coeffs[root_col] = coeffs.get(root_col, 0.0) + 1.0
                    for eid, (p, q, _) in enumerate(edges):
                        if (p in target_side) != (q in target_side):
                            edge_col = int(yvar[rid, eid])
                            coeffs[edge_col] = coeffs.get(edge_col, 0.0) + 1.0
                    new_cuts.append(coeffs)
                    if len(new_cuts) >= max(1, int(cuts_per_round)):
                        break
                if len(new_cuts) >= max(1, int(cuts_per_round)):
                    break
            if not new_cuts:
                break
            for coeffs in new_cuts:
                rows.add(coeffs, 0.0, np.inf)
            separated_cut_count += len(new_cuts)
            separation_rounds = round_id + 1

    separation_time = time.perf_counter() - separation_start
    constraint = LinearConstraint(rows.matrix(cursor), np.asarray(rows.lb), np.asarray(rows.ub))
    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=constraint,
        options={
            "time_limit": max(0.1, float(time_limit) - root_lp_time - separation_time),
            "mip_rel_gap": float(mip_rel_gap),
            "presolve": True,
        },
    )
    if result.x is None:
        raise RuntimeError(f"FACT-MCPP oracle found no incumbent (status={result.status}: {result.message})")

    selected: List[Tile] = []
    assignment = np.full((n, n), -1, dtype=np.int16)
    for rid in range(k):
        for pid, placement in enumerate(placements):
            if result.x[z[rid, pid]] > 0.5:
                selected.append((placement.x, placement.y, placement.w, placement.h, rid))
                for cell in placement.cells:
                    if assignment[cell] >= 0:
                        raise AssertionError("MILP incumbent violates exact cover")
                    assignment[cell] = rid
    assignment[mask] = -1
    if np.any(assignment[~mask] < 0):
        raise AssertionError("MILP incumbent leaves free cells uncovered")

    # Reuse the common safe portal router and validator for the executable tour.
    planner = HybridMCPP(
        n, k, obstacle_mask=mask, robot_starts=starts,
        service_time_per_tile=service_time_per_tile, robot_speed=robot_speed,
        tile_shapes=tile_shapes,
    )
    planner.assignments = assignment
    planner.tiles = selected
    planner.centroids_xy = starts
    planner.route_tsp()
    validity = planner.validate()
    mission_times = [planner.routes[r]["mission_time"] for r in range(k)]
    return {
        "status": int(result.status),
        "message": str(result.message),
        "optimal": int(result.status) == 0,
        "tree_makespan": float(result.fun),
        "model_makespan": float(result.fun),
        "tree_multiplier": float(tree_multiplier),
        "mip_dual_bound": float(getattr(result, "mip_dual_bound", math.nan)),
        "mip_gap": float(getattr(result, "mip_gap", math.nan)),
        "mip_node_count": int(getattr(result, "mip_node_count", 0)),
        "executable_makespan": float(max(mission_times)),
        "mission_times": mission_times,
        "tile_count": len(selected),
        "tiles": selected,
        "assignment": assignment,
        "routes": planner.routes,
        "placements": p_count,
        "compatibility_edges": e_count,
        "variables": cursor,
        "constraints": len(rows.lb),
        "connectivity_cuts": connectivity_cut_count + separated_cut_count,
        "static_connectivity_cuts": connectivity_cut_count,
        "separated_connectivity_cuts": separated_cut_count,
        "separation_rounds": separation_rounds,
        "separation_time": separation_time,
        "root_lp_bound": root_lp_bound,
        "root_lp_time": root_lp_time,
        "tile_shapes": tuple(tile_shapes),
        "build_time": time.perf_counter() - build_start,
        **validity,
    }


def solve_exact_fact_bounds(
    obstacle_mask: np.ndarray,
    robot_starts: Sequence[Cell],
    **kwargs,
) -> dict:
    """Solve certified lower/upper FACT-tree models on the same instance.

    The returned lower bound is valid for the optimal closed portal-graph
    FACT-MCPP makespan.  If the one-tree model times out, its MIP dual bound
    remains valid.  The upper bound is the executable route expanded from the
    doubled-tree configuration, so ``upper_bound / lower_bound`` is an
    end-to-end certificate for the produced route (possibly larger than two
    when either solve is truncated).
    """
    if "tree_multiplier" in kwargs:
        raise TypeError("solve_exact_fact_bounds controls tree_multiplier")
    lower = solve_exact_fact(
        obstacle_mask, robot_starts, tree_multiplier=1.0, **kwargs
    )
    upper = solve_exact_fact(
        obstacle_mask, robot_starts, tree_multiplier=2.0, **kwargs
    )
    lower_bound = (
        lower["model_makespan"]
        if lower["optimal"]
        else lower["mip_dual_bound"]
    )
    executable_upper = upper["executable_makespan"]
    ratio = executable_upper / lower_bound if lower_bound > 0 else math.inf
    return {
        "lower_bound": float(lower_bound),
        "upper_bound": float(executable_upper),
        "certificate_ratio": float(ratio),
        "certified_two_approx": bool(np.isfinite(ratio) and ratio <= 2.0 + 1e-8),
        "lower": lower,
        "upper": upper,
    }


# ============================================================================
# Component-contracted FACT boundary LNS


# ============================================================================
"""Component-contracted exact boundary-band neighborhood for FACT-MCPP.

The outside actions of two neighboring robot regions are frozen.  Removing a
band can split either outside region into several portal-connected components;
each such component is contracted to a mandatory supernode with fixed service
and internal-MST cost.  A local MIP jointly reassigns band cells, selects an
exact non-overlapping action cover, and connects all selected actions and
mandatory components with a rooted tree.  The objective is exact for this
frozen-outside tree neighborhood.
"""


from dataclasses import dataclass
import math
import time
from typing import Dict, List, Sequence, Set, Tuple

import networkx as nx
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp



@dataclass
class _OutsideComponent:
    owner: int
    tiles: Tuple[Tile, ...]
    internal_tree_cost: float
    contains_depot: bool


def _tile_cells(tile: Tile) -> Set[Cell]:
    x, y, w, h, _ = tile
    return {(px, py) for px in range(x, x + w) for py in range(y, y + h)}


def _portal_cost(a: np.ndarray, b: np.ndarray, u: Cell, v: Cell) -> float:
    portal = (np.asarray(u, dtype=float) + np.asarray(v, dtype=float)) / 2.0 + 0.5
    return float(np.linalg.norm(a - portal) + np.linalg.norm(portal - b))


def _outside_components(
    tiles: Sequence[Tile], owner: int, depot: Cell, patch: Set[Cell]
) -> Tuple[List[_OutsideComponent], Dict[Cell, int], Dict[Tile, int]]:
    outside = [tile for tile in tiles if tile[4] == owner and _tile_cells(tile).isdisjoint(patch)]
    if not outside:
        raise ValueError("The contracted neighborhood must leave each robot an outside root component")
    cell_tile: Dict[Cell, int] = {}
    for tid, tile in enumerate(outside):
        for cell in _tile_cells(tile):
            cell_tile[cell] = tid
    graph = nx.Graph()
    graph.add_nodes_from(range(len(outside)))
    edge_weights: Dict[Tuple[int, int], float] = {}
    for cell, tid in cell_tile.items():
        x, y = cell
        for neighbor in ((x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)):
            other = cell_tile.get(neighbor)
            if other is None or other == tid:
                continue
            key = (min(tid, other), max(tid, other))
            ca = np.array((outside[tid][0] + outside[tid][2] / 2,
                           outside[tid][1] + outside[tid][3] / 2), dtype=float)
            cb = np.array((outside[other][0] + outside[other][2] / 2,
                           outside[other][1] + outside[other][3] / 2), dtype=float)
            weight = _portal_cost(ca, cb, cell, neighbor)
            edge_weights[key] = min(edge_weights.get(key, math.inf), weight)
    for (u, v), weight in edge_weights.items():
        graph.add_edge(u, v, weight=weight)

    components: List[_OutsideComponent] = []
    cell_component: Dict[Cell, int] = {}
    tile_component: Dict[Tile, int] = {}
    for nodes in nx.connected_components(graph):
        subgraph = graph.subgraph(nodes)
        mst_cost = float(sum(
            data["weight"] for _, _, data in nx.minimum_spanning_edges(subgraph, data=True)
        )) if len(nodes) > 1 else 0.0
        component_tiles = tuple(outside[node] for node in sorted(nodes))
        cid = len(components)
        components.append(_OutsideComponent(
            owner=owner,
            tiles=component_tiles,
            internal_tree_cost=mst_cost,
            contains_depot=any(depot in _tile_cells(tile) for tile in component_tiles),
        ))
        for tile in component_tiles:
            tile_component[tile] = cid
            for cell in _tile_cells(tile):
                cell_component[cell] = cid
    if sum(component.contains_depot for component in components) != 1:
        raise ValueError("Exactly one outside component must contain the robot depot")
    return components, cell_component, tile_component


def solve_contracted_fact_band(
    obstacle_mask: np.ndarray,
    assignments: np.ndarray,
    tiles: Sequence[Tile],
    robot_starts: Sequence[Cell],
    donor: int,
    receiver: int,
    patch_cells: Sequence[Cell],
    *,
    tile_shapes: Sequence[Tuple[int, int]],
    service_time_per_tile: float = 1.0,
    robot_speed: float = 1.0,
    tree_multiplier: float = 2.0,
    other_robot_makespan: float = 0.0,
    time_limit: float = 5.0,
    mip_rel_gap: float = 0.0,
) -> dict:
    """Exactly reoptimize a two-robot band under frozen outside trees."""
    started = time.perf_counter()
    mask = np.asarray(obstacle_mask, dtype=bool)
    labels = np.asarray(assignments)
    n = mask.shape[0]
    if mask.shape != (n, n) or labels.shape != mask.shape:
        raise ValueError("Square obstacle and assignment grids are required")
    pair = (int(donor), int(receiver))
    if donor == receiver or any(not (0 <= rid < len(robot_starts)) for rid in pair):
        raise ValueError("Invalid donor/receiver pair")
    patch = {tuple(map(int, cell)) for cell in patch_cells}
    if not patch or any(mask[cell] or int(labels[cell]) not in pair for cell in patch):
        raise ValueError("Patch must contain only free cells owned by the robot pair")
    if any(tuple(robot_starts[rid]) in patch for rid in pair):
        raise ValueError("Contracted bands may not contain robot depots")
    for tile in tiles:
        overlap = _tile_cells(tile).intersection(patch)
        if overlap and overlap != _tile_cells(tile):
            raise ValueError("Patch must be a union of complete current actions")

    components_by_robot = []
    outside_cell_components = []
    for rid in pair:
        components, cell_components, _ = _outside_components(
            tiles, rid, tuple(robot_starts[rid]), patch
        )
        components_by_robot.append(components)
        outside_cell_components.append(cell_components)

    full_placements, _ = _placements(mask, tile_shapes)
    placements = [placement for placement in full_placements if placement.cells.issubset(patch)]
    if not placements:
        raise ValueError("No legal coverage actions fit inside the patch")
    p_count = len(placements)
    by_cell: Dict[Cell, List[int]] = {cell: [] for cell in patch}
    for pid, placement in enumerate(placements):
        for cell in placement.cells:
            by_cell[cell].append(pid)
    if any(not candidates for candidates in by_cell.values()):
        raise ValueError("Patch action library cannot cover every band cell")

    # Placement-placement portal edges inside the band.
    pp_weights: Dict[Tuple[int, int], float] = {}
    for u in patch:
        x, y = u
        for v in ((x + 1, y), (x, y + 1)):
            if v not in patch:
                continue
            for p in by_cell[u]:
                for q in by_cell[v]:
                    if p == q or not placements[p].cells.isdisjoint(placements[q].cells):
                        continue
                    key = (min(p, q), max(p, q))
                    weight = _portal_cost(placements[p].center, placements[q].center, u, v)
                    pp_weights[key] = min(pp_weights.get(key, math.inf), weight)

    robot_edges: List[List[Tuple[int, int, float]]] = []
    for local_rid, rid in enumerate(pair):
        components = components_by_robot[local_rid]
        weights = dict(pp_weights)
        component_cells = outside_cell_components[local_rid]
        for u in patch:
            x, y = u
            for v in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                cid = component_cells.get(v)
                if cid is None:
                    continue
                outside_tile = next(
                    tile for tile in components[cid].tiles if v in _tile_cells(tile)
                )
                outside_center = np.array(
                    (outside_tile[0] + outside_tile[2] / 2,
                     outside_tile[1] + outside_tile[3] / 2), dtype=float
                )
                component_node = p_count + cid
                for pid in by_cell[u]:
                    key = (min(pid, component_node), max(pid, component_node))
                    weight = _portal_cost(placements[pid].center, outside_center, u, v)
                    weights[key] = min(weights.get(key, math.inf), weight)
        robot_edges.append([(u, v, weight) for (u, v), weight in weights.items()])

    cursor = 0
    z = np.arange(cursor, cursor + 2 * p_count).reshape(2, p_count); cursor += 2 * p_count
    yvars = []
    flows = []
    for edges in robot_edges:
        yv = np.arange(cursor, cursor + len(edges)); cursor += len(edges)
        fv = np.arange(cursor, cursor + 2 * len(edges)).reshape(len(edges), 2)
        cursor += 2 * len(edges)
        yvars.append(yv); flows.append(fv)
    makespan_var = cursor; cursor += 1

    lower = np.zeros(cursor)
    upper = np.full(cursor, np.inf)
    integrality = np.zeros(cursor, dtype=np.int8)
    upper[z.ravel()] = 1.0; integrality[z.ravel()] = 1
    for yv, fv in zip(yvars, flows):
        upper[yv] = 1.0; integrality[yv] = 1
        upper[fv.ravel()] = p_count + max(len(c) for c in components_by_robot)

    rows = _Rows()
    for cell, candidates in by_cell.items():
        rows.add({int(z[r, pid]): 1.0 for r in range(2) for pid in candidates}, 1.0, 1.0)

    fixed_costs = []
    for local_rid, rid in enumerate(pair):
        components = components_by_robot[local_rid]
        edges = robot_edges[local_rid]
        yv, fv = yvars[local_rid], flows[local_rid]
        component_count = len(components)
        big_m = float(p_count + component_count)
        incident: List[List[Tuple[int, int]]] = [
            [] for _ in range(p_count + component_count)
        ]
        for eid, (u, v, _) in enumerate(edges):
            incident[u].append((eid, 0)); incident[v].append((eid, 1))
            if u < p_count:
                rows.add({int(yv[eid]): 1.0, int(z[local_rid, u]): -1.0}, -np.inf, 0.0)
            if v < p_count:
                rows.add({int(yv[eid]): 1.0, int(z[local_rid, v]): -1.0}, -np.inf, 0.0)
            rows.add({int(fv[eid, 0]): 1.0, int(yv[eid]): -big_m}, -np.inf, 0.0)
            rows.add({int(fv[eid, 1]): 1.0, int(yv[eid]): -big_m}, -np.inf, 0.0)
        tree_count = {int(yv[eid]): 1.0 for eid in range(len(edges))}
        tree_count.update({int(z[local_rid, pid]): -1.0 for pid in range(p_count)})
        rows.add(tree_count, component_count - 1, component_count - 1)

        root_cid = next(i for i, component in enumerate(components) if component.contains_depot)
        for node in range(p_count + component_count):
            coeffs: Dict[int, float] = {}
            for eid, outward in incident[node]:
                coeffs[int(fv[eid, outward])] = -1.0
                coeffs[int(fv[eid, 1 - outward])] = 1.0
            if node < p_count:
                coeffs[int(z[local_rid, node])] = -1.0
                rows.add(coeffs, 0.0, 0.0)
            elif node - p_count != root_cid:
                rows.add(coeffs, 1.0, 1.0)
            else:
                # Root supplies every selected placement and other component.
                coeffs = {col: -value for col, value in coeffs.items()}
                for pid in range(p_count):
                    coeffs[int(z[local_rid, pid])] = -1.0
                rows.add(coeffs, component_count - 1, component_count - 1)

        outside_tiles = [tile for component in components for tile in component.tiles]
        root_tile = next(tile for tile in outside_tiles if tuple(robot_starts[rid]) in _tile_cells(tile))
        root_center = np.array(
            (root_tile[0] + root_tile[2] / 2, root_tile[1] + root_tile[3] / 2), dtype=float
        )
        depot = np.asarray(robot_starts[rid], dtype=float) + 0.5
        fixed_cost = (
            service_time_per_tile * len(outside_tiles)
            + tree_multiplier * (
                sum(component.internal_tree_cost for component in components)
                + float(np.linalg.norm(depot - root_center))
            ) / robot_speed
        )
        fixed_costs.append(float(fixed_cost))

    objective = np.zeros(cursor)
    objective[makespan_var] = 1.0
    for local_rid in range(2):
        coeffs = {makespan_var: -1.0}
        for pid in range(p_count):
            coeffs[int(z[local_rid, pid])] = service_time_per_tile
        for eid, (_, _, weight) in enumerate(robot_edges[local_rid]):
            coeffs[int(yvars[local_rid][eid])] = tree_multiplier * weight / robot_speed
        rows.add(coeffs, -np.inf, -fixed_costs[local_rid])
    lower[makespan_var] = max(0.0, float(other_robot_makespan))

    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(rows.matrix(cursor), np.asarray(rows.lb), np.asarray(rows.ub)),
        options={"time_limit": max(0.1, float(time_limit)), "mip_rel_gap": float(mip_rel_gap), "presolve": True},
    )
    if result.x is None:
        raise RuntimeError(f"Contracted FACT band found no incumbent: {result.message}")

    selected_tiles: List[Tile] = []
    patch_assignment: Dict[Cell, int] = {}
    for local_rid, rid in enumerate(pair):
        for pid, placement in enumerate(placements):
            if result.x[z[local_rid, pid]] > 0.5:
                selected_tiles.append((placement.x, placement.y, placement.w, placement.h, rid))
                for cell in placement.cells:
                    if cell in patch_assignment:
                        raise AssertionError("Contracted incumbent overlaps patch actions")
                    patch_assignment[cell] = rid
    if set(patch_assignment) != patch:
        raise AssertionError("Contracted incumbent does not exactly cover the patch")
    outside_tiles = [tile for tile in tiles if _tile_cells(tile).isdisjoint(patch)]
    new_tiles = outside_tiles + selected_tiles
    new_assignments = labels.copy()
    for cell, rid in patch_assignment.items():
        new_assignments[cell] = rid
    return {
        "assignment": new_assignments,
        "tiles": new_tiles,
        "model_makespan": float(result.fun),
        "optimal": int(result.status) == 0,
        "mip_gap": float(getattr(result, "mip_gap", math.nan)),
        "mip_dual_bound": float(getattr(result, "mip_dual_bound", math.nan)),
        "patch_cells": len(patch),
        "patch_placements": p_count,
        "component_counts": tuple(len(items) for items in components_by_robot),
        "fixed_costs": tuple(fixed_costs),
        "variables": cursor,
        "constraints": len(rows.lb),
        "runtime": time.perf_counter() - started,
    }


FACTMCCA = HybridMCPP
__all__ = [
    'FACTMCCA', 'HybridMCPP', 'make_random_map', 'TILE_SHAPES',
    'solve_exact_fact', 'solve_exact_fact_bounds',
    'solve_contracted_fact_band',
]
