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
from dataclasses import dataclass
import heapq
import math
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra as sparse_dijkstra
from scipy.optimize import Bounds, LinearConstraint, milp


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
        self.assignments: Optional[np.ndarray] = None
        self.tiles: List[Tile] = []
        self.routes: Dict[int, dict] = {}
        self.centroids_xy: List[Cell] = []

    # ------------------------------- partition ----------------------------
    def _initial_seeds(self) -> List[int]:
        """Grid seeds compatible with the uploaded program, snapped to free cells."""
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
        sizes = np.bincount(flat[self.free_flat], minlength=self.k).astype(int)
        for _ in range(max_passes):
            changed = 0
            boundary = self.free_flat.copy()
            self.rng.shuffle(boundary)
            for u in boundary:
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
        d = self._seed_distances(seeds)
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
            local = np.argmin(d + lam[:, None], axis=0).astype(np.int16)
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
        for w, h in TILE_SHAPES:
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
        for w, h in TILE_SHAPES:
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
        for w, h in TILE_SHAPES + ((1, 1),):
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

    def route_tsp(self) -> Dict[int, dict]:
        routes = {}
        for rid in range(self.k):
            tiles = [t for t in self.tiles if t[4] == rid]
            if not tiles:
                routes[rid] = {"order": [], "centers": [], "length": 0.0}
                continue
            centers, d, predecessors, edge_portals = self._tile_geodesic_metric(tiles)
            start_point = np.array(self.centroids_xy[rid], dtype=float) + 0.5
            start = int(np.argmin([np.linalg.norm(c - start_point) for c in centers]))
            order = [start]
            unused = np.ones(len(tiles), dtype=bool)
            unused[start] = False
            while unused.any():
                costs = d[order[-1]].copy()
                costs[~unused] = np.inf
                nxt = int(np.argmin(costs))
                order.append(nxt)
                unused[nxt] = False
            order = self._two_opt(order, d, passes=2 if len(order) > 1000 else 3)
            length = sum(d[order[i], order[(i + 1) % len(order)]] for i in range(len(order))) if len(order) > 1 else 0.0
            routes[rid] = {
                "order": order,
                "centers": [centers[i] for i in order],
                "path": self._expand_metric_tour(order, predecessors, centers, edge_portals),
                "length": float(length),
            }
        self.routes = routes
        return routes

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
                routes[rid] = {"order": [], "centers": [], "path": [], "length": 0.0}
                continue
            if len(tiles) == 1:
                center = [tiles[0][0] + tiles[0][2] / 2, tiles[0][1] + tiles[0][3] / 2]
                routes[rid] = {"order": [0], "centers": [center], "path": [center], "length": 0.0}
                continue
            owner = {}
            centers = []
            for i, t in enumerate(tiles):
                centers.append(np.array((t[0] + t[2] / 2, t[1] + t[3] / 2)))
                for c in self._tile_cells(t):
                    owner[c] = i
            edges = {}
            for c, i in owner.items():
                x, y = c
                for nb in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
                    j = owner.get(nb)
                    if j is not None and i != j:
                        e = (min(i, j), max(i, j))
                        edges[e] = float(np.linalg.norm(centers[i] - centers[j]))
            parent = list(range(len(tiles)))
            def find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            weight = 0.0
            tree = [[] for _ in tiles]
            for (u, v), w in sorted(edges.items(), key=lambda z: z[1]):
                ru, rv = find(u), find(v)
                if ru != rv:
                    parent[ru] = rv
                    weight += w
                    tree[u].append(v)
                    tree[v].append(u)
            # Iterative Euler/DFS walk: every tree edge is traversed twice.
            walk = [0]
            stack = [(0, -1, iter(tree[0]))]
            while stack:
                node, parent_node, children = stack[-1]
                try:
                    child = next(children)
                    if child == parent_node:
                        continue
                    walk.append(child)
                    stack.append((child, node, iter(tree[child])))
                except StopIteration:
                    stack.pop()
                    if stack:
                        walk.append(stack[-1][0])
            robot_length = 2.0 * weight
            center_lists = [c.tolist() for c in centers]
            routes[rid] = {
                "order": walk,
                "centers": center_lists,
                "path": [center_lists[i] for i in walk],
                "length": robot_length,
            }
            total += robot_length
        self.routes = routes
        return total

    # -------------------------------- public -------------------------------
    def _partition_sizes(self) -> np.ndarray:
        return np.array([np.count_nonzero(self.assignments == rid) for rid in range(self.k)])

    def validate(self) -> dict:
        covered = np.zeros((self.n, self.n), dtype=np.int16)
        for x, y, w, h, rid in self.tiles:
            if not (self.assignments[x:x+w, y:y+h] == rid).all():
                raise AssertionError("Tile crosses an obstacle or partition boundary")
            covered[x:x+w, y:y+h] += 1
        if not np.all(covered[~self.obstacles] == 1) or np.any(covered[self.obstacles]):
            raise AssertionError("Tiling is not an exact cover")
        connected = []
        for rid in range(self.k):
            cells = np.flatnonzero(self.assignments.ravel() == rid)
            seen = {int(cells[0])}
            q = deque(seen)
            while q:
                for v in _neighbors(q.popleft(), self.n):
                    if self.assignments.ravel()[v] == rid and v not in seen:
                        seen.add(v); q.append(v)
            connected.append(len(seen) == len(cells))
        return {"exact_cover": True, "connected_regions": all(connected)}

    def solve(self, method: str = "hybrid") -> dict:
        times = StageTimes()
        t = time.perf_counter()
        if method == "baseline":
            self.partition_baseline()
        elif method == "hybrid":
            self.partition_balanced()
        else:
            raise ValueError("method must be 'baseline' or 'hybrid'")
        times.partition = time.perf_counter() - t
        t = time.perf_counter()
        self.tile_baseline() if method == "baseline" else self.tile_hybrid()
        times.tiling = time.perf_counter() - t
        t = time.perf_counter()
        if method == "baseline":
            path_length = self.route_mst_walk_length()
        else:
            self.route_tsp()
            path_length = sum(r["length"] for r in self.routes.values())
        times.routing = time.perf_counter() - t
        valid = self.validate()
        sizes = self._partition_sizes()
        tile_loads = np.array([sum(t[4] == rid for t in self.tiles) for rid in range(self.k)])
        path_loads = np.array([self.routes.get(rid, {}).get("length", 0.0) for rid in range(self.k)])
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
            "partition_time": times.partition,
            "tiling_time": times.tiling,
            "routing_time": times.routing,
            "total_time": times.total,
            **valid,
        }


if __name__ == "__main__":
    mask = make_random_map(40, 0.10, 42)
    for name in ("baseline", "hybrid"):
        solver = HybridMCPP(40, 5, obstacle_mask=mask, seed=42, tiling_time_limit=4.0)
        print(solver.solve(name))
