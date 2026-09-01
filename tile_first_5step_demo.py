"""
Tile-First MCPP — 7-step visual demo (8x8, 3 robots)
=====================================================

Purpose
-------
A self-contained teaching/visualization script distilled from the core flow of
mainline_tile_first_v2_3_2.py.  The algorithmic story is intentionally limited
to SEVEN reader-facing steps:

    1) Input map
    2) Global tile-first exact cover
    3) Connected balanced initial partition
    4) Connectivity-preserving tile-transfer balancing by closed-tour CV
    5) Obstacle-safe closed patrol cycles (raw metric expansion)
    6) Strict shortcut postprocess and final patrol paths
    7) Obstacle-safe curved closed loops with explicit z-layer crossings

Only the COLOR / DRAWING STYLE is inspired by voronoi-Adapt-MST.py:
    Robot 1 = #FF6B6B
    Robot 2 = #4ECDC4
    Robot 3 = #45B7D1
    obstacles = black
    regions/tiles = semi-transparent colored blocks with black boundaries

Important output rule
---------------------
Every algorithm step is saved as ONE INDEPENDENT SQUARE PNG.  No multi-panel
figure is generated.

Running this file creates a folder next to this .py file:
    tile_first_5step_output/

Files created:
    step_01_input_map.png
    step_02_global_tiling.png
    step_03_connected_initial_partition.png
    step_04_cv_balanced_partition.png
    step_05_final_closed_cycles.png
    step_06_shortcut_final_paths.png
    step_07_curved_final_closed_loops.png
    robot_01_closed_cycle.png
    robot_02_closed_cycle.png
    robot_03_closed_cycle.png
    route_data.json
    route_summary.txt

Dependencies: numpy, matplotlib, scipy
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
import heapq
import json
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy import sparse


# ============================================================================
# 0. Fixed demo configuration
# ============================================================================
MAP_SIZE = 8
ROBOT_NUM = 3
OBSTACLE_NUM = 10       # Number of random obstacle cells
RANDOM_SEED = 89       # Change ONLY this value to get a new obstacle layout,89

# Same first-three robot colors as voronoi-Adapt-MST.py.
ROBOT_COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
CURVED_ROUTE_COLORS = ["#C00000", "#3B5F21", "#2E54A1"]
OBSTACLE_COLOR = "#000000"
FREE_COLOR = "#FFFFFF"
GRID_COLOR = "#252525"
NEUTRAL_TILE_COLOR = "#ECECEC"
TEXT_COLOR = "#111111"

# Supported rectangle shapes.  1x1 is included as the exact-cover fallback.
TILE_SHAPES = [(4, 4), (4, 2), (2, 4), (2, 2), (2, 1), (1, 2), (1, 1)]

# Obstacles are generated from RANDOM_SEED at runtime.  They are NOT hard-coded.
# Coordinates are (x, y), origin at lower-left.
OBSTACLES: Set[Tuple[int, int]] = set()

# Reader-facing optimization limit.  8x8 is tiny, so exact candidate evaluation
# is fast while still making the logic transparent.
MAX_TRANSFER_ITERATIONS = 30

# STEP 7 curve generation and rendering.  The curve still interpolates every
# STEP-6 waypoint.  Tangents are reduced only where rounding would leave the
# robot region or touch a closed obstacle cell.
CURVE_SAMPLES_PER_SEGMENT = 28
CURVE_INITIAL_TANGENT_SCALE = 0.90
CURVE_MAX_SAFETY_PASSES = 18

# STEP 7 tube appearance.  The colored center line keeps the same thickness as
# STEP 5/6.  A SOLID darker casing is drawn around it continuously.
CURVED_ROUTE_LINEWIDTH = 6.0
CURVED_ROUTE_OUTLINE_LINEWIDTH = 11.5
CURVED_ROUTE_OUTLINE_DARKEN = 0.62

# Z-layer rule for self intersections: later-traversed Hermite pieces are on
# top of earlier pieces.  Each piece uses two adjacent z levels: casing first,
# colored center line second.
CURVED_ROUTE_Z_STEP = 0.04

OUTPUT_DIR = Path(__file__).resolve().parent / "tile_first_5step_output"

Cell = Tuple[int, int]
Tile = Tuple[int, int, int, int]  # x, y, width, height


@dataclass
class TileGraph:
    adjacency: List[Set[int]]
    centers: List[np.ndarray]
    edge_weights: Dict[Tuple[int, int], float]
    edge_portals: Dict[Tuple[int, int], np.ndarray]


# ============================================================================
# Shared helpers
# ============================================================================
def coefficient_of_variation(values: Sequence[float]) -> float:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return 0.0
    mean = float(a.mean())
    return float(a.std() / mean) if mean > 1e-12 else 0.0


def tile_cells(tile: Tile) -> List[Cell]:
    x, y, w, h = tile
    return [(px, py) for px in range(x, x + w) for py in range(y, y + h)]


def tile_center(tile: Tile) -> np.ndarray:
    x, y, w, h = tile
    return np.array([x + w / 2.0, y + h / 2.0], dtype=float)


def map_is_connected(obstacles: Set[Cell], map_size: int = MAP_SIZE) -> bool:
    free = {(x, y) for x in range(map_size) for y in range(map_size)} - obstacles
    if not free:
        return False
    start = next(iter(free))
    seen = {start}
    stack = [start]
    while stack:
        x, y = stack.pop()
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if (nx, ny) in free and (nx, ny) not in seen:
                seen.add((nx, ny))
                stack.append((nx, ny))
    return len(seen) == len(free)


def generate_random_obstacles(
    map_size: int, obstacle_num: int, seed: int, max_trials: int = 10000
) -> Set[Cell]:
    """Generate seed-controlled random obstacles while keeping free space connected.

    The same seed always produces the same obstacle layout.  Changing the seed
    changes the random sequence and therefore normally changes obstacle positions.
    A candidate is accepted only when all remaining free cells form one 4-neighbor
    connected component, which is required by the later tile/partition/route stages.
    """
    cell_count = map_size * map_size
    if obstacle_num < 0:
        raise ValueError("OBSTACLE_NUM must be >= 0.")
    if obstacle_num > cell_count - ROBOT_NUM:
        raise ValueError(
            f"OBSTACLE_NUM={obstacle_num} leaves fewer free cells than robots."
        )
    if obstacle_num == 0:
        return set()

    rng = np.random.default_rng(seed)
    all_cells = [(x, y) for x in range(map_size) for y in range(map_size)]

    for _ in range(max_trials):
        chosen = rng.choice(cell_count, size=obstacle_num, replace=False)
        obstacles = {all_cells[int(i)] for i in chosen}
        if map_is_connected(obstacles, map_size):
            return obstacles

    raise RuntimeError(
        f"Could not generate a connected random map after {max_trials} trials. "
        "Reduce OBSTACLE_NUM or use another RANDOM_SEED."
    )


def validate_configuration(obstacles: Set[Cell]) -> None:
    if MAP_SIZE != 8:
        raise ValueError("This teaching file is intentionally fixed to an 8x8 map.")
    if ROBOT_NUM != 3:
        raise ValueError("This teaching file is intentionally fixed to 3 robots.")
    if len(obstacles) != OBSTACLE_NUM:
        raise ValueError("Generated obstacle count does not match OBSTACLE_NUM.")
    if not map_is_connected(obstacles, MAP_SIZE):
        raise ValueError("The generated free space must be 4-neighbor connected.")


# ============================================================================
# STEP 2 core: global tile-first exact cover
# ============================================================================
def enumerate_legal_tiles(obstacles: Set[Cell]) -> List[Tile]:
    """Enumerate every legal placement of the allowed rectangle shapes."""
    candidates: List[Tile] = []
    for w, h in TILE_SHAPES:
        for x in range(MAP_SIZE - w + 1):
            for y in range(MAP_SIZE - h + 1):
                t = (x, y, w, h)
                if all(c not in obstacles for c in tile_cells(t)):
                    candidates.append(t)
    return candidates


def greedy_tiling(obstacles: Set[Cell], x_reverse: bool, y_reverse: bool) -> List[Tile]:
    """Large-to-small greedy exact cover used as a transparent baseline."""
    covered: Set[Cell] = set()
    out: List[Tile] = []
    x_order = list(range(MAP_SIZE))
    y_order = list(range(MAP_SIZE))
    if x_reverse:
        x_order.reverse()
    if y_reverse:
        y_order.reverse()

    for w, h in TILE_SHAPES:
        for x in x_order:
            for y in y_order:
                if x + w > MAP_SIZE or y + h > MAP_SIZE:
                    continue
                t = (x, y, w, h)
                cells = tile_cells(t)
                if any(c in obstacles or c in covered for c in cells):
                    continue
                out.append(t)
                covered.update(cells)

    free = {(x, y) for x in range(MAP_SIZE) for y in range(MAP_SIZE)} - obstacles
    if covered != free:
        raise AssertionError("Greedy tiling failed to exactly cover free cells.")
    return out


def milp_exact_tiling(obstacles: Set[Cell]) -> Optional[List[Tile]]:
    """Minimum-cardinality exact rectangle cover for the small 8x8 teaching map."""
    candidates = enumerate_legal_tiles(obstacles)
    free_cells = sorted(
        {(x, y) for x in range(MAP_SIZE) for y in range(MAP_SIZE)} - obstacles
    )
    row_of = {c: i for i, c in enumerate(free_cells)}

    rows: List[int] = []
    cols: List[int] = []
    for j, t in enumerate(candidates):
        for c in tile_cells(t):
            rows.append(row_of[c])
            cols.append(j)

    A = sparse.coo_matrix(
        (np.ones(len(rows), dtype=float), (rows, cols)),
        shape=(len(free_cells), len(candidates)),
    ).tocsr()

    # Primary objective: minimum tile count.
    # Tiny deterministic tie-breaks prefer larger rectangles, then lower x/y.
    c = np.array(
        [
            1.0
            + (16 - t[2] * t[3]) * 1e-5
            + t[0] * 1e-7
            + t[1] * 1e-8
            for t in candidates
        ],
        dtype=float,
    )

    result = milp(
        c,
        integrality=np.ones(len(candidates), dtype=int),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(A, 1.0, 1.0),
        options={"time_limit": 5.0, "mip_rel_gap": 0.0, "presolve": True},
    )
    if result.x is None:
        return None
    chosen = [candidates[i] for i, z in enumerate(result.x) if z > 0.5]
    return chosen


def global_tile_first(obstacles: Set[Cell]) -> List[Tile]:
    """Multi-direction greedy + exact MILP refinement, then choose the best cover."""
    schemes = [
        greedy_tiling(obstacles, False, False),
        greedy_tiling(obstacles, True, True),
        greedy_tiling(obstacles, False, True),
        greedy_tiling(obstacles, True, False),
    ]
    exact = milp_exact_tiling(obstacles)
    if exact is not None:
        schemes.append(exact)
    tiles = min(schemes, key=lambda s: (len(s), -sum(t[2] * t[3] for t in s)))
    return sorted(tiles, key=lambda t: (t[0], t[1], -t[2] * t[3], t[2], t[3]))


# ============================================================================
# Tile adjacency + obstacle-safe center/portal geometry
# ============================================================================
def build_tile_graph(tiles: Sequence[Tile], obstacles: Set[Cell]) -> TileGraph:
    cell_owner: Dict[Cell, int] = {}
    for tid, t in enumerate(tiles):
        for c in tile_cells(t):
            if c in cell_owner:
                raise AssertionError("Global tiling overlaps.")
            if c in obstacles:
                raise AssertionError("A tile covers an obstacle.")
            cell_owner[c] = tid

    free = {(x, y) for x in range(MAP_SIZE) for y in range(MAP_SIZE)} - obstacles
    if set(cell_owner) != free:
        raise AssertionError("Global tiling is not an exact cover.")

    adjacency: List[Set[int]] = [set() for _ in tiles]
    centers = [tile_center(t) for t in tiles]
    edge_weights: Dict[Tuple[int, int], float] = {}
    edge_portals: Dict[Tuple[int, int], np.ndarray] = {}

    for (x, y), i in cell_owner.items():
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            j = cell_owner.get((nx, ny))
            if j is None or j == i:
                continue
            adjacency[i].add(j)
            adjacency[j].add(i)
            e = (min(i, j), max(i, j))

            if nx == x + 1:
                portal = np.array([x + 1.0, y + 0.5])
            elif nx == x - 1:
                portal = np.array([x + 0.0, y + 0.5])
            elif ny == y + 1:
                portal = np.array([x + 0.5, y + 1.0])
            else:
                portal = np.array([x + 0.5, y + 0.0])

            w = float(np.linalg.norm(centers[i] - portal) + np.linalg.norm(portal - centers[j]))
            if e not in edge_weights or w < edge_weights[e] - 1e-12:
                edge_weights[e] = w
                edge_portals[e] = portal

    if not tiles:
        raise AssertionError("No tiles produced.")
    seen = {0}
    stack = [0]
    while stack:
        u = stack.pop()
        for v in adjacency[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    if len(seen) != len(tiles):
        raise AssertionError("The global tile graph is disconnected.")

    return TileGraph(adjacency, centers, edge_weights, edge_portals)


def edge_weight(graph: TileGraph, u: int, v: int) -> float:
    return graph.edge_weights[(min(u, v), max(u, v))]


def dijkstra_on_subset(
    graph: TileGraph, source: int, allowed: Set[int]
) -> Tuple[Dict[int, float], Dict[int, int]]:
    dist = {u: math.inf for u in allowed}
    prev: Dict[int, int] = {}
    dist[source] = 0.0
    pq = [(0.0, source)]
    while pq:
        du, u = heapq.heappop(pq)
        if du > dist[u] + 1e-12:
            continue
        for v in graph.adjacency[u]:
            if v not in allowed:
                continue
            nd = du + edge_weight(graph, u, v)
            if nd + 1e-12 < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, prev


def all_pairs_metric(
    graph: TileGraph, tile_ids: Sequence[int]
) -> Tuple[np.ndarray, List[Dict[int, int]]]:
    ids = list(tile_ids)
    allowed = set(ids)
    local_of = {gid: i for i, gid in enumerate(ids)}
    d = np.full((len(ids), len(ids)), np.inf, dtype=float)
    prevs: List[Dict[int, int]] = []
    for i, source in enumerate(ids):
        dist, prev = dijkstra_on_subset(graph, source, allowed)
        prevs.append(prev)
        for gid, val in dist.items():
            d[i, local_of[gid]] = val
    if not np.isfinite(d).all():
        raise AssertionError("A robot tile subset is disconnected.")
    return d, prevs


def reconstruct_global_path(source: int, target: int, prev: Dict[int, int]) -> List[int]:
    if source == target:
        return [source]
    rev = [target]
    cur = target
    while cur != source:
        if cur not in prev:
            raise AssertionError("Missing predecessor in finite path.")
        cur = prev[cur]
        rev.append(cur)
    return list(reversed(rev))


# ============================================================================
# STEP 3 core: connected balanced initial partition
# ============================================================================
def choose_partition_seeds(tiles: Sequence[Tile], graph: TileGraph) -> List[int]:
    areas = np.asarray([t[2] * t[3] for t in tiles], dtype=float)
    centers = np.asarray(graph.centers)
    workspace_center = np.average(centers, axis=0, weights=areas)
    first = int(np.argmin(np.linalg.norm(centers - workspace_center, axis=1)))
    seeds = [first]

    all_ids = list(range(len(tiles)))
    for _ in range(1, ROBOT_NUM):
        min_dist = np.full(len(tiles), np.inf, dtype=float)
        for s in seeds:
            dist, _ = dijkstra_on_subset(graph, s, set(all_ids))
            for gid, val in dist.items():
                min_dist[gid] = min(min_dist[gid], val)
        min_dist[seeds] = -np.inf
        seeds.append(int(np.argmax(min_dist)))
    return seeds


def connected_balanced_growth(
    tiles: Sequence[Tile], graph: TileGraph, seeds: Sequence[int]
) -> np.ndarray:
    """Grow each robot only through adjacent tiles while balancing covered area."""
    areas = np.asarray([t[2] * t[3] for t in tiles], dtype=float)
    target = float(areas.sum()) / ROBOT_NUM
    owners = np.full(len(tiles), -1, dtype=np.int16)
    loads = np.zeros(ROBOT_NUM, dtype=float)
    frontiers: List[List[Tuple[float, int]]] = [[] for _ in range(ROBOT_NUM)]
    best_cost = np.full((ROBOT_NUM, len(tiles)), np.inf, dtype=float)

    for rid, seed in enumerate(seeds):
        owners[seed] = rid
        loads[rid] = areas[seed]
        best_cost[rid, seed] = 0.0

    for rid, seed in enumerate(seeds):
        for v in graph.adjacency[seed]:
            if owners[v] < 0:
                c = edge_weight(graph, seed, v)
                best_cost[rid, v] = c
                heapq.heappush(frontiers[rid], (c, v))

    remaining = int(np.count_nonzero(owners < 0))
    while remaining:
        options: List[Tuple[float, int]] = []
        for rid in range(ROBOT_NUM):
            while frontiers[rid] and owners[frontiers[rid][0][1]] >= 0:
                heapq.heappop(frontiers[rid])
            if frontiers[rid]:
                options.append((loads[rid] / max(target, 1e-12), rid))
        if not options:
            raise AssertionError("Connected growth cannot reach all tiles.")

        _, rid = min(options)
        cost, u = heapq.heappop(frontiers[rid])
        if owners[u] >= 0:
            continue
        owners[u] = rid
        loads[rid] += areas[u]
        remaining -= 1

        for v in graph.adjacency[u]:
            if owners[v] >= 0:
                continue
            new_cost = cost + edge_weight(graph, u, v)
            if new_cost + 1e-12 < best_cost[rid, v]:
                best_cost[rid, v] = new_cost
                heapq.heappush(frontiers[rid], (new_cost, v))

    return owners


def owner_connected(graph: TileGraph, owners: np.ndarray, rid: int) -> bool:
    nodes = [i for i, r in enumerate(owners) if int(r) == rid]
    if not nodes:
        return False
    allowed = set(nodes)
    seen = {nodes[0]}
    stack = [nodes[0]]
    while stack:
        u = stack.pop()
        for v in graph.adjacency[u]:
            if v in allowed and v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == len(nodes)


def can_transfer(graph: TileGraph, owners: np.ndarray, tid: int, receiver: int) -> bool:
    donor = int(owners[tid])
    if donor == receiver:
        return False
    donor_nodes = np.flatnonzero(owners == donor)
    if len(donor_nodes) <= 1:
        return False
    if not any(int(owners[v]) == receiver for v in graph.adjacency[tid]):
        return False

    trial = owners.copy()
    trial[tid] = receiver
    return owner_connected(graph, trial, donor) and owner_connected(graph, trial, receiver)


# ============================================================================
# Closed metric-TSP route used both for STEP 4 objective and STEP 5 geometry
# ============================================================================
def two_opt(order: List[int], d: np.ndarray, passes: int = 5) -> List[int]:
    n = len(order)
    if n < 4:
        return order
    order = list(order)
    for _ in range(passes):
        improved = False
        for i in range(n - 1):
            a = order[i]
            b = order[(i + 1) % n]
            for j in range(i + 2, n if i > 0 else n - 1):
                c = order[j]
                e = order[(j + 1) % n]
                old = d[a, b] + d[c, e]
                new = d[a, c] + d[b, e]
                if new + 1e-9 < old:
                    order[i + 1 : j + 1] = reversed(order[i + 1 : j + 1])
                    improved = True
        if not improved:
            break
    return order


def tour_length(order: Sequence[int], d: np.ndarray) -> float:
    if len(order) <= 1:
        return 0.0
    return float(sum(d[order[i], order[(i + 1) % len(order)]] for i in range(len(order))))


def mst_preorder(d: np.ndarray, start: int) -> List[int]:
    """Prim MST on the metric closure, then DFS preorder."""
    n = len(d)
    if n <= 1:
        return [start] if n else []
    in_tree = np.zeros(n, dtype=bool)
    key = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=int)
    key[start] = 0.0
    for _ in range(n):
        masked = np.where(in_tree, np.inf, key)
        u = int(np.argmin(masked))
        in_tree[u] = True
        for v in range(n):
            if not in_tree[v] and d[u, v] + 1e-12 < key[v]:
                key[v] = d[u, v]
                parent[v] = u

    adj = [[] for _ in range(n)]
    for v in range(n):
        if parent[v] >= 0:
            adj[v].append(int(parent[v]))
            adj[int(parent[v])].append(v)
    for a in adj:
        a.sort()

    out: List[int] = []
    seen: Set[int] = set()
    stack = [start]
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
        for v in reversed(adj[u]):
            if v not in seen:
                stack.append(v)
    return out


def closed_route_for_tiles(
    tile_ids: Sequence[int],
    tiles: Sequence[Tile],
    graph: TileGraph,
    expand_geometry: bool,
) -> Dict[str, object]:
    ids = list(tile_ids)
    if not ids:
        return {"order_global": [], "length": 0.0, "path": []}
    if len(ids) == 1:
        c = graph.centers[ids[0]].tolist()
        # Explicitly repeat the start to make the cycle visible in serialized data.
        return {"order_global": ids, "length": 0.0, "path": [c, c] if expand_geometry else []}

    d, prevs = all_pairs_metric(graph, ids)
    center_arr = np.asarray([graph.centers[gid] for gid in ids])
    start = int(np.argmin(np.linalg.norm(center_arr - center_arr.mean(axis=0), axis=1)))

    # Candidate A: nearest-neighbor cycle.
    nn = [start]
    unused = set(range(len(ids))) - {start}
    while unused:
        last = nn[-1]
        nxt = min(unused, key=lambda j: (d[last, j], j))
        nn.append(nxt)
        unused.remove(nxt)

    # Candidate B: MST preorder cycle.
    preorder = mst_preorder(d, start)
    candidates = [two_opt(nn, d), two_opt(preorder, d)]
    order_local = min(candidates, key=lambda o: (tour_length(o, d), tuple(o)))
    length = tour_length(order_local, d)
    order_global = [ids[i] for i in order_local]

    if not expand_geometry:
        return {"order_global": order_global, "length": length, "path": []}

    # Expand every TSP edge through shortest adjacent-tile paths and shared-edge
    # portals.  Therefore every segment lies inside one assigned rectangle or on
    # the shared boundary between two assigned rectangles.
    path: List[List[float]] = []
    local_of = {gid: i for i, gid in enumerate(ids)}
    for k, source_gid in enumerate(order_global):
        target_gid = order_global[(k + 1) % len(order_global)]
        source_local = local_of[source_gid]
        global_chain = reconstruct_global_path(source_gid, target_gid, prevs[source_local])

        if not path:
            path.append(graph.centers[source_gid].tolist())
        for a, b in zip(global_chain, global_chain[1:]):
            portal = graph.edge_portals[(min(a, b), max(a, b))]
            # Avoid exact duplicate consecutive points.
            for p in (portal, graph.centers[b]):
                p_list = [float(p[0]), float(p[1])]
                if not path or np.linalg.norm(np.asarray(path[-1]) - p) > 1e-12:
                    path.append(p_list)

    if np.linalg.norm(np.asarray(path[0]) - np.asarray(path[-1])) > 1e-12:
        path.append(list(path[0]))

    return {"order_global": order_global, "length": float(length), "path": path}


def route_lengths_for_owners(
    owners: np.ndarray, tiles: Sequence[Tile], graph: TileGraph
) -> np.ndarray:
    lengths = np.zeros(ROBOT_NUM, dtype=float)
    for rid in range(ROBOT_NUM):
        ids = [i for i, r in enumerate(owners) if int(r) == rid]
        lengths[rid] = float(closed_route_for_tiles(ids, tiles, graph, False)["length"])
    return lengths


# ============================================================================
# STEP 6 core: mainline-compatible strict shortcut postprocess
# ============================================================================
def polyline_length(path: Sequence[Sequence[float]]) -> float:
    """Euclidean length of a geometric polyline."""
    if len(path) <= 1:
        return 0.0
    return float(
        sum(
            np.linalg.norm(np.asarray(b, dtype=float) - np.asarray(a, dtype=float))
            for a, b in zip(path, path[1:])
        )
    )


def dedupe_polyline(path: Sequence[Sequence[float]]) -> List[List[float]]:
    out: List[List[float]] = []
    for point in path:
        p = np.asarray(point, dtype=float)
        if not out or not np.allclose(
            p, np.asarray(out[-1], dtype=float), atol=1e-12, rtol=0.0
        ):
            out.append(p.tolist())
    return out


def point_key(point: Sequence[float], digits: int = 10) -> Tuple[float, float]:
    p = np.asarray(point, dtype=float)
    return (round(float(p[0]), digits), round(float(p[1]), digits))


def segment_closed_cell_interval(
    a: Sequence[float],
    b: Sequence[float],
    cell_x: int,
    cell_y: int,
    tol: float = 1e-11,
    expand: float = 0.0,
) -> Optional[Tuple[float, float]]:
    """Clip a segment against one closed unit grid cell."""
    p = np.asarray(a, dtype=float)
    q = np.asarray(b, dtype=float)
    d = q - p
    lo = np.array((cell_x - expand, cell_y - expand), dtype=float)
    hi = np.array((cell_x + 1.0 + expand, cell_y + 1.0 + expand), dtype=float)
    t_enter, t_exit = 0.0, 1.0
    for axis in range(2):
        if abs(float(d[axis])) <= tol:
            if (
                float(p[axis]) < float(lo[axis]) - tol
                or float(p[axis]) > float(hi[axis]) + tol
            ):
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


def segment_intersects_closed_cell(
    a: Sequence[float], b: Sequence[float], cell_x: int, cell_y: int, tol: float = 1e-11
) -> bool:
    """Black obstacle cells are closed: even edge/corner contact is forbidden."""
    return (
        segment_closed_cell_interval(a, b, cell_x, cell_y, tol=tol, expand=tol)
        is not None
    )


def other_region_contact_is_corner_only(
    a: Sequence[float], b: Sequence[float], cell_x: int, cell_y: int, tol: float = 1e-10
) -> bool:
    """Other robot cells may be contacted only at one isolated grid corner."""
    interval = segment_closed_cell_interval(
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


def cell_owner_map(tiles: Sequence[Tile], owners: np.ndarray) -> Dict[Cell, int]:
    out: Dict[Cell, int] = {}
    for tid, tile in enumerate(tiles):
        rid = int(owners[tid])
        for c in tile_cells(tile):
            out[c] = rid
    return out


def segment_is_shortcut_legal(
    a: Sequence[float],
    b: Sequence[float],
    rid: int,
    cell_owners: Dict[Cell, int],
) -> bool:
    """Exact mainline rule for one newly exposed shortcut segment."""
    p = np.asarray(a, dtype=float)
    q = np.asarray(b, dtype=float)
    if p.shape != (2,) or q.shape != (2,):
        return False
    eps = 1e-10
    if (
        np.any(p < -eps)
        or np.any(q < -eps)
        or p[0] > MAP_SIZE + eps
        or q[0] > MAP_SIZE + eps
        or p[1] > MAP_SIZE + eps
        or q[1] > MAP_SIZE + eps
    ):
        return False

    min_x = max(0, int(math.floor(min(float(p[0]), float(q[0])) - 1e-9)))
    max_x = min(MAP_SIZE - 1, int(math.floor(max(float(p[0]), float(q[0])) + 1e-9)))
    min_y = max(0, int(math.floor(min(float(p[1]), float(q[1])) - 1e-9)))
    max_y = min(MAP_SIZE - 1, int(math.floor(max(float(p[1]), float(q[1])) + 1e-9)))

    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            if (x, y) in OBSTACLES:
                if segment_intersects_closed_cell(p, q, x, y):
                    return False
            elif cell_owners.get((x, y), -1) != rid:
                if not other_region_contact_is_corner_only(p, q, x, y):
                    return False
    return True


def legacy_portal_shortcut_path(
    raw_path: Sequence[Sequence[float]],
    rid: int,
    center_keys: Set[Tuple[float, float]],
    cell_owners: Dict[Cell, int],
) -> Tuple[List[List[float]], int, int, List[Tuple[List[float], List[float]]]]:
    """Mainline v2.2-compatible immutable center->portal->center deletion."""
    raw = dedupe_polyline(raw_path)
    remove_portals: Set[int] = set()
    segments: List[Tuple[List[float], List[float]]] = []
    tested = 0
    for i in range(1, len(raw) - 1):
        prev_key = point_key(raw[i - 1])
        mid_key = point_key(raw[i])
        next_key = point_key(raw[i + 1])
        if prev_key not in center_keys or next_key not in center_keys:
            continue
        if mid_key in center_keys:
            continue
        tested += 1
        a, c = raw[i - 1], raw[i + 1]
        if segment_is_shortcut_legal(a, c, rid, cell_owners):
            remove_portals.add(i)
            segments.append((list(a), list(c)))
    out = [p for i, p in enumerate(raw) if i not in remove_portals]
    return dedupe_polyline(out), tested, len(remove_portals), segments


def shortcut_redundant_route_points(
    path: Sequence[Sequence[float]],
    rid: int,
    center_keys: Set[Tuple[float, float]],
    cell_owners: Dict[Cell, int],
) -> Tuple[List[List[float]], int, int, List[Tuple[List[float], List[float]]]]:
    """Cascade legal deletions of portals and duplicate tile-center visits."""
    pts = dedupe_polyline(path)
    if len(pts) < 3:
        return pts, 0, 0, []

    center_count: Dict[Tuple[float, float], int] = {}
    for p in pts:
        key = point_key(p)
        if key in center_keys:
            center_count[key] = center_count.get(key, 0) + 1

    stack: List[List[float]] = []
    tested = 0
    accepted = 0
    shortcut_segments: List[Tuple[List[float], List[float]]] = []
    for point in pts:
        stack.append(list(map(float, point)))
        while len(stack) >= 3:
            a, mid, c = stack[-3], stack[-2], stack[-1]
            mid_key = point_key(mid)
            if mid_key in center_keys and center_count.get(mid_key, 0) <= 1:
                break
            old_len = (
                float(np.linalg.norm(np.asarray(mid) - np.asarray(a)))
                + float(np.linalg.norm(np.asarray(c) - np.asarray(mid)))
            )
            new_len = float(np.linalg.norm(np.asarray(c) - np.asarray(a)))
            if new_len > old_len + 1e-12:
                break
            tested += 1
            if not segment_is_shortcut_legal(a, c, rid, cell_owners):
                break
            stack.pop(-2)
            if mid_key in center_keys:
                center_count[mid_key] -= 1
            accepted += 1
            shortcut_segments.append((list(a), list(c)))

    return dedupe_polyline(stack), tested, accepted, shortcut_segments


def shortcut_route_mainline_style(
    rid: int,
    route: Dict[str, object],
    tiles: Sequence[Tile],
    owners: np.ndarray,
) -> Dict[str, object]:
    """Apply the mainline v2.3.2 two-stage strict shortcut to one STEP-5 route."""
    raw_path = dedupe_polyline(route.get("path", []))
    raw_length = polyline_length(raw_path)
    selected_ids = [i for i, r in enumerate(owners) if int(r) == rid]
    center_keys: Set[Tuple[float, float]] = {
        point_key(tile_center(tiles[tid])) for tid in selected_ids
    }
    owners_by_cell = cell_owner_map(tiles, owners)

    if len(raw_path) < 3:
        legacy_path = raw_path
        new_path = raw_path
        legacy_tested = legacy_accepted = extra_tested = extra_accepted = 0
        legacy_segments: List[Tuple[List[float], List[float]]] = []
        extra_segments: List[Tuple[List[float], List[float]]] = []
    else:
        legacy_path, legacy_tested, legacy_accepted, legacy_segments = (
            legacy_portal_shortcut_path(raw_path, rid, center_keys, owners_by_cell)
        )
        forward = shortcut_redundant_route_points(
            legacy_path, rid, center_keys, owners_by_cell
        )
        backward = shortcut_redundant_route_points(
            list(reversed(legacy_path)), rid, center_keys, owners_by_cell
        )
        backward_path = list(reversed(backward[0]))
        if polyline_length(backward_path) + 1e-12 < polyline_length(forward[0]):
            new_path = backward_path
            extra_tested, extra_accepted, extra_segments = backward[1], backward[2], backward[3]
        else:
            new_path = forward[0]
            extra_tested, extra_accepted, extra_segments = forward[1], forward[2], forward[3]

    new_path = dedupe_polyline(new_path)
    missing = center_keys - {point_key(p) for p in new_path}
    if missing:
        raise AssertionError(
            f"Robot {rid+1} shortcut removed {len(missing)} mandatory tile center(s)."
        )
    if len(new_path) >= 2 and not np.allclose(new_path[0], new_path[-1]):
        raise AssertionError(f"Robot {rid+1} shortcut route is not closed.")
    for a, b in zip(new_path, new_path[1:]):
        if not segment_is_shortcut_legal(a, b, rid, owners_by_cell):
            raise AssertionError(f"Robot {rid+1} final shortcut segment is illegal.")

    final_length = polyline_length(new_path)
    if final_length > raw_length + 1e-9:
        raise AssertionError(f"Robot {rid+1} shortcut increased path length.")

    return {
        "order_global": list(route["order_global"]),
        "path": new_path,
        "length": float(final_length),
        "raw_path": raw_path,
        "raw_length": float(raw_length),
        "legacy_path": legacy_path,
        "legacy_length": float(polyline_length(legacy_path)),
        "legacy_shortcuts_tested": int(legacy_tested),
        "legacy_shortcuts_accepted": int(legacy_accepted),
        "legacy_shortcut_segments": legacy_segments,
        "portal_shortcuts_tested": int(legacy_tested + extra_tested),
        "portal_shortcuts_accepted": int(legacy_accepted + extra_accepted),
        "shortcut_segments": legacy_segments + extra_segments,
        "shortcut_gain": float(raw_length - final_length),
        "shortcut_gain_pct": (
            100.0 * (raw_length - final_length) / raw_length
            if raw_length > 1e-12
            else 0.0
        ),
    }


# ============================================================================
# STEP 7 core: obstacle-safe curved closed-loop interpolation
# ============================================================================
def _closed_route_vertices(path: Sequence[Sequence[float]]) -> np.ndarray:
    """Return unique periodic vertices, dropping only the repeated closing point."""
    pts = dedupe_polyline(path)
    if len(pts) >= 2 and np.allclose(pts[0], pts[-1], atol=1e-12, rtol=0.0):
        pts = pts[:-1]
    return np.asarray(pts, dtype=float)


def _safe_vertex_tangent(
    points: np.ndarray, index: int, scale: float
) -> np.ndarray:
    """Bisector tangent for periodic cubic Hermite interpolation.

    The tangent magnitude is limited by the shorter adjacent segment.  A
    180-degree reversal receives a zero tangent because there is no unique
    smooth direction at that waypoint.
    """
    n = len(points)
    prev_p = points[(index - 1) % n]
    cur_p = points[index]
    next_p = points[(index + 1) % n]
    incoming = cur_p - prev_p
    outgoing = next_p - cur_p
    len_in = float(np.linalg.norm(incoming))
    len_out = float(np.linalg.norm(outgoing))
    if len_in <= 1e-12 or len_out <= 1e-12 or scale <= 0.0:
        return np.zeros(2, dtype=float)

    direction = incoming / len_in + outgoing / len_out
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-10:
        return np.zeros(2, dtype=float)
    direction /= norm
    magnitude = 0.5 * min(len_in, len_out) * float(scale)
    return direction * magnitude


def _sample_cubic_hermite(
    p0: np.ndarray,
    p1: np.ndarray,
    m0: np.ndarray,
    m1: np.ndarray,
    samples: int,
) -> np.ndarray:
    """Sample one endpoint-interpolating cubic Hermite segment."""
    t = np.linspace(0.0, 1.0, max(3, int(samples)), dtype=float)[:, None]
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1


def _curve_segment_is_legal(
    sampled: np.ndarray, rid: int, cell_owners: Dict[Cell, int]
) -> bool:
    """Audit every small chord used to render one smooth curve segment."""
    return all(
        segment_is_shortcut_legal(a, b, rid, cell_owners)
        for a, b in zip(sampled, sampled[1:])
    )


def curved_route_obstacle_safe(
    rid: int,
    route: Dict[str, object],
    tiles: Sequence[Tile],
    owners: np.ndarray,
) -> Dict[str, object]:
    """Round a STEP-6 closed polyline without changing its required waypoints.

    A periodic cubic Hermite curve passes through every STEP-6 waypoint.  The
    tangent at each waypoint is shared by its neighboring curve pieces.  If a
    rounded piece would touch an obstacle or illegally enter another robot
    region, only the involved vertex tangents are halved until the sampled
    curve is safe.

    ``render_segments`` deliberately preserves the Hermite piece boundaries.
    STEP 7 uses those boundaries to assign different z-orders at self crossings.
    """
    vertices = _closed_route_vertices(route.get("path", []))
    if len(vertices) < 3:
        fallback = dedupe_polyline(route.get("path", []))
        fallback_arr = np.asarray(fallback, dtype=float)
        return {
            "path": fallback,
            "length": float(polyline_length(fallback)),
            "waypoint_path": fallback,
            "render_segments": [fallback_arr.tolist()] if len(fallback_arr) >= 2 else [],
            "vertex_tangent_scales": [],
            "safety_passes": 0,
            "closed": bool(len(fallback) < 2 or np.allclose(fallback[0], fallback[-1])),
        }

    n = len(vertices)
    scales = np.full(n, CURVE_INITIAL_TANGENT_SCALE, dtype=float)
    owners_by_cell = cell_owner_map(tiles, owners)
    sampled_segments: List[np.ndarray] = []
    safety_passes = 0

    for safety_passes in range(CURVE_MAX_SAFETY_PASSES + 1):
        tangents = [
            _safe_vertex_tangent(vertices, i, float(scales[i])) for i in range(n)
        ]
        sampled_segments = []
        bad_vertices: Set[int] = set()
        for i in range(n):
            j = (i + 1) % n
            seg = _sample_cubic_hermite(
                vertices[i], vertices[j], tangents[i], tangents[j],
                CURVE_SAMPLES_PER_SEGMENT,
            )
            sampled_segments.append(seg)
            if not _curve_segment_is_legal(seg, rid, owners_by_cell):
                bad_vertices.add(i)
                bad_vertices.add(j)

        if not bad_vertices:
            break
        for idx in bad_vertices:
            scales[idx] *= 0.5
            if scales[idx] < 1e-5:
                scales[idx] = 0.0
    else:
        raise AssertionError(f"Robot {rid+1} curve safety iteration did not converge.")

    # Defensive fallback: zero tangents reproduce STEP-6 straight segments as
    # densely sampled cubic easing pieces.
    if any(
        not _curve_segment_is_legal(seg, rid, owners_by_cell)
        for seg in sampled_segments
    ):
        scales[:] = 0.0
        tangents = [np.zeros(2, dtype=float) for _ in range(n)]
        sampled_segments = [
            _sample_cubic_hermite(
                vertices[i], vertices[(i + 1) % n], tangents[i],
                tangents[(i + 1) % n], CURVE_SAMPLES_PER_SEGMENT,
            )
            for i in range(n)
        ]
        if any(
            not _curve_segment_is_legal(seg, rid, owners_by_cell)
            for seg in sampled_segments
        ):
            raise AssertionError(f"Robot {rid+1} curved fallback is illegal.")

    curve: List[List[float]] = []
    for i, seg in enumerate(sampled_segments):
        start = 0 if i == 0 else 1
        curve.extend(seg[start:].tolist())
    if curve and not np.allclose(curve[0], curve[-1], atol=1e-12, rtol=0.0):
        curve.append(list(curve[0]))

    waypoint_keys = {point_key(p) for p in vertices}
    curve_keys = {point_key(p) for p in curve}
    if not waypoint_keys.issubset(curve_keys):
        raise AssertionError(f"Robot {rid+1} curved path lost a STEP-6 waypoint.")
    if len(curve) >= 2 and not np.allclose(curve[0], curve[-1]):
        raise AssertionError(f"Robot {rid+1} curved path is not closed.")
    for a, b in zip(curve, curve[1:]):
        if not segment_is_shortcut_legal(a, b, rid, owners_by_cell):
            raise AssertionError(f"Robot {rid+1} curved path contains an illegal chord.")

    return {
        "path": curve,
        "length": float(polyline_length(curve)),
        "waypoint_path": [p.tolist() for p in vertices] + [vertices[0].tolist()],
        "render_segments": [seg.tolist() for seg in sampled_segments],
        "vertex_tangent_scales": [float(x) for x in scales],
        "safety_passes": int(safety_passes),
        "closed": True,
    }


# ============================================================================
# STEP 4 core: connectivity-preserving tile transfer accepted by closed-tour CV
# ============================================================================
def optimize_partition_by_closed_tour_cv(
    initial_owners: np.ndarray,
    tiles: Sequence[Tile],
    graph: TileGraph,
) -> Tuple[np.ndarray, List[Dict[str, object]], np.ndarray, np.ndarray]:
    owners = initial_owners.copy()
    initial_lengths = route_lengths_for_owners(owners, tiles, graph)
    lengths = initial_lengths.copy()
    current_cv = coefficient_of_variation(lengths)
    trace: List[Dict[str, object]] = []

    for iteration in range(1, MAX_TRANSFER_ITERATIONS + 1):
        best = None

        # Only boundary tiles can move to another robot while receiver connectivity
        # is preserved by construction.
        for tid, donor_value in enumerate(owners):
            donor = int(donor_value)
            receivers = sorted(
                {
                    int(owners[v])
                    for v in graph.adjacency[tid]
                    if int(owners[v]) != donor
                }
            )
            for receiver in receivers:
                if not can_transfer(graph, owners, tid, receiver):
                    continue

                trial = owners.copy()
                trial[tid] = receiver
                trial_lengths = lengths.copy()
                for rid in (donor, receiver):
                    ids = [i for i, r in enumerate(trial) if int(r) == rid]
                    trial_lengths[rid] = float(
                        closed_route_for_tiles(ids, tiles, graph, False)["length"]
                    )
                trial_cv = coefficient_of_variation(trial_lengths)
                key = (
                    trial_cv,
                    float(trial_lengths.max(initial=0.0)),
                    float(trial_lengths.sum()),
                    tid,
                    receiver,
                )
                if best is None or key < best[0]:
                    best = (key, tid, donor, receiver, trial, trial_lengths)

        if best is None or best[0][0] >= current_cv - 1e-9:
            break

        key, tid, donor, receiver, owners, lengths = best
        current_cv = float(key[0])
        trace.append(
            {
                "iteration": iteration,
                "tile_id": int(tid),
                "tile": list(tiles[tid]),
                "donor_robot": donor + 1,
                "receiver_robot": receiver + 1,
                "path_lengths": [float(x) for x in lengths],
                "path_cv": current_cv,
            }
        )

    return owners, trace, initial_lengths, lengths


# ============================================================================
# Visualization — every call writes ONE square image
# ============================================================================
def new_square_figure():
    # Exact square canvas: no tight-bbox cropping, no multi-panel composition.
    fig = plt.figure(figsize=(8, 8), dpi=300, facecolor="white")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, MAP_SIZE)
    ax.set_ylim(0, MAP_SIZE)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    return fig, ax


def draw_base_grid(ax, owners: Optional[np.ndarray] = None, tiles: Optional[Sequence[Tile]] = None):
    cell_to_owner: Dict[Cell, int] = {}
    if owners is not None and tiles is not None:
        for tid, t in enumerate(tiles):
            rid = int(owners[tid])
            for c in tile_cells(t):
                cell_to_owner[c] = rid

    for x in range(MAP_SIZE):
        for y in range(MAP_SIZE):
            if (x, y) in OBSTACLES:
                face = OBSTACLE_COLOR
                alpha = 1.0
                edge = "black"
            elif (x, y) in cell_to_owner:
                face = ROBOT_COLORS[cell_to_owner[(x, y)]]
                alpha = 0.35
                edge = GRID_COLOR
            else:
                face = FREE_COLOR
                alpha = 1.0
                edge = GRID_COLOR
            ax.add_patch(
                Rectangle(
                    (x, y), 1, 1,
                    facecolor=face,
                    alpha=alpha,
                    edgecolor=edge,
                    linewidth=1.15,
                    zorder=1,
                )
            )


def draw_tile_boundaries(ax, tiles: Sequence[Tile], owners: Optional[np.ndarray] = None, linewidth: float = 3.2):
    for tid, (x, y, w, h) in enumerate(tiles):
        if owners is None:
            face = NEUTRAL_TILE_COLOR
            alpha = 0.36
        else:
            face = ROBOT_COLORS[int(owners[tid])]
            alpha = 0.20
        ax.add_patch(
            Rectangle(
                (x, y), w, h,
                facecolor=face,
                alpha=alpha,
                edgecolor="black",
                linewidth=linewidth,
                zorder=3,
            )
        )


def stage_badge(ax, title: str, subtitle: str = ""):
    # text = title if not subtitle else f"{title}\n{subtitle}"
    # ax.text(
    #     0.18, MAP_SIZE - 0.18, text,
    #     ha="left", va="top", color=TEXT_COLOR,
    #     fontsize=12.5, fontweight="bold", linespacing=1.35,
    #     bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="black", alpha=0.92),
    #     zorder=20,
    # )
    return


def save_square(fig, filename: str) -> Path:
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return path


def plot_step_01() -> Path:
    fig, ax = new_square_figure()
    draw_base_grid(ax)
    stage_badge(ax, "STEP 1  Input map", f"8x8 grid | 3 robots | {len(OBSTACLES)} obstacles")
    return save_square(fig, "step_01_input_map.png")


def plot_step_02(tiles: Sequence[Tile]) -> Path:
    fig, ax = new_square_figure()
    draw_base_grid(ax)

    # STEP 2 only: keep the neutral tile fill, then add a stronger dark-yellow
    # skeleton on top.  Drawing the outline separately keeps it fully opaque
    # instead of inheriting the semi-transparent tile fill alpha.
    draw_tile_boundaries(ax, tiles, owners=None, linewidth=0.0)
    for x, y, w, h in tiles:
        ax.add_patch(
            Rectangle(
                (x, y), w, h,
                facecolor="none",
                edgecolor="#B8860B",  # dark yellow / dark goldenrod
                linewidth=10,
                zorder=5,
            )
        )

    centers = np.asarray([tile_center(t) for t in tiles])
    ax.scatter(centers[:, 0], centers[:, 1], s=28, facecolors="white", edgecolors="black", linewidths=1.1, zorder=8)
    stage_badge(ax, "STEP 2  Global Tile-first", f"exact cover | {len(tiles)} rectangles")
    return save_square(fig, "step_02_global_tiling.png")


def plot_partition(
    filename: str,
    title: str,
    subtitle: str,
    tiles: Sequence[Tile],
    owners: np.ndarray,
    seeds: Optional[Sequence[int]] = None,
    moved_tile_ids: Optional[Sequence[int]] = None,
) -> Path:
    fig, ax = new_square_figure()
    draw_base_grid(ax, owners, tiles)
    draw_tile_boundaries(ax, tiles, owners, linewidth=2.8)

    if seeds is not None:
        for rid, tid in enumerate(seeds):
            c = tile_center(tiles[tid])
            ax.scatter(
                [c[0]], [c[1]], s=180, marker="o",
                facecolor="white", edgecolor=ROBOT_COLORS[rid], linewidth=3.4, zorder=12,
            )
            ax.text(c[0], c[1], f"R{rid+1}", ha="center", va="center", fontsize=8.5, fontweight="bold", zorder=13)

    if moved_tile_ids:
        for tid in moved_tile_ids:
            x, y, w, h = tiles[tid]
            ax.add_patch(
                Rectangle((x + 0.05, y + 0.05), w - 0.10, h - 0.10,
                          fill=False, edgecolor="white", linewidth=2.8,
                          linestyle=(0, (3, 2)), zorder=14)
            )

    stage_badge(ax, title, subtitle)
    return save_square(fig, filename)


def draw_routes(ax, routes: Dict[int, Dict[str, object]], only_robot: Optional[int] = None):
    for rid in range(ROBOT_NUM):
        if only_robot is not None and rid != only_robot:
            continue
        path = np.asarray(routes[rid]["path"], dtype=float)
        if len(path) < 2:
            continue
        ax.plot(
            path[:, 0], path[:, 1],
            color=ROBOT_COLORS[rid], linewidth=6.0,
            solid_capstyle="round", solid_joinstyle="round", zorder=15,
        )
        # Tile-center visits as white beads, matching the reference visual language.
        order_global = routes[rid]["order_global"]
        centers = np.asarray([FINAL_GRAPH.centers[gid] for gid in order_global])
        if len(centers):
            ax.scatter(
                centers[:, 0], centers[:, 1],
                s=45, facecolors="white", edgecolors=ROBOT_COLORS[rid], linewidths=1.6, zorder=17,
            )
            start = centers[0]
            ax.scatter([start[0]], [start[1]], s=190, facecolor="white", edgecolor="black", linewidth=2.0, zorder=18)
            ax.text(start[0], start[1], f"R{rid+1}", ha="center", va="center", fontsize=8.5, fontweight="bold", zorder=19)


def plot_step_05(tiles: Sequence[Tile], owners: np.ndarray, routes: Dict[int, Dict[str, object]]) -> Path:
    fig, ax = new_square_figure()
    draw_base_grid(ax, owners, tiles)
    draw_tile_boundaries(ax, tiles, owners, linewidth=2.2)
    draw_routes(ax, routes)
    lengths = [float(routes[r]["length"]) for r in range(ROBOT_NUM)]
    stage_badge(ax, "STEP 5  Final closed cycles", " / ".join(f"R{i+1}={v:.2f}" for i, v in enumerate(lengths)))
    return save_square(fig, "step_05_final_closed_cycles.png")


def plot_step_06(
    tiles: Sequence[Tile],
    owners: np.ndarray,
    raw_routes: Dict[int, Dict[str, object]],
    final_routes: Dict[int, Dict[str, object]],
) -> Path:
    fig, ax = new_square_figure()
    draw_base_grid(ax, owners, tiles)
    draw_tile_boundaries(ax, tiles, owners, linewidth=2.2)

    # Thin dashed STEP-5 geometry shows what was removed; the thick colored line
    # is the actual post-shortcut final path.
    for rid in range(ROBOT_NUM):
        raw = np.asarray(raw_routes[rid]["path"], dtype=float)
        if len(raw) >= 2:
            ax.plot(
                raw[:, 0], raw[:, 1],
                color="#777777", linewidth=1.8, linestyle=(0, (4, 4)),
                alpha=0.55, zorder=11,
            )
    draw_routes(ax, final_routes)
    lengths = [float(final_routes[r]["length"]) for r in range(ROBOT_NUM)]
    gains = [float(final_routes[r]["shortcut_gain"]) for r in range(ROBOT_NUM)]
    stage_badge(
        ax,
        "STEP 6  Strict shortcut final paths",
        " / ".join(
            f"R{i+1}={lengths[i]:.2f} (-{gains[i]:.2f})" for i in range(ROBOT_NUM)
        ),
    )
    return save_square(fig, "step_06_shortcut_final_paths.png")



def _darken_hex(color: str, factor: float = CURVED_ROUTE_OUTLINE_DARKEN) -> Tuple[float, float, float]:
    """Return a darker RGB tuple for the continuous outer casing."""
    color = color.lstrip("#")
    if len(color) != 6:
        return (0.15, 0.15, 0.15)
    rgb = tuple(int(color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return tuple(max(0.0, min(1.0, c * factor)) for c in rgb)


def _route_render_segments(route: Dict[str, object]) -> List[np.ndarray]:
    """Return ordered Hermite pieces for explicit crossing z-layers."""
    stored = route.get("render_segments", [])
    segments = [np.asarray(seg, dtype=float) for seg in stored if len(seg) >= 2]
    if segments:
        return segments
    path = np.asarray(route.get("path", []), dtype=float)
    return [path] if len(path) >= 2 else []


def draw_curved_routes(
    ax, curved_routes: Dict[int, Dict[str, object]], only_robot: Optional[int] = None
) -> None:
    for rid in range(ROBOT_NUM):
        if only_robot is not None and rid != only_robot:
            continue

        path = np.asarray(curved_routes[rid]["path"], dtype=float)
        if len(path) < 2:
            continue

        outline_color = _darken_hex(CURVED_ROUTE_COLORS[rid])
        segments = _route_render_segments(curved_routes[rid])
        robot_z_base = 20.0 + 10.0 * rid

        ax.plot(
            path[:, 0], path[:, 1],
            color=outline_color,
            linewidth=CURVED_ROUTE_OUTLINE_LINEWIDTH,
            linestyle="-",
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=robot_z_base,
        )
        ax.plot(
            path[:, 0], path[:, 1],
            color=CURVED_ROUTE_COLORS[rid],
            linewidth=CURVED_ROUTE_LINEWIDTH,
            linestyle="-",
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=robot_z_base + CURVED_ROUTE_Z_STEP,
        )

        segment_z0 = robot_z_base + 2.0 * CURVED_ROUTE_Z_STEP
        for seg_index, seg in enumerate(segments):
            if len(seg) < 2:
                continue
            outer_z = segment_z0 + (2 * seg_index) * CURVED_ROUTE_Z_STEP
            inner_z = outer_z + CURVED_ROUTE_Z_STEP

            ax.plot(
                seg[:, 0], seg[:, 1],
                color=outline_color,
                linewidth=CURVED_ROUTE_OUTLINE_LINEWIDTH,
                linestyle="-",
                solid_capstyle="butt",
                solid_joinstyle="round",
                zorder=outer_z,
            )
            ax.plot(
                seg[:, 0], seg[:, 1],
                color=CURVED_ROUTE_COLORS[rid],
                linewidth=CURVED_ROUTE_LINEWIDTH,
                linestyle="-",
                solid_capstyle="butt",
                solid_joinstyle="round",
                zorder=inner_z,
            )

        start = path[0]
        ax.scatter(
            [start[0]], [start[1]],
            s=34,
            facecolor="white",
            edgecolor=CURVED_ROUTE_COLORS[rid],
            linewidth=1.5,
            zorder=100,
        )


def plot_step_07(
    tiles: Sequence[Tile],
    owners: np.ndarray,
    shortcut_routes: Dict[int, Dict[str, object]],
    curved_routes: Dict[int, Dict[str, object]],
) -> Path:
    fig, ax = new_square_figure()
    draw_base_grid(ax, owners, tiles)
    draw_tile_boundaries(ax, tiles, owners, linewidth=2.2)
    draw_curved_routes(ax, curved_routes)

    lengths = [float(curved_routes[r]["length"]) for r in range(ROBOT_NUM)]
    reduced = any(
        min(curved_routes[r]["vertex_tangent_scales"], default=CURVE_INITIAL_TANGENT_SCALE)
        < CURVE_INITIAL_TANGENT_SCALE - 1e-9
        for r in range(ROBOT_NUM)
    )
    stage_badge(
        ax,
        "STEP 7  Curved final closed loops",
        " / ".join(f"R{i+1}={lengths[i]:.2f}" for i in range(ROBOT_NUM))
        + (" | adaptive safe rounding | z-layer crossings" if reduced
           else " | full rounding | z-layer crossings"),
    )
    return save_square(fig, "step_07_curved_final_closed_loops.png")



def plot_single_robot(
    rid: int,
    tiles: Sequence[Tile],
    owners: np.ndarray,
    routes: Dict[int, Dict[str, object]],
) -> Path:
    fig, ax = new_square_figure()

    # Obstacles stay black.  The selected robot keeps the reference color; all
    # other free cells fade to near-white so the cycle is immediately readable.
    for x in range(MAP_SIZE):
        for y in range(MAP_SIZE):
            if (x, y) in OBSTACLES:
                face, alpha = OBSTACLE_COLOR, 1.0
            else:
                face, alpha = "#F7F7F7", 1.0
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor=face, alpha=alpha,
                                   edgecolor=GRID_COLOR, linewidth=1.0, zorder=1))

    selected_ids = [i for i, r in enumerate(owners) if int(r) == rid]
    for tid in selected_ids:
        x, y, w, h = tiles[tid]
        ax.add_patch(Rectangle((x, y), w, h, facecolor=ROBOT_COLORS[rid], alpha=0.20,
                               edgecolor="black", linewidth=2.8, zorder=3))

    draw_routes(ax, routes, only_robot=rid)
    stage_badge(ax, f"Robot {rid+1} closed cycle", f"length = {float(routes[rid]['length']):.2f}")
    return save_square(fig, f"robot_{rid+1:02d}_closed_cycle.png")


# Global used only by draw_routes for concise plotting code; set inside main().
FINAL_GRAPH: TileGraph


# ============================================================================
# Seven-step main program
# ============================================================================
def main() -> None:
    global FINAL_GRAPH, OBSTACLES

    # IMPORTANT: this is where RANDOM_SEED actually controls obstacle positions.
    OBSTACLES = generate_random_obstacles(MAP_SIZE, OBSTACLE_NUM, RANDOM_SEED)
    validate_configuration(OBSTACLES)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Tile-First MCPP — 7-step visual demo")
    print(
        f"Map: {MAP_SIZE}x{MAP_SIZE} | Robots: {ROBOT_NUM} | "
        f"Obstacles: {len(OBSTACLES)} | Seed: {RANDOM_SEED}"
    )
    print(f"Obstacle cells: {sorted(OBSTACLES)}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 72)

    # ------------------------------------------------------------------ STEP 1
    p1 = plot_step_01()
    print(f"[1/7] Input map                         -> {p1.name}")

    # ------------------------------------------------------------------ STEP 2
    tiles = global_tile_first(OBSTACLES)
    graph = build_tile_graph(tiles, OBSTACLES)
    FINAL_GRAPH = graph
    if len(tiles) < ROBOT_NUM:
        raise RuntimeError("Too few global tiles for three nonempty robot regions.")
    p2 = plot_step_02(tiles)
    print(f"[2/7] Global tile-first exact cover     -> {p2.name} ({len(tiles)} tiles)")

    # ------------------------------------------------------------------ STEP 3
    seeds = choose_partition_seeds(tiles, graph)
    initial_owners = connected_balanced_growth(tiles, graph, seeds)
    for rid in range(ROBOT_NUM):
        if not owner_connected(graph, initial_owners, rid):
            raise AssertionError(f"Initial region R{rid+1} is disconnected.")
    initial_lengths = route_lengths_for_owners(initial_owners, tiles, graph)
    initial_cv = coefficient_of_variation(initial_lengths)
    p3 = plot_partition(
        "step_03_connected_initial_partition.png",
        "STEP 3  Connected initialization",
        f"closed-tour CV = {initial_cv:.3f}",
        tiles, initial_owners, seeds=seeds,
    )
    print(f"[3/7] Connected balanced partition      -> {p3.name} (CV={initial_cv:.4f})")

    # ------------------------------------------------------------------ STEP 4
    final_owners, trace, cv_start_lengths, final_lengths = optimize_partition_by_closed_tour_cv(
        initial_owners, tiles, graph
    )
    final_cv = coefficient_of_variation(final_lengths)
    moved_ids = [int(row["tile_id"]) for row in trace]
    for rid in range(ROBOT_NUM):
        if not owner_connected(graph, final_owners, rid):
            raise AssertionError(f"Final region R{rid+1} is disconnected.")
    p4 = plot_partition(
        "step_04_cv_balanced_partition.png",
        "STEP 4  CV tile-transfer balance",
        f"CV {initial_cv:.3f} -> {final_cv:.3f} | accepted moves = {len(trace)}",
        tiles, final_owners, seeds=None, moved_tile_ids=moved_ids,
    )
    print(f"[4/7] Connectivity-preserving transfers -> {p4.name} (CV={final_cv:.4f}, moves={len(trace)})")

    # ------------------------------------------------------------------ STEP 5
    routes: Dict[int, Dict[str, object]] = {}
    for rid in range(ROBOT_NUM):
        ids = [i for i, r in enumerate(final_owners) if int(r) == rid]
        routes[rid] = closed_route_for_tiles(ids, tiles, graph, expand_geometry=True)
        path = np.asarray(routes[rid]["path"], dtype=float)
        if len(path) < 2 or not np.allclose(path[0], path[-1]):
            raise AssertionError(f"Robot {rid+1} route is not explicitly closed.")

    p5 = plot_step_05(tiles, final_owners, routes)
    print(f"[5/7] Obstacle-safe closed patrol cycles -> {p5.name}")

    robot_paths = [plot_single_robot(rid, tiles, final_owners, routes) for rid in range(ROBOT_NUM)]

    # ------------------------------------------------------------------ STEP 6
    shortcut_routes: Dict[int, Dict[str, object]] = {
        rid: shortcut_route_mainline_style(rid, routes[rid], tiles, final_owners)
        for rid in range(ROBOT_NUM)
    }
    p6 = plot_step_06(tiles, final_owners, routes, shortcut_routes)
    print(f"[6/7] Strict shortcut final patrol paths -> {p6.name}")

    # ------------------------------------------------------------------ STEP 7
    curved_routes: Dict[int, Dict[str, object]] = {
        rid: curved_route_obstacle_safe(rid, shortcut_routes[rid], tiles, final_owners)
        for rid in range(ROBOT_NUM)
    }
    p7 = plot_step_07(tiles, final_owners, shortcut_routes, curved_routes)
    print(f"[7/7] Curved final closed loops         -> {p7.name}")

    # ------------------------------------------------------------------ data export
    route_data = {
        "map_size": MAP_SIZE,
        "robot_num": ROBOT_NUM,
        "random_seed": RANDOM_SEED,
        "obstacle_num": OBSTACLE_NUM,
        "obstacles": [list(c) for c in sorted(OBSTACLES)],
        "tiles": [list(t) for t in tiles],
        "initial_tile_owners_1based": [int(x) + 1 for x in initial_owners],
        "final_tile_owners_1based": [int(x) + 1 for x in final_owners],
        "initial_path_lengths": [float(x) for x in initial_lengths],
        "initial_path_cv": float(initial_cv),
        "final_path_lengths": [float(routes[r]["length"]) for r in range(ROBOT_NUM)],
        "final_path_cv": float(coefficient_of_variation([routes[r]["length"] for r in range(ROBOT_NUM)])),
        "accepted_transfers": trace,
        "routes": {
            f"robot_{rid+1}": {
                "closed": True,
                "tile_visit_order_global_ids": [int(x) for x in routes[rid]["order_global"]],
                "length": float(routes[rid]["length"]),
                "path_xy": routes[rid]["path"],
            }
            for rid in range(ROBOT_NUM)
        },
        "shortcut_final_path_lengths": [float(shortcut_routes[r]["length"]) for r in range(ROBOT_NUM)],
        "shortcut_final_path_cv": float(coefficient_of_variation([shortcut_routes[r]["length"] for r in range(ROBOT_NUM)])),
        "shortcut_routes": {
            f"robot_{rid+1}": {
                "closed": True,
                "tile_visit_order_global_ids": [int(x) for x in routes[rid]["order_global"]],
                "raw_length": float(routes[rid]["length"]),
                "raw_path_xy": routes[rid]["path"],
                "length": float(shortcut_routes[rid]["length"]),
                "path_xy": shortcut_routes[rid]["path"],
                "legacy_shortcut_length": float(shortcut_routes[rid]["legacy_length"]),
                "shortcut_gain": float(shortcut_routes[rid]["shortcut_gain"]),
                "shortcut_gain_pct": float(shortcut_routes[rid]["shortcut_gain_pct"]),
                "shortcuts_tested": int(shortcut_routes[rid]["portal_shortcuts_tested"]),
                "shortcuts_accepted": int(shortcut_routes[rid]["portal_shortcuts_accepted"]),
                "shortcut_segments": shortcut_routes[rid]["shortcut_segments"],
            }
            for rid in range(ROBOT_NUM)
        },
        "curved_final_path_lengths": [float(curved_routes[r]["length"]) for r in range(ROBOT_NUM)],
        "curved_routes": {
            f"robot_{rid+1}": {
                "closed": bool(curved_routes[rid]["closed"]),
                "source_shortcut_length": float(shortcut_routes[rid]["length"]),
                "length": float(curved_routes[rid]["length"]),
                "path_xy": curved_routes[rid]["path"],
                "waypoint_path_xy": curved_routes[rid]["waypoint_path"],
                "render_segments_xy": curved_routes[rid]["render_segments"],
                "render_segment_z_order": [
                    int(i) for i in range(len(curved_routes[rid]["render_segments"]))
                ],
                "vertex_tangent_scales": curved_routes[rid]["vertex_tangent_scales"],
                "safety_passes": int(curved_routes[rid]["safety_passes"]),
            }
            for rid in range(ROBOT_NUM)
        },
    }
    with (OUTPUT_DIR / "route_data.json").open("w", encoding="utf-8") as f:
        json.dump(route_data, f, ensure_ascii=False, indent=2)

    with (OUTPUT_DIR / "route_summary.txt").open("w", encoding="utf-8") as f:
        f.write("Tile-First MCPP — 7-step demo\n")
        f.write("=" * 60 + "\n")
        f.write(
            f"Map: {MAP_SIZE}x{MAP_SIZE}\nRobots: {ROBOT_NUM}\n"
            f"Random seed: {RANDOM_SEED}\nObstacles: {len(OBSTACLES)}\n"
            f"Obstacle cells: {sorted(OBSTACLES)}\nGlobal tiles: {len(tiles)}\n"
        )
        f.write(f"Initial closed-tour CV: {initial_cv:.6f}\n")
        f.write(f"Final closed-tour CV:   {final_cv:.6f}\n")
        f.write(f"Accepted tile transfers: {len(trace)}\n\n")
        for rid in range(ROBOT_NUM):
            f.write(f"Robot {rid+1}\n")
            f.write(f"  closed route length: {float(routes[rid]['length']):.6f}\n")
            f.write(f"  path xy (first point == last point):\n")
            for p in routes[rid]["path"]:
                f.write(f"    ({p[0]:.3f}, {p[1]:.3f})\n")
            f.write("\n")
        f.write("STEP 6 shortcut final paths\n")
        f.write("-" * 60 + "\n")
        for rid in range(ROBOT_NUM):
            f.write(
                f"Robot {rid+1}: {float(routes[rid]['length']):.6f} -> "
                f"{float(shortcut_routes[rid]['length']):.6f} "
                f"(gain={float(shortcut_routes[rid]['shortcut_gain']):.6f}, "
                f"accepted/tested={int(shortcut_routes[rid]['portal_shortcuts_accepted'])}/"
                f"{int(shortcut_routes[rid]['portal_shortcuts_tested'])})\n"
            )
            for p in shortcut_routes[rid]["path"]:
                f.write(f"    ({p[0]:.3f}, {p[1]:.3f})\n")
            f.write("\n")

        f.write("STEP 7 curved final closed loops with z-layer crossings\n")
        f.write("-" * 60 + "\n")
        for rid in range(ROBOT_NUM):
            f.write(
                f"Robot {rid+1}: shortcut={float(shortcut_routes[rid]['length']):.6f} -> "
                f"curved={float(curved_routes[rid]['length']):.6f}, "
                f"pieces={len(curved_routes[rid]['render_segments'])}, "
                f"safety_passes={int(curved_routes[rid]['safety_passes'])}\n"
            )
            f.write(
                "  crossing z-order: later traversal piece is drawn above earlier piece\n"
            )
            f.write(f"  tangent scales: {curved_routes[rid]['vertex_tangent_scales']}\n")
            f.write(f"  sampled points: {len(curved_routes[rid]['path'])}\n\n")

    print("\nFinal closed routes:")
    for rid in range(ROBOT_NUM):
        path = routes[rid]["path"]
        print(
            f"  Robot {rid+1}: length={float(routes[rid]['length']):.3f}, "
            f"points={len(path)}, closed={np.allclose(path[0], path[-1])}"
        )
    print("\nSTEP 6 shortcut final routes:")
    for rid in range(ROBOT_NUM):
        path = shortcut_routes[rid]["path"]
        print(
            f"  Robot {rid+1}: raw={float(routes[rid]['length']):.3f} -> "
            f"final={float(shortcut_routes[rid]['length']):.3f}, "
            f"gain={float(shortcut_routes[rid]['shortcut_gain']):.3f}, "
            f"points={len(path)}, closed={np.allclose(path[0], path[-1])}"
        )
    print("\nSTEP 7 curved final routes:")
    for rid in range(ROBOT_NUM):
        path = curved_routes[rid]["path"]
        print(
            f"  Robot {rid+1}: shortcut={float(shortcut_routes[rid]['length']):.3f} -> "
            f"curved={float(curved_routes[rid]['length']):.3f}, "
            f"pieces={len(curved_routes[rid]['render_segments'])}, "
            f"points={len(path)}, closed={np.allclose(path[0], path[-1])}"
        )

    print(f"\nSeparate robot figures: {', '.join(p.name for p in robot_paths)}")
    print("route_data.json and route_summary.txt also written.")


if __name__ == "__main__":
    main()
