import random
import numpy as np
import math
import os
import time
from datetime import datetime
import concurrent.futures

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import coo_matrix
    HAS_MILP = True
except Exception:
    HAS_MILP = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except Exception:
    HAS_NETWORKX = False

# ============================================================
# 两种铺砖方案：A=SCHEME_FULL，B=SCHEME_SIMPLE
# ============================================================
SCHEME_FULL = {
    2: (2, 2, '#FF6B6B', '2x2'),
    3: (2, 1, '#4ECDC4', '2x1'),
    4: (1, 2, '#45B7D1', '1x2'),
    5: (1, 1, '#96CEB4', '1x1'),
    6: (2, 4, '#FFD93D', '2x4'),
    7: (4, 2, '#FF6F91', '4x2'),
    8: (4, 4, '#9C9E09', '4x4')
}

SCHEME_SIMPLE = {
    2: (2, 2, '#FF6B6B', '2x2'),
    5: (1, 1, '#96CEB4', '1x1'),
    8: (4, 4, '#9C9E09', '4x4')
}

# 定义您提供的50个随机种子
EXPERIMENT_SEEDS = [
    42, 100, 256, 512, 1024, 2048, 4096, 8192, 12345, 99999,
    7, 13, 29, 63, 127, 255, 511, 777, 1337, 2024,
    3141, 4097, 5003, 6666, 7001, 8191, 9001, 10007, 12011, 15013,
    17021, 19031, 21001, 23003, 25013, 27011, 29009, 31013, 33023, 35023,
    37019, 39041, 41047, 43051, 45053, 47057, 49069, 51071, 53087, 55073
]

# ============================================================
# 新算法参数
# ============================================================
# 铺砖：回退前=0-1整数规划MILP；回退后=4方向贪心。
TILE_OPTIMIZER = "milp"
ILP_TIME_LIMIT = 1.0
# 大图MILP候选太多会导致内存/时间不可控；超过阈值时直接标记为“回退后”并使用贪心。
# 如需强制尝试MILP，可调大下面两个阈值。
TILE_MILP_MAX_CELLS = 2500
TILE_MILP_MAX_CANDIDATES = 200000

# TSP：严格按用户提供算法：
# - 若砖块中心数 <= 80：尝试 连续避障距离矩阵 + MTZ精确TSP；
# - 若砖块中心数 > 80：回退到 Christofides 近似TSP；
# - 若60秒内精确求解失败或超时：回退到 Christofides 近似TSP。
# 注意：Christofides 仍使用连续避障距离矩阵，不再加入 fast_sweep 或 fast_nearest_neighbor。
TSP_SOLVER = "exact_mtz"
TSP_TIME_LIMIT = 60.0
TSP_EXACT_MAX_POINTS = 80
VISIBILITY_EPS = 1e-9
ALLOW_OBSTACLE_BOUNDARY_SLIDING = True

BEFORE_FALLBACK = "before_fallback"
AFTER_FALLBACK = "after_fallback"


class GridFiller:
    """
    实验流程保持原脚本形式：同样通过 GridFiller 生成指定障碍物下的铺砖结果。

    算法替换为：
    1) 铺砖：优先使用0-1整数规划求最少砖数，失败/超阈值/无MILP时回退到四方向贪心；
    2) 路径：砖块中心数<=80时优先使用连续避障距离矩阵 + MTZ精确TSP；点数>80或精确求解失败/超时时回退到Christofides。

    标记规则：
    - self.tile_stage == before_fallback：铺砖使用MILP结果；
    - self.tile_stage == after_fallback ：铺砖使用贪心回退结果；
    - TSP返回 tsp_stage == before_fallback：TSP使用精确MTZ结果；
    - TSP返回 tsp_stage == after_fallback ：TSP使用Christofides回退结果。
    """
    def __init__(self, grid_size=10, obstacle_num=None, block_types=None, obstacle_positions=None, verbose=True):
        self.grid_size = grid_size
        self.verbose = verbose
        self.grid = np.zeros((grid_size, grid_size), dtype=int)  # 0=空，1=障碍物，>=2=已铺砖

        if obstacle_num is None:
            obstacle_num = max(1, int(self.grid_size * self.grid_size * 0.1))

        if obstacle_positions is not None:
            self._apply_obstacles(obstacle_positions)
        else:
            self._generate_obstacles(obstacle_num)

        self.original_grid = self.grid.copy()
        self.block_types = block_types if block_types is not None else SCHEME_FULL
        if not self.block_types:
            raise ValueError("block_types 不能为空，至少配置一种方块")
        if any(block_id <= 1 for block_id in self.block_types):
            raise ValueError("方块编号必须大于1，编号1保留给障碍物")
        if not any(info[0] == 1 and info[1] == 1 for info in self.block_types.values()):
            raise ValueError("block_types 必须包含1x1方块，否则无法保证覆盖完整")

        self.block_counts = {block_id: 0 for block_id in self.block_types}
        self.placed_blocks = []
        self.tile_stage = None
        self.tile_solver_status = None
        self.tsp_stage = None
        self.tsp_solver_status = None

    def _apply_obstacles(self, obstacles):
        for x, y in obstacles:
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                self.grid[x][y] = 1

    def _generate_obstacles(self, obstacle_num):
        max_obstacles = self.grid_size * self.grid_size
        obstacle_num = min(obstacle_num, max_obstacles - 1)
        obstacles = set()
        while len(obstacles) < obstacle_num:
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            obstacles.add((x, y))
        self._apply_obstacles(obstacles)

    def _get_block_fill_order(self):
        return sorted(
            self.block_types.items(),
            key=lambda item: item[1][0] * item[1][1],
            reverse=True
        )

    def _is_area_available_on_mask(self, mask, covered_mask, x, y, h, w):
        if x + h > self.grid_size or y + w > self.grid_size:
            return False
        return np.all(mask[x:x + h, y:y + w]) and not np.any(covered_mask[x:x + h, y:y + w])

    def _fill_area_on_grid(self, grid, x, y, h, w, block_id):
        grid[x:x + h, y:y + w] = block_id

    def _block_counts_from_blocks(self, blocks):
        counts = {block_id: 0 for block_id in self.block_types}
        for block in blocks:
            counts[block['id']] += 1
        return counts

    def _build_grid_from_blocks(self, blocks):
        grid = self.original_grid.copy()
        for block in blocks:
            self._fill_area_on_grid(grid, block['x'], block['y'], block['h'], block['w'], block['id'])
        return grid

    def _greedy_best_of_four(self, region_mask):
        """回退后铺砖算法：四个扫描方向中选砖数最少者。"""
        def tile_in_order(x_scan_order, y_scan_order):
            covered_mask = np.zeros_like(region_mask, dtype=bool)
            placed_blocks = []
            block_counts = {block_id: 0 for block_id in self.block_types}

            for block_id, (h, w, _, _) in self._get_block_fill_order():
                for x in x_scan_order:
                    for y in y_scan_order:
                        if x >= self.grid_size or y >= self.grid_size:
                            continue
                        if covered_mask[x, y] or not region_mask[x, y]:
                            continue
                        if self._is_area_available_on_mask(region_mask, covered_mask, x, y, h, w):
                            covered_mask[x:x + h, y:y + w] = True
                            block_counts[block_id] += 1
                            placed_blocks.append({'x': x, 'y': y, 'h': h, 'w': w, 'id': block_id})

            # 兜底：1x1保证完整覆盖
            one_by_one_ids = [bid for bid, (h, w, _, _) in self.block_types.items() if h == 1 and w == 1]
            one_by_one_id = one_by_one_ids[0]
            for x in x_scan_order:
                for y in y_scan_order:
                    if not covered_mask[x, y] and region_mask[x, y]:
                        covered_mask[x, y] = True
                        block_counts[one_by_one_id] += 1
                        placed_blocks.append({'x': x, 'y': y, 'h': 1, 'w': 1, 'id': one_by_one_id})

            return len(placed_blocks), placed_blocks, block_counts, covered_mask

        scan_orders = [
            (list(range(self.grid_size)), list(range(self.grid_size))),
            (list(range(self.grid_size - 1, -1, -1)), list(range(self.grid_size - 1, -1, -1))),
            (list(range(self.grid_size)), list(range(self.grid_size - 1, -1, -1))),
            (list(range(self.grid_size - 1, -1, -1)), list(range(self.grid_size))),
        ]
        results = [tile_in_order(xs, ys) for xs, ys in scan_orders]
        total_region = int(np.sum(region_mask))
        valid_results = [r for r in results if int(np.sum(r[3])) == total_region]
        if not valid_results:
            # 理论上不会发生，因为1x1兜底可覆盖所有自由格。
            valid_results = results
        return min(valid_results, key=lambda r: r[0])

    def _milp_min_tile_cover(self, region_mask):
        """回退前铺砖算法：0-1整数规划最少砖数覆盖。"""
        if not HAS_MILP:
            raise RuntimeError("当前环境没有 scipy.optimize.milp")

        region_cells = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size) if region_mask[x, y]]
        total_region = len(region_cells)
        if total_region == 0:
            return [], {block_id: 0 for block_id in self.block_types}, np.zeros_like(region_mask, dtype=bool), "empty"
        if total_region > TILE_MILP_MAX_CELLS:
            raise RuntimeError(f"区域自由格={total_region}超过TILE_MILP_MAX_CELLS={TILE_MILP_MAX_CELLS}")

        cell_to_row = {cell: i for i, cell in enumerate(region_cells)}
        candidate_tiles = []
        candidate_cell_rows = []

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if not region_mask[x, y]:
                    continue
                for block_id, (h, w, _, _) in self._get_block_fill_order():
                    if x + h > self.grid_size or y + w > self.grid_size:
                        continue
                    if np.all(region_mask[x:x + h, y:y + w]):
                        cells = [(px, py) for px in range(x, x + h) for py in range(y, y + w)]
                        candidate_tiles.append({'x': x, 'y': y, 'h': h, 'w': w, 'id': block_id})
                        candidate_cell_rows.append([cell_to_row[c] for c in cells])

        if not candidate_tiles:
            raise RuntimeError("没有可行候选砖块")
        if len(candidate_tiles) > TILE_MILP_MAX_CANDIDATES:
            raise RuntimeError(f"候选砖块={len(candidate_tiles)}超过TILE_MILP_MAX_CANDIDATES={TILE_MILP_MAX_CANDIDATES}")

        rows, cols, data = [], [], []
        for j, cell_rows in enumerate(candidate_cell_rows):
            rows.extend(cell_rows)
            cols.extend([j] * len(cell_rows))
            data.extend([1.0] * len(cell_rows))
        A = coo_matrix((data, (rows, cols)), shape=(total_region, len(candidate_tiles))).tocsr()

        areas = np.array([block['h'] * block['w'] for block in candidate_tiles], dtype=float)
        # 主目标：砖数最少。极小扰动只用于同砖数时倾向大砖，不改变最少砖数目标。
        c = np.ones(len(candidate_tiles), dtype=float) + 1e-6 / areas

        constraints = LinearConstraint(A, lb=np.ones(total_region), ub=np.ones(total_region))
        integrality = np.ones(len(candidate_tiles), dtype=np.int8)
        bounds = Bounds(lb=np.zeros(len(candidate_tiles)), ub=np.ones(len(candidate_tiles)))

        res = milp(
            c=c,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options={"time_limit": ILP_TIME_LIMIT, "mip_rel_gap": 0.0, "disp": False},
        )
        if res.x is None:
            raise RuntimeError(f"MILP未返回可行解，status={res.status}, message={res.message}")

        selected_indices = np.where(res.x > 0.5)[0]
        placed_blocks = [candidate_tiles[i] for i in selected_indices]
        covered_mask = np.zeros_like(region_mask, dtype=bool)
        for block in placed_blocks:
            covered_mask[block['x']:block['x'] + block['h'], block['y']:block['y'] + block['w']] = True
        if int(np.sum(covered_mask)) != total_region:
            raise RuntimeError("MILP结果覆盖数量异常")

        status_text = "milp_optimal" if res.status == 0 else f"milp_feasible_not_proven_status_{res.status}"
        return placed_blocks, self._block_counts_from_blocks(placed_blocks), covered_mask, status_text

    def fill_grid(self):
        region_mask = (self.original_grid == 0)
        total_region = int(np.sum(region_mask))
        if total_region == 0:
            self.grid = self.original_grid.copy()
            self.placed_blocks = []
            self.block_counts = {block_id: 0 for block_id in self.block_types}
            self.tile_stage = BEFORE_FALLBACK
            self.tile_solver_status = "empty"
            return

        if TILE_OPTIMIZER == "milp":
            try:
                blocks, counts, _, status = self._milp_min_tile_cover(region_mask)
                self.placed_blocks = blocks
                self.block_counts = counts
                self.grid = self._build_grid_from_blocks(blocks)
                self.tile_stage = BEFORE_FALLBACK
                self.tile_solver_status = status
                return
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ 铺砖MILP失败/超阈值，使用贪心回退：{e}")

        total_blocks, blocks, counts, _ = self._greedy_best_of_four(region_mask)
        self.placed_blocks = blocks
        self.block_counts = counts
        self.grid = self._build_grid_from_blocks(blocks)
        self.tile_stage = AFTER_FALLBACK
        self.tile_solver_status = f"greedy_fallback_blocks_{total_blocks}"

    def calculate_tsp(self):
        result = compute_tile_tsp(self.placed_blocks, self.obstacle_positions(), self.grid_size)
        self.tsp_stage = result['tsp_stage']
        self.tsp_solver_status = result['solver_status']
        return result

    def obstacle_positions(self):
        pts = np.argwhere(self.original_grid == 1)
        return [(int(x), int(y)) for x, y in pts]


# ============================================================
# 连续平面避障最短路 + TSP
# ============================================================
def segment_rect_intersection_interval(p1, p2, rect, eps=1e-9):
    x1, y1 = p1
    x2, y2 = p2
    xmin, ymin, xmax, ymax = rect
    dx = x2 - x1
    dy = y2 - y1
    t0, t1 = 0.0, 1.0

    def clip(p, q):
        nonlocal t0, t1
        if abs(p) < eps:
            return q >= -eps
        r = q / p
        if p < 0:
            if r > t1 + eps:
                return False
            if r > t0:
                t0 = r
        else:
            if r < t0 - eps:
                return False
            if r < t1:
                t1 = r
        return True

    if not clip(-dx, x1 - xmin):
        return None
    if not clip(dx, xmax - x1):
        return None
    if not clip(-dy, y1 - ymin):
        return None
    if not clip(dy, ymax - y1):
        return None
    if t1 < t0 - eps:
        return None
    return max(0.0, t0), min(1.0, t1)


def segment_crosses_obstacle(p1, p2, obstacle_xy, eps=1e-9):
    ox, oy = obstacle_xy
    if ALLOW_OBSTACLE_BOUNDARY_SLIDING:
        rect = (ox + eps, oy + eps, ox + 1 - eps, oy + 1 - eps)
        if rect[0] >= rect[2] or rect[1] >= rect[3]:
            return False
        interval = segment_rect_intersection_interval(p1, p2, rect, eps=eps)
        return interval is not None

    rect = (ox, oy, ox + 1, oy + 1)
    interval = segment_rect_intersection_interval(p1, p2, rect, eps=eps)
    if interval is None:
        return False
    t0, t1 = interval
    return abs(t1 - t0) > eps


def is_visible_segment(p1, p2, obstacle_indices, eps=1e-9):
    if abs(p1[0] - p2[0]) <= eps and abs(p1[1] - p2[1]) <= eps:
        return False
    minx, maxx = min(p1[0], p2[0]) - eps, max(p1[0], p2[0]) + eps
    miny, maxy = min(p1[1], p2[1]) - eps, max(p1[1], p2[1]) + eps
    for ox, oy in obstacle_indices:
        if ox + 1 < minx or ox > maxx or oy + 1 < miny or oy > maxy:
            continue
        if segment_crosses_obstacle(p1, p2, (ox, oy), eps=eps):
            return False
    return True


def collect_obstacle_boundary_corners(obstacle_indices, map_size):
    obstacle_set = set(obstacle_indices)
    corners = set()
    for ox, oy in obstacle_set:
        for vx, vy in [(ox, oy), (ox + 1, oy), (ox, oy + 1), (ox + 1, oy + 1)]:
            adjacent_cells = [(vx - 1, vy - 1), (vx - 1, vy), (vx, vy - 1), (vx, vy)]
            has_obstacle = False
            has_free = False
            for cx, cy in adjacent_cells:
                if 0 <= cx < map_size and 0 <= cy < map_size:
                    if (cx, cy) in obstacle_set:
                        has_obstacle = True
                    else:
                        has_free = True
            if has_obstacle and has_free:
                corners.add((float(vx), float(vy)))
    return sorted(corners)


def build_continuous_visibility_graph(required_points, obstacle_indices, map_size):
    if not HAS_NETWORKX:
        raise RuntimeError("当前环境没有 networkx，无法构建连续避障可视图")
    obstacle_corners = collect_obstacle_boundary_corners(obstacle_indices, map_size)
    nodes = [tuple(map(float, p)) for p in required_points] + obstacle_corners

    G = nx.Graph()
    for idx, p in enumerate(nodes):
        node_type = 'required' if idx < len(required_points) else 'corner'
        G.add_node(idx, pos=p, node_type=node_type)

    for i in range(len(nodes)):
        p1 = nodes[i]
        for j in range(i + 1, len(nodes)):
            p2 = nodes[j]
            if is_visible_segment(p1, p2, obstacle_indices, eps=VISIBILITY_EPS):
                dist = float(math.hypot(p1[0] - p2[0], p1[1] - p2[1]))
                G.add_edge(i, j, weight=dist)
    return G, nodes, obstacle_corners


def compute_required_shortest_paths(visibility_graph, required_count):
    dist_matrix = np.full((required_count, required_count), np.inf, dtype=float)
    path_dict = {}
    for src in range(required_count):
        lengths, paths = nx.single_source_dijkstra(visibility_graph, source=src, weight='weight')
        for dst in range(required_count):
            if dst == src:
                dist_matrix[src, dst] = 0.0
                path_dict[(src, dst)] = [src]
            elif dst in lengths:
                dist_matrix[src, dst] = float(lengths[dst])
                path_dict[(src, dst)] = paths[dst]
    return dist_matrix, path_dict


def solve_tsp_exact_mtz(distance_matrix, time_limit=60.0):
    if not HAS_MILP:
        raise RuntimeError("当前SciPy版本没有 scipy.optimize.milp，不能使用精确MTZ TSP")
    n = distance_matrix.shape[0]
    if n <= 1:
        return list(range(n)), 0.0, "trivial"
    if not np.all(np.isfinite(distance_matrix)):
        raise RuntimeError("距离矩阵存在不可达点对，不能求闭合TSP")

    arc_index = {}
    idx = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            arc_index[(i, j)] = idx
            idx += 1
    x_count = idx
    u_index = {}
    for i in range(1, n):
        u_index[i] = idx
        idx += 1

    var_count = idx
    c = np.zeros(var_count, dtype=float)
    for (i, j), k in arc_index.items():
        c[k] = distance_matrix[i, j]

    lb = np.zeros(var_count, dtype=float)
    ub = np.ones(var_count, dtype=float)
    for i in range(1, n):
        lb[u_index[i]] = 1.0
        ub[u_index[i]] = float(n - 1)

    integrality = np.zeros(var_count, dtype=np.int8)
    integrality[:x_count] = 1

    rows, cols, data = [], [], []
    cons_lb, cons_ub = [], []
    row = 0

    for i in range(n):
        for j in range(n):
            if i != j:
                rows.append(row)
                cols.append(arc_index[(i, j)])
                data.append(1.0)
        cons_lb.append(1.0)
        cons_ub.append(1.0)
        row += 1

    for j in range(n):
        for i in range(n):
            if i != j:
                rows.append(row)
                cols.append(arc_index[(i, j)])
                data.append(1.0)
        cons_lb.append(1.0)
        cons_ub.append(1.0)
        row += 1

    for i in range(1, n):
        for j in range(1, n):
            if i == j:
                continue
            rows.append(row)
            cols.append(u_index[i])
            data.append(1.0)
            rows.append(row)
            cols.append(u_index[j])
            data.append(-1.0)
            rows.append(row)
            cols.append(arc_index[(i, j)])
            data.append(float(n))
            cons_lb.append(-np.inf)
            cons_ub.append(float(n - 1))
            row += 1

    A = coo_matrix((data, (rows, cols)), shape=(row, var_count)).tocsr()
    constraints = LinearConstraint(A, lb=np.array(cons_lb), ub=np.array(cons_ub))
    res = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lb=lb, ub=ub),
        constraints=constraints,
        options={"time_limit": time_limit, "mip_rel_gap": 0.0, "disp": False},
    )
    if res.x is None:
        raise RuntimeError(f"精确TSP未返回可行解，status={res.status}, message={res.message}")

    selected = {(i, j) for (i, j), k in arc_index.items() if res.x[k] > 0.5}
    successor = {i: j for i, j in selected}
    route = [0]
    current = 0
    visited = {0}
    for _ in range(n + 1):
        if current not in successor:
            raise RuntimeError("TSP解解析失败：缺少后继节点")
        nxt = successor[current]
        route.append(nxt)
        current = nxt
        if current == 0:
            break
        if current in visited:
            raise RuntimeError("TSP解解析失败：发现异常子回路")
        visited.add(current)

    if route[-1] != 0 or len(set(route[:-1])) != n:
        raise RuntimeError("TSP解没有形成包含所有节点的闭合回路")

    objective = sum(distance_matrix[u, v] for u, v in zip(route[:-1], route[1:]))
    status_text = "exact_mtz_optimal" if res.status == 0 else f"exact_mtz_feasible_not_proven_status_{res.status}"
    return route, float(objective), status_text


def solve_tsp_christofides(distance_matrix):
    if not HAS_NETWORKX:
        raise RuntimeError("当前环境没有 networkx，不能使用Christofides回退")
    n = distance_matrix.shape[0]
    if n <= 1:
        return list(range(n)), 0.0, "trivial"
    if not np.all(np.isfinite(distance_matrix)):
        raise RuntimeError("距离矩阵存在不可达点对，不能用Christofides求TSP")

    K = nx.Graph()
    K.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            K.add_edge(i, j, weight=float(distance_matrix[i, j]))

    route = nx.approximation.traveling_salesman_problem(
        K,
        weight='weight',
        cycle=True,
        method=nx.approximation.christofides,
    )
    if route[0] != route[-1]:
        route.append(route[0])
    if 0 in route[:-1]:
        k = route[:-1].index(0)
        route = route[k:-1] + route[:k] + [0]
    objective = sum(distance_matrix[u, v] for u, v in zip(route[:-1], route[1:]))
    return route, float(objective), "christofides_after_fallback"


def solve_tsp_route(distance_matrix):
    """
    严格按照用户给定规则选择TSP算法：
    1) 若砖块中心数 <= TSP_EXACT_MAX_POINTS(80)，尝试MTZ精确TSP；
    2) 若点数 > 80，直接回退到Christofides；
    3) 若MTZ在TSP_TIME_LIMIT内失败、超时或未返回可用解，回退到Christofides。

    回退前/后标记：
    - BEFORE_FALLBACK：MTZ精确TSP成功返回；
    - AFTER_FALLBACK：Christofides回退。
    """
    n = distance_matrix.shape[0]
    if n <= 1:
        return list(range(n)), 0.0, "trivial", BEFORE_FALLBACK

    if TSP_SOLVER == "exact_mtz" and n <= TSP_EXACT_MAX_POINTS:
        try:
            route, length, status = solve_tsp_exact_mtz(distance_matrix, time_limit=TSP_TIME_LIMIT)
            return route, length, status, BEFORE_FALLBACK
        except Exception as e:
            route, length, status = solve_tsp_christofides(distance_matrix)
            return route, length, f"{status}; exact_mtz_failed_or_timeout: {e}", AFTER_FALLBACK

    route, length, status = solve_tsp_christofides(distance_matrix)
    return route, length, status, AFTER_FALLBACK


def compute_tile_tsp(blocks, obstacle_indices, map_size):
    """
    对全部砖块中心构造连续避障距离矩阵，然后按用户指定规则求TSP：
    - n <= 80：MTZ精确TSP，失败/超时回退Christofides；
    - n > 80：Christofides。

    本函数不包含 fast_sweep、fast_nearest_neighbor 或其他额外回退算法。
    """
    if len(blocks) == 0:
        return {
            'total_length': 0.0,
            'tile_count': 0,
            'solver_status': 'empty',
            'tsp_stage': BEFORE_FALLBACK,
        }

    centroids = np.array(
        [(block['x'] + block['h'] / 2.0, block['y'] + block['w'] / 2.0) for block in blocks],
        dtype=float,
    )
    n = len(centroids)

    if n == 1:
        return {
            'total_length': 0.0,
            'tile_count': 1,
            'solver_status': 'trivial',
            'tsp_stage': BEFORE_FALLBACK,
        }

    visibility_graph, _, obstacle_corners = build_continuous_visibility_graph(
        centroids, obstacle_indices, map_size
    )
    distance_matrix, _ = compute_required_shortest_paths(visibility_graph, n)
    route, length, status, stage = solve_tsp_route(distance_matrix)

    return {
        'total_length': length,
        'tile_count': n,
        'solver_status': status,
        'tsp_stage': stage,
        'visibility_node_count': visibility_graph.number_of_nodes(),
        'visibility_edge_count': visibility_graph.number_of_edges(),
        'obstacle_corner_count': len(obstacle_corners),
    }


# ============================================================
# 实验流程：保持原脚本的尺寸、密度、50种子、并行和输出结构
# ============================================================
def generate_obstacles_for_size(grid_size, obstacle_num, seed):
    rng = random.Random(seed)
    max_obstacles = grid_size * grid_size - 1
    obstacle_num = max(1, min(obstacle_num, max_obstacles))
    obstacles = set()
    while len(obstacles) < obstacle_num:
        obstacles.add((rng.randint(0, grid_size - 1), rng.randint(0, grid_size - 1)))
    return obstacles


def _stage_to_counts(stage):
    return (1, 0) if stage == BEFORE_FALLBACK else (0, 1)


# 定义被多进程调用的单次实验任务
def _run_single_experiment(args):
    grid_size, obstacle_ratio, seed = args
    obstacle_num = max(1, int(grid_size * grid_size * obstacle_ratio))
    obstacles = generate_obstacles_for_size(grid_size, obstacle_num, seed)

    # 方案 A：SCHEME_FULL
    filler_a = GridFiller(
        grid_size=grid_size,
        obstacle_num=obstacle_num,
        block_types=SCHEME_FULL,
        obstacle_positions=obstacles,
        verbose=False,
    )
    start_a = time.perf_counter()
    filler_a.fill_grid()
    blocks_a = sum(filler_a.block_counts.values())
    tsp_a_data = filler_a.calculate_tsp()
    tsp_a = tsp_a_data['total_length']
    time_a = time.perf_counter() - start_a
    a_tile_before, a_tile_after = _stage_to_counts(filler_a.tile_stage)
    a_tsp_before, a_tsp_after = _stage_to_counts(tsp_a_data['tsp_stage'])

    # 方案 B：SCHEME_SIMPLE
    filler_b = GridFiller(
        grid_size=grid_size,
        obstacle_num=obstacle_num,
        block_types=SCHEME_SIMPLE,
        obstacle_positions=obstacles,
        verbose=False,
    )
    start_b = time.perf_counter()
    filler_b.fill_grid()
    blocks_b = sum(filler_b.block_counts.values())
    tsp_b_data = filler_b.calculate_tsp()
    tsp_b = tsp_b_data['total_length']
    time_b = time.perf_counter() - start_b
    b_tile_before, b_tile_after = _stage_to_counts(filler_b.tile_stage)
    b_tsp_before, b_tsp_after = _stage_to_counts(tsp_b_data['tsp_stage'])

    return (
        blocks_a, tsp_a, time_a, a_tile_before, a_tile_after, a_tsp_before, a_tsp_after,
        blocks_b, tsp_b, time_b, b_tile_before, b_tile_after, b_tsp_before, b_tsp_after,
    )


def run_experiment_for_ratio(min_size, max_size, step, obstacle_ratio, seeds_list):
    sizes = list(range(min_size, max_size + 1, step))
    lines = [
        "grid_size,"
        "A_blocks,A_tsp,A_time,A_tile_before_count,A_tile_after_count,A_tsp_before_count,A_tsp_after_count,"
        "B_blocks,B_tsp,B_time,B_tile_before_count,B_tile_after_count,B_tsp_before_count,B_tsp_after_count"
    ]
    num_experiments = len(seeds_list)
    max_workers = max(1, os.cpu_count() - 1)

    for grid_size in sizes:
        sum_blocks_a = sum_tsp_a = sum_time_a = 0.0
        sum_blocks_b = sum_tsp_b = sum_time_b = 0.0
        a_tile_before = a_tile_after = a_tsp_before = a_tsp_after = 0
        b_tile_before = b_tile_after = b_tsp_before = b_tsp_after = 0

        tasks = [(grid_size, obstacle_ratio, seed) for seed in seeds_list]
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_run_single_experiment, tasks))

        for res in results:
            (
                b_a, tsp_a, t_a, atb, ata, aspb, aspa,
                b_b, tsp_b, t_b, btb, bta, bspb, bspa,
            ) = res
            sum_blocks_a += b_a
            sum_tsp_a += tsp_a
            sum_time_a += t_a
            a_tile_before += atb
            a_tile_after += ata
            a_tsp_before += aspb
            a_tsp_after += aspa

            sum_blocks_b += b_b
            sum_tsp_b += tsp_b
            sum_time_b += t_b
            b_tile_before += btb
            b_tile_after += bta
            b_tsp_before += bspb
            b_tsp_after += bspa

        avg_blocks_a = sum_blocks_a / num_experiments
        avg_tsp_a = sum_tsp_a / num_experiments
        avg_time_a = sum_time_a / num_experiments
        avg_blocks_b = sum_blocks_b / num_experiments
        avg_tsp_b = sum_tsp_b / num_experiments
        avg_time_b = sum_time_b / num_experiments

        print(
            f"密度 {int(obstacle_ratio * 100)}% | 网格 {grid_size}x{grid_size} (基于50个种子平均): "
            f"A(铺砖={avg_blocks_a:.2f}, TSP={avg_tsp_a:.2f}, 时间={avg_time_a:.4f}s, "
            f"铺砖前/后={a_tile_before}/{a_tile_after}, TSP前/后={a_tsp_before}/{a_tsp_after}) | "
            f"B(铺砖={avg_blocks_b:.2f}, TSP={avg_tsp_b:.2f}, 时间={avg_time_b:.4f}s, "
            f"铺砖前/后={b_tile_before}/{b_tile_after}, TSP前/后={b_tsp_before}/{b_tsp_after})"
        )

        lines.append(
            f"{grid_size},"
            f"{avg_blocks_a:.6f},{avg_tsp_a:.6f},{avg_time_a:.6f},{a_tile_before},{a_tile_after},{a_tsp_before},{a_tsp_after},"
            f"{avg_blocks_b:.6f},{avg_tsp_b:.6f},{avg_time_b:.6f},{b_tile_before},{b_tile_after},{b_tsp_before},{b_tsp_after}"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    # 多进程在 Windows 环境下必须在这个保护块内执行
    min_size = 20
    max_size = 200
    step = 10

    output_dir = os.path.dirname(os.path.abspath(__file__))
    if output_dir == "":
        output_dir = "."

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_path = os.path.join(output_dir, f'experiment_data_{timestamp}.txt')

    ratios = [0.10, 0.15, 0.20]
    all_experiments_output = []

    print("\n================ 算法标记说明 ================")
    print("铺砖 before_fallback = MILP最少砖数；铺砖 after_fallback = 四方向贪心回退")
    print("TSP  before_fallback = 连续避障距离 + MTZ精确TSP；TSP  after_fallback = Christofides回退")

    for ratio in ratios:
        ratio_int = int(ratio * 100)
        print(f"\n================ 开始运行障碍物密度 {ratio_int}% 的实验 (使用指定的50个种子) ================")

        start_ratio_time = time.perf_counter()
        data_str = run_experiment_for_ratio(min_size, max_size, step, ratio, EXPERIMENT_SEEDS)
        end_ratio_time = time.perf_counter()

        print(f"[{ratio_int}% 实验完成] 耗时: {end_ratio_time - start_ratio_time:.2f} 秒")

        var_name = f"data_{ratio_int}_str"
        final_output = f'{var_name} = """\n{data_str}\n"""'
        all_experiments_output.append(final_output)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(all_experiments_output))

    print(f"\n>> 所有实验完成！平均值数据已成功保存至: {file_path}")
