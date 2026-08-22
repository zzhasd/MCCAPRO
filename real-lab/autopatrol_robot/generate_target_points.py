from time import sleep

import yaml
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import os
import math
import collections
import datetime
import cv2                                  
import scipy.sparse as sp                   
from scipy.sparse.csgraph import dijkstra   

# =========================================================================================
# 🌍 动态加载 YAML 配置文件
# =========================================================================================
# 优先使用源码目录绝对路径，保证修改 yaml 后即时生效，免除重新 colcon build 的烦恼
CONFIG_FILE_PATH = '/home/jetson/chapt7_ws/src/autopatrol_robot/autopatrol_robot/config.yaml'

def load_global_config():
    target_path = CONFIG_FILE_PATH
    if not os.path.exists(target_path):
        # 兼容本地 Windows 测试：如果绝对路径不存在，则寻找当前目录下的 config.yaml
        target_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
        
    try:
        with open(target_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            print(f"读取yaml成功: {target_path}")
            return config
    except Exception as e:
        print(f"⚠️ 读取yaml失败: {e}，将使用默认空配置")
        return {}

global_config = load_global_config()

# ====================== 从 YAML 提取配置 ======================
SELECT_CONFIG = global_config.get('select_config', 1)
YAML_PATH = global_config.get('paths', {}).get('map_yaml', {}).get(SELECT_CONFIG, 'test_yaml')
IMAGE_DIR = global_config.get('paths', {}).get('image_dir', {}).get(SELECT_CONFIG, '')
BASE_SIZE = global_config.get('algorithm', {}).get('base_size', 15)
CELL_SIZE = global_config.get('algorithm', {}).get('cell_size', 1)
SAFETY_RADIUS_M = global_config.get('algorithm', {}).get('safety_radius_m', 0)
WHITE_RATIO_THRESHOLD_CONFIG = global_config.get('algorithm', {}).get('white_ratio_threshold', 0.85)

# =========================================================================================
# 🛠️ 核心工具函数
# =========================================================================================
def load_map_core_params(yaml_file, base_size):
    yaml_file = os.path.abspath(yaml_file)
    with open(yaml_file, 'r') as f:
        metadata = yaml.safe_load(f)
    
    resolution = metadata['resolution']
    origin = metadata['origin']
    free_thresh = metadata['free_thresh']
    
    yaml_dir = os.path.dirname(yaml_file)
    pgm_path = os.path.join(yaml_dir, metadata['image'])
    if not os.path.exists(pgm_path):
        raise FileNotFoundError(f"PGM文件不存在：{pgm_path}")
    
    with open(pgm_path, 'rb') as f:
        f.readline()
        while True:
            line = f.readline().strip()
            if not line.startswith(b'#'):
                map_w_px, map_h_px = map(int, line.split())
                break
    
    map_img = np.array(Image.open(pgm_path).convert('L'))
    
    return {
        'resolution': resolution, 'origin': origin, 'free_thresh': free_thresh,
        'map_img': map_img, 'map_w_px': map_w_px, 'map_h_px': map_h_px,
        'grid_reso': base_size * resolution, 
        'grid_h': map_h_px // base_size, 
        'grid_w': map_w_px // base_size,
        'base_size': base_size  
    }

def build_obstacle_grid(map_params, safety_radius_m):
    grid_h, grid_w = map_params['grid_h'], map_params['grid_w']
    map_img = map_params['map_img']
    base_size = map_params['base_size']
    safe_thresh = map_params.get('safe_thresh', 0.9) 
    
    pixel_safe = int(safety_radius_m / map_params['resolution'])
    WHITE_RATIO_THRESHOLD = WHITE_RATIO_THRESHOLD_CONFIG # 从全局配置读取
    
    obstacle_grid = np.zeros((grid_h, grid_w), dtype=int)
    for r in range(grid_h):
        for c in range(grid_w):
            sr = max(0, r * base_size - pixel_safe)
            er = min(map_params['map_h_px'], (r + 1) * base_size + pixel_safe)
            sc = max(0, c * base_size - pixel_safe)
            ec = min(map_params['map_w_px'], (c + 1) * base_size + pixel_safe)
            
            region_pixels = map_img[sr:er, sc:ec] / 255.0
            total_pixels = region_pixels.size
            
            if total_pixels == 0:
                obstacle_grid[r, c] = 1
                continue
                
            white_pixels = np.sum(region_pixels >= safe_thresh)
            white_ratio = white_pixels / total_pixels
            
            if white_ratio <= WHITE_RATIO_THRESHOLD:
                obstacle_grid[r, c] = 1
                
    return obstacle_grid

def partition_grid(obstacle_grid, robot_num, weights):
    h, w = obstacle_grid.shape
    free_cells = [(r, c) for r in range(h) for c in range(w) if obstacle_grid[r, c] == 0]
    if not free_cells:
        return np.full((h, w), -1)

    xy_to_idx = {rc: idx for idx, rc in enumerate(free_cells)}
    row, col, data = [], [], []
    
    for r, c in free_cells:
        u = xy_to_idx[(r, c)]
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and obstacle_grid[nr, nc] == 0:
                row.append(u)
                col.append(xy_to_idx[(nr, nc)])
                data.append(1.0)
                
    adj_matrix = sp.csr_matrix((data, (row, col)), shape=(len(free_cells), len(free_cells)))
    centroids_rc = []
    step = min(h, w) / np.sqrt(robot_num)
    
    for i in range(robot_num):
        c_p = int(np.clip(round((i % int(np.ceil(np.sqrt(robot_num)))) * step + step / 2 - 0.5), 0, w - 1))
        r_p = int(np.clip(round((i // int(np.ceil(np.sqrt(robot_num)))) * step + step / 2 - 0.5), 0, h - 1))
        
        if obstacle_grid[r_p, c_p] == 1:
            found = False
            for dist in range(1, max(h, w)):
                for dr in range(-dist, dist + 1):
                    for dc in range(-dist, dist + 1):
                        if abs(dr) + abs(dc) != dist: continue
                        nr, nc = r_p + dr, c_p + dc
                        if 0 <= nr < h and 0 <= nc < w and obstacle_grid[nr, nc] == 0:
                            r_p, c_p = nr, nc
                            found = True
                            break
                    if found: break
                if found: break
        centroids_rc.append((r_p, c_p))

    for _ in range(20):
        all_distances = []
        for rc in centroids_rc:
            start_idx = xy_to_idx.get(rc)
            if start_idx is not None:
                dist = dijkstra(csgraph=adj_matrix, directed=False, indices=start_idx, return_predecessors=False)
            else:
                dist = np.full(len(free_cells), np.inf)
            all_distances.append(dist)
            
        all_distances = np.array(all_distances)
        sqrt_weights = np.sqrt(weights).reshape(-1, 1)
        weighted_distances = all_distances / (sqrt_weights + 1e-9)
        
        min_dist = np.min(weighted_distances, axis=0)
        grid_assignments = np.full(len(free_cells), -1)
        valid_mask = min_dist < np.inf
        grid_assignments[valid_mask] = np.argmin(weighted_distances[:, valid_mask], axis=0)
        
        new_centroids_rc = []
        for i in range(robot_num):
            region_indices = np.where(grid_assignments == i)[0]
            if len(region_indices) == 0:
                new_centroids_rc.append(centroids_rc[i])
                continue
            
            region_rc = [free_cells[idx] for idx in region_indices]
            mean_r, mean_c = np.mean([rc[0] for rc in region_rc]), np.mean([rc[1] for rc in region_rc])
            
            best_rc = min(region_rc, key=lambda rc: abs(rc[0] - mean_r) + abs(rc[1] - mean_c))
            new_centroids_rc.append(best_rc)
            
        if centroids_rc == new_centroids_rc:
            break
        centroids_rc = new_centroids_rc
        
    assignment_grid = np.full((h, w), -1)
    for idx, rc in enumerate(free_cells):
        assignment_grid[rc[0], rc[1]] = grid_assignments[idx]
        
    return assignment_grid

class GridFiller:
    def __init__(self, obstacle_grid, cell_size):
        self.grid_h, self.grid_w = obstacle_grid.shape
        self.obstacle_grid = obstacle_grid.copy()
        self.cell_size = cell_size
        self.block_defs = {
            6: (4, 4, '#FFA502'), 7: (4, 2, '#FF6348'),
            8: (2, 4, '#1E90FF'), 2: (2, 2, '#FF6B6B'),
            3: (2, 1, '#4ECDC4'), 4: (1, 2, '#45B7D1'),
            5: (1, 1, '#96CEB4')
        }
        self.placed_blocks = []
    
    def _get_traversal_order(self, direction, block_h, block_w):
        max_r, max_c = self.grid_h - block_h, self.grid_w - block_w
        r_range, c_range = range(max_r + 1), range(max_c + 1)
        
        if direction == 'top_left_to_bottom_right': return [(r, c) for r in r_range for c in c_range]
        if direction == 'bottom_right_to_top_left': return [(r, c) for r in reversed(r_range) for c in reversed(c_range)]
        if direction == 'bottom_left_to_top_right': return [(r, c) for r in reversed(r_range) for c in c_range]
        return [(r, c) for r in r_range for c in reversed(c_range)]

    def _fill_direction(self, direction):
        grid = self.obstacle_grid.copy()
        placed = []
        counts = 0
        sorted_bids = sorted(self.block_defs.keys(), key=lambda k: self.block_defs[k][0] * self.block_defs[k][1], reverse=True)
        
        for bid in sorted_bids:
            bh_base, bw_base, _ = self.block_defs[bid]
            bh, bw = bh_base * self.cell_size, bw_base * self.cell_size
            
            for r, c in self._get_traversal_order(direction, bh, bw):
                if r + bh > self.grid_h or c + bw > self.grid_w: continue
                if all(grid[r+i][c+j] == 0 for i in range(bh) for j in range(bw)):
                    for i in range(bh):
                        for j in range(bw): grid[r+i][c+j] = bid
                    counts += 1
                    placed.append({'r': r, 'c': c, 'bh': bh, 'bw': bw, 'id': bid})
        return counts, placed

    def fill_grid(self):
        directions = ['top_left_to_bottom_right', 'bottom_right_to_top_left',
                      'bottom_left_to_top_right', 'top_right_to_bottom_left']
        best_total = float('inf')
        
        for d in directions:
            total, placed = self._fill_direction(d)
            if total < best_total:
                best_total = total
                self.placed_blocks = placed

    def calculate_mst(self, world_pts):
        if len(world_pts) <= 1: return []
        edges, n = len(world_pts), len(world_pts)
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                b1, b2 = self.placed_blocks[i], self.placed_blocks[j]
                y_overlap = max(b1['r'], b2['r']) < min(b1['r'] + b1['bh'], b2['r'] + b2['bh'])
                x_overlap = max(b1['c'], b2['c']) < min(b1['c'] + b1['bw'], b2['c'] + b2['bw'])
                x_touch = (b1['c'] + b1['bw'] == b2['c']) or (b2['c'] + b2['bw'] == b1['c'])
                y_touch = (b1['r'] + b1['bh'] == b2['r']) or (b2['r'] + b2['bh'] == b1['r'])
                
                if (x_touch and y_overlap) or (y_touch and x_overlap):
                    p1, p2 = world_pts[i], world_pts[j]
                    edges.append((math.hypot(p1[0]-p2[0], p1[1]-p2[1]), i, j))
                    
        edges.sort()
        parent = list(range(n))
        def find(u):
            if parent[u] != u: parent[u] = find(parent[u])
            return parent[u]
            
        mst_edges, components = [], n 
        for _, u, v in edges:
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv
                mst_edges.append((world_pts[u], world_pts[v]))
                components -= 1
                if components == 1: break
        return mst_edges

# =========================================================================================
# ⚙️ 核心内部执行函数（私有，不暴露给机器人调用）
# =========================================================================================
def _compute_all_robot_goals(robot_num, weights, query_robot_id=None, query_robot_pos=None):
    """ 统一计算所有机器人的路径点序列，将结果存入数组，内部调用计算 """
    map_params = load_map_core_params(YAML_PATH, BASE_SIZE)
    ox, oy, _ = map_params['origin']       
    grid_reso = map_params['grid_reso']    
    resolution = map_params['resolution']
    orig_map_img = map_params['map_img'].copy()
    orig_h, orig_w = orig_map_img.shape
    
    # 移除PCA旋转相关代码，直接基于原图网格化
    obstacle_grid = build_obstacle_grid(map_params, SAFETY_RADIUS_M)
    
    if robot_num > 1:
        assignment_grid = partition_grid(obstacle_grid, robot_num, weights)
    else:
        assignment_grid = np.zeros_like(obstacle_grid)
        assignment_grid[obstacle_grid == 1] = -1

    all_robot_goals = []
    all_placed_blocks_data = [] 
    all_mst_edges = []
    all_sequences = []

    for rid in range(robot_num):
        robot_grid = obstacle_grid.copy()
        robot_grid[assignment_grid != rid] = 1 
        
        filler = GridFiller(robot_grid, CELL_SIZE)
        filler.fill_grid()
        
        world_points, block_world_centers = [], []
        
        for block in filler.placed_blocks:
            # 获取块中心的像素坐标系下的U, V
            px_u = (block['c'] + block['bw'] / 2.0) * BASE_SIZE
            px_v = (block['r'] + block['bh'] / 2.0) * BASE_SIZE
            
            # 直接转换为 ROS Map 的真实物理世界坐标 (原点+偏移量)
            # ROS坐标系中图片是从左下角作为世界坐标系基准(通常oy在图片最下方)
            wx = ox + px_u * resolution
            wy = oy + (orig_h - px_v) * resolution  
            
            world_points.append((wx, wy))
            block_world_centers.append((wx, wy)) 
            
        all_placed_blocks_data.append((filler.placed_blocks, block_world_centers))
            
        if not world_points:
            all_robot_goals.append([])
            all_mst_edges.append([])
            all_sequences.append([])
            continue
        
        mst_edges = filler.calculate_mst(world_points)
        all_mst_edges.append(mst_edges)
        
        adj = collections.defaultdict(list)
        for p1, p2 in mst_edges:
            adj[p1].append(p2)
            adj[p2].append(p1)
            
        # 选择起点逻辑
        if rid == query_robot_id and query_robot_pos is not None:
            curr_x, curr_y = query_robot_pos
            start_point = min(world_points, key=lambda p: math.hypot(p[0] - curr_x, p[1] - curr_y))
        else:
            start_point = min(world_points, key=lambda p: math.hypot(p[0], p[1]))

        sequence, visited = [], set()
        
        def stc_dfs(curr_pt):
            visited.add(curr_pt)
            sequence.append(curr_pt)
            for neighbor in adj[curr_pt]:
                if neighbor not in visited:
                    stc_dfs(neighbor)
                    sequence.append(curr_pt)
                    
        stc_dfs(start_point)
        all_sequences.append(sequence)

        goal_points = []
        for i in range(len(sequence)):
            x, y = sequence[i]
            yaw = math.atan2(sequence[i+1][1] - y, sequence[i+1][0] - x) if i < len(sequence)-1 else (goal_points[-1][2] if goal_points else 0.0)
            goal_points.append((round(x, 2), round(y, 2), round(yaw, 2)))
        
        all_robot_goals.append(goal_points)
    
    # --- 保存可视化图像 ---
    if IMAGE_DIR:
        plt.figure(figsize=(14, 12))
        plt.imshow(orig_map_img, cmap='gray', extent=[ox, ox + orig_w * resolution, oy, oy + orig_h * resolution])
        colors = ['yellow', 'cyan', 'lime', 'magenta', 'orange', 'pink', 'teal', 'purple']
        dummy_filler = GridFiller(np.zeros((1, 1)), CELL_SIZE) 
        
        # 记录是否绘制了图例标签的标志，防止重复
        legend_drawn = False
        
        for rid in range(robot_num):
            blocks, centers = all_placed_blocks_data[rid]
            edge_color = colors[rid % len(colors)]
            for block, (wx, wy) in zip(blocks, centers):
                bh, bw, bid = block['bh'], block['bw'], block['id']
                rect_w, rect_h = bw * grid_reso, bh * grid_reso
                # 直接添加方块不进行任何角度的倾斜变化
                rect = patches.Rectangle((wx - rect_w/2, wy - rect_h/2), rect_w, rect_h,
                                         linewidth=2.5, edgecolor=edge_color,
                                         facecolor=dummy_filler.block_defs[bid][2], alpha=0.5)
                
                plt.gca().add_patch(rect)
            
            for p1, p2 in all_mst_edges[rid]:
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=edge_color, linewidth=2, alpha=0.9)
                     
            if all_sequences[rid]:
                seq_x, seq_y = [p[0] for p in all_sequences[rid]], [p[1] for p in all_sequences[rid]]
                
                # 新增：绘制绕树路径的连线（使用虚线连接所有路径点）
                path_label = 'STC Path' if not legend_drawn else ""
                plt.plot(seq_x, seq_y, color=edge_color, linestyle='--', linewidth=1.5, zorder=3, label=path_label)
                
                plt.scatter(seq_x, seq_y, c=edge_color, s=25, edgecolors='black', zorder=4)
                
                # 新增：标记路径点的访问顺序 (合并同一位置的多次访问序号，防重叠)
                point_orders = collections.defaultdict(list)
                for idx, (px, py) in enumerate(all_sequences[rid]):
                    point_orders[(px, py)].append(str(idx))
                
                for (px, py), orders in point_orders.items():
                    order_str = ",".join(orders)
                    # 添加微小的偏移量避免被路径点挡住
                    plt.text(px + 0.1, py + 0.1, order_str, color='black', fontsize=7, 
                             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.3), zorder=5)

                # 绘制 DFS 计算出的路线起点（红色星号）
                plt.scatter(seq_x[0], seq_y[0], c='red', s=150, marker='*', zorder=5, 
                            label='DFS Start Node' if not legend_drawn else "")
                
                # --- 绘制所有机器人的物理位置 ---
                if rid == query_robot_id and query_robot_pos is not None:
                    curr_x, curr_y = query_robot_pos
                else:
                    curr_x, curr_y = (0.0, 0.0)
                
                pos_label = 'Robot Position' if not legend_drawn else ""
                
                plt.scatter(curr_x, curr_y, c=edge_color, s=150, marker='^', zorder=6, edgecolors='black', label=pos_label)
                plt.text(curr_x + 0.15, curr_y + 0.15, f"R{rid}", color=edge_color, fontsize=10, fontweight='bold')
                
                legend_drawn = True
        
        plt.legend(loc='upper right', framealpha=0.9)
        plt.axis('equal')
        plt.axis('off')
        
        os.makedirs(IMAGE_DIR, exist_ok=True)
        save_path = os.path.join(IMAGE_DIR, f"multi_robot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        print(f"🔍 目标点分配与可视化已完成，保存至: {save_path}")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return all_robot_goals


def generate_assignment_rectangles(weights: list) -> dict:
    """
    计算当前权重下每台机器人被分配区域的矩形边框数据，供前端 HTML 叠加绘制。

    返回格式：
    {
        "weights": [...],
        "rects": [
            [ {"rid": 0, "bid": 6, "x": ..., "y": ..., "w": ..., "h": ...}, ... ],
            [ {"rid": 1, "bid": 6, "x": ..., "y": ..., "w": ..., "h": ...}, ... ]
        ]
    }

    注意：这里只输出矩形几何信息，不输出 facecolor；前端只按轨迹颜色画浅色边框。
    """
    weights_arr = np.array(weights, dtype=float)
    robot_num = len(weights_arr)

    if robot_num <= 0:
        return {"weights": [], "rects": []}

    map_params = load_map_core_params(YAML_PATH, BASE_SIZE)
    ox, oy, _ = map_params['origin']
    grid_reso = map_params['grid_reso']
    resolution = map_params['resolution']
    orig_h = map_params['map_img'].shape[0]

    obstacle_grid = build_obstacle_grid(map_params, SAFETY_RADIUS_M)

    if robot_num > 1:
        assignment_grid = partition_grid(obstacle_grid, robot_num, weights_arr)
    else:
        assignment_grid = np.zeros_like(obstacle_grid)
        assignment_grid[obstacle_grid == 1] = -1

    rects_by_robot = []

    for rid in range(robot_num):
        robot_grid = obstacle_grid.copy()
        robot_grid[assignment_grid != rid] = 1

        filler = GridFiller(robot_grid, CELL_SIZE)
        filler.fill_grid()

        robot_rects = []
        for block in filler.placed_blocks:
            # 块中心像素坐标
            px_u = (block['c'] + block['bw'] / 2.0) * BASE_SIZE
            px_v = (block['r'] + block['bh'] / 2.0) * BASE_SIZE

            # 转 ROS/world 坐标
            wx = ox + px_u * resolution
            wy = oy + (orig_h - px_v) * resolution

            rect_w = block['bw'] * grid_reso
            rect_h = block['bh'] * grid_reso

            # 输出左下角 + 宽高，前端负责把 world 坐标转 canvas 像素坐标
            robot_rects.append({
                "rid": int(rid),
                "bid": int(block['id']),
                "x": round(wx - rect_w / 2.0, 4),
                "y": round(wy - rect_h / 2.0, 4),
                "w": round(rect_w, 4),
                "h": round(rect_h, 4)
            })

        rects_by_robot.append(robot_rects)

    return {
        "weights": [float(w) for w in weights_arr],
        "rects": rects_by_robot
    }


# =========================================================================================
# 🚀 暴露给机器人的公共 API 
# =========================================================================================
def generate_target_points(robot_id: int, weights: list, current_pos: tuple = (0.0, 0.0)) -> list:
    weights_arr = np.array(weights)
    robot_num = len(weights_arr)
    
    if not (0 <= robot_id < robot_num):
        raise ValueError(f"提供的机器人编号有误: {robot_id}，根据权重推断系统共 {robot_num} 台机器人。")
        
    all_goals = _compute_all_robot_goals(robot_num, weights_arr, robot_id, current_pos)
        
    return all_goals[robot_id]


# =========================================================================================
# 📝 测试用例
# =========================================================================================
if __name__ == '__main__':
    print("🚗 开始多机器人目标点分配测试...")
    test_weights = [0.75, 0.25]
    test_robot_num = len(test_weights)
    mock_current_positions = [(-1.5, -1.5), (-1.0, -1.0)]
    
    for i in range(test_robot_num):
        mock_pos = mock_current_positions[i]
        my_target_points = generate_target_points(i, test_weights, mock_pos)
        
        if not my_target_points:
            print(f"\n⚠️ 机器人 {i} 未分配到有效目标点。")
            continue
            
        print(f"\n✅ 机器人 {i} 分配到目标点数量：{len(my_target_points)}")
        start_pt = (my_target_points[0][0], my_target_points[0][1])
        end_pt = (my_target_points[-1][0], my_target_points[-1][1])
        print(f"   传入的当前坐标: {mock_pos} | 生成的起点坐标: {start_pt}")
        print(f"   起点坐标: {start_pt} | 终点坐标: {end_pt} | 闭环是否重合: {start_pt == end_pt}")
        print(f"   路径点序列前3个: {my_target_points[:3]} ... (省略中间) ... 结尾: {my_target_points[-1:]}")
        print("\n" + "-"*60)
        sleep(2)  # 模拟处理间隔