import yaml
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import math
import collections
import datetime

# -------------------------- 配置项 --------------------------
yaml_path = "/home/zzh/ros2bookcode-master/chapt7/chapt7_ws/src/fishbot_navigation2/maps/room.yaml"
SAFETY_RADIUS_M = 0
BASE_SIZE = 10    # 基础网格对应的像素数（用于计算网格物理尺寸）
CELL_SIZE = 3     # 砖块缩放倍数（基础砖块尺寸×CELL_SIZE=实际网格数）

# -------------------------- 解析地图元数据（仅取关键参数） --------------------------
def load_map_core_params(yaml_file):
    with open(yaml_file, 'r') as f:
        metadata = yaml.safe_load(f)
    
    # 核心参数
    resolution = metadata['resolution']  # 每个像素的物理尺寸（米/像素）
    origin = metadata['origin']          # 地图原点（x,y,yaw）
    free_thresh = metadata['free_thresh']
    
    # 解析PGM尺寸（仅用于构建网格地图，不参与坐标转换）
    yaml_dir = os.path.dirname(yaml_file)
    pgm_path = os.path.join(yaml_dir, metadata['image'])
    if not os.path.exists(pgm_path):
        raise FileNotFoundError(f"PGM文件不存在：{pgm_path}")
    
    with open(pgm_path, 'rb') as f:
        f.readline()  # 跳过magic number
        while True:
            line = f.readline().strip()
            if not line.startswith(b'#'):
                map_w_px, map_h_px = map(int, line.split())
                break
    
    # 加载PGM图像（仅用于障碍物判定和底图显示）
    map_img = np.array(Image.open(pgm_path).convert('L'))
    
    # 计算核心：网格物理尺寸（1个基础网格 = 多少米）
    grid_reso = BASE_SIZE * resolution  # 关键！网格→世界的直接转换系数
    
    # 计算网格总数（向下取整）
    grid_h = map_h_px // BASE_SIZE  # 网格行数（Y方向）
    grid_w = map_w_px // BASE_SIZE  # 网格列数（X方向）
    
    return {
        'resolution': resolution,
        'origin': origin,
        'free_thresh': free_thresh,
        'map_img': map_img,
        'map_w_px': map_w_px,
        'map_h_px': map_h_px,
        'grid_reso': grid_reso,       # 1网格 = grid_reso 米
        'grid_h': grid_h,             # 网格总行数
        'grid_w': grid_w              # 网格总列数
    }

# -------------------------- 构建障碍物网格（仅用一次像素判定） --------------------------
def build_obstacle_grid(map_params):
    grid_h = map_params['grid_h']
    grid_w = map_params['grid_w']
    map_img = map_params['map_img']
    free_thresh = map_params['free_thresh']
    BASE_SIZE = globals()['BASE_SIZE']
    pixel_safe = int(SAFETY_RADIUS_M / map_params['resolution'])
    
    # 初始化网格：0=无障碍物，1=有障碍物
    obstacle_grid = np.zeros((grid_h, grid_w), dtype=int)
    
    # 仅这一步用到像素（判定障碍物），之后全程用网格数
    for r in range(grid_h):
        for c in range(grid_w):
            # 像素范围（用于障碍物判定）
            sr = max(0, r*BASE_SIZE - pixel_safe)
            er = min(map_params['map_h_px'], (r+1)*BASE_SIZE + pixel_safe)
            sc = max(0, c*BASE_SIZE - pixel_safe)
            ec = min(map_params['map_w_px'], (c+1)*BASE_SIZE + pixel_safe)
            
            # 判定是否有障碍物
            if np.any((map_img[sr:er, sc:ec]/255.0) <= free_thresh):
                obstacle_grid[r, c] = 1
    return obstacle_grid

# -------------------------- 铺砖算法类（全程基于网格数） --------------------------
class GridFiller:
    def __init__(self, obstacle_grid, cell_size):
        self.grid_h, self.grid_w = obstacle_grid.shape
        self.obstacle_grid = obstacle_grid.copy()
        self.cell_size = cell_size
        
        # 砖块定义：基础网格数（不再关联像素）
        self.block_defs = {
            2: (2, 2, '#FF6B6B', '2x2'),  # 基础2×2网格
            3: (2, 1, '#4ECDC4', '2x1'),  # 基础2×1网格
            4: (1, 2, '#45B7D1', '1x2'),  # 基础1×2网格
            5: (1, 1, '#96CEB4', '1x1')   # 基础1×1网格
        }
        self.placed_blocks = []
        self.block_counts = {2:0,3:0,4:0,5:0}
    
    def _get_traversal_order(self, direction, block_h, block_w):
        max_r = self.grid_h - block_h
        max_c = self.grid_w - block_w
        r_range = range(max_r + 1)
        c_range = range(max_c + 1)
        
        if direction == 'top_left_to_bottom_right':
            return [(r, c) for r in r_range for c in c_range]
        elif direction == 'bottom_right_to_top_left':
            return [(r, c) for r in reversed(r_range) for c in reversed(c_range)]
        elif direction == 'bottom_left_to_top_right':
            return [(r, c) for r in reversed(r_range) for c in c_range]
        else:
            return [(r, c) for r in r_range for c in reversed(c_range)]

    def _fill_direction(self, direction):
        grid = self.obstacle_grid.copy()
        placed = []
        counts = {2:0,3:0,4:0,5:0}
        
        # 按优先级填充
        for bid in [2,3,4,5]:
            bh_base, bw_base, _, _ = self.block_defs[bid]
            bh = bh_base * self.cell_size  # 实际网格高度
            bw = bw_base * self.cell_size  # 实际网格宽度
            
            for r, c in self._get_traversal_order(direction, bh, bw):
                if r+bh > self.grid_h or c+bw > self.grid_w:
                    continue
                # 判定该区域是否可填充
                if all(grid[r+i][c+j] == 0 for i in range(bh) for j in range(bw)):
                    # 标记已填充
                    for i in range(bh):
                        for j in range(bw):
                            grid[r+i][c+j] = bid
                    counts[bid] += 1
                    placed.append({
                        'r': r, 'c': c,    # 网格坐标（行、列）
                        'bh': bh, 'bw': bw, # 砖块网格尺寸
                        'id': bid
                    })
        return sum(counts.values()), grid, placed, counts

    def fill_grid(self):
        directions = ['top_left_to_bottom_right', 'bottom_right_to_top_left',
                      'bottom_left_to_top_right', 'top_right_to_bottom_left']
        best_total = float('inf')
        best_result = None
        
        for d in directions:
            total, grid, placed, counts = self._fill_direction(d)
            if total < best_total:
                best_total = total
                best_result = {
                    'direction': d,
                    'grid': grid,
                    'placed_blocks': placed,
                    'counts': counts
                }
        
        self.placed_blocks = best_result['placed_blocks']
        self.block_counts = best_result['counts']
        print(f"最优填充方向：{best_result['direction']}，总砖块数：{best_total}")

    def calculate_mst(self, world_pts):
        if len(world_pts) <= 1:
            return []
        # 生成所有边并排序
        edges = []
        for i, p1 in enumerate(world_pts):
            for j, p2 in enumerate(world_pts):
                if i < j:
                    dist = math.hypot(p1[0]-p2[0], p1[1]-p2[1])
                    edges.append((dist, i, j))
        edges.sort()
        
        # Kruskal算法
        parent = list(range(len(world_pts)))
        def find(u):
            if parent[u] != u:
                parent[u] = find(parent[u])
            return parent[u]
        
        mst_edges = []
        for dist, u, v in edges:
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv
                mst_edges.append((world_pts[u], world_pts[v]))
                if len(mst_edges) == len(world_pts)-1:
                    break
        return mst_edges

# -------------------------- 主流程（纯网格数计算世界坐标） --------------------------
def generate_target_points(image_dir='/home/zzh/ros2bookcode-master/chapt7/chapt7_ws/src/autopatrol_robot/tempphoto'):
    # 1. 加载地图核心参数
    map_params = load_map_core_params(yaml_path)
    ox, oy, _ = map_params['origin']       # 地图原点（世界坐标）
    grid_reso = map_params['grid_reso']    # 1网格 = grid_reso 米（核心转换系数）
    grid_h = map_params['grid_h']
    grid_w = map_params['grid_w']
    
    # 2. 构建障碍物网格（仅这一步用到像素）
    obstacle_grid = build_obstacle_grid(map_params)
    
    # 3. 铺砖
    filler = GridFiller(obstacle_grid, CELL_SIZE)
    filler.fill_grid()
    
    # 4. 网格坐标 → 世界坐标（核心：纯网格数计算，无像素中转）
    world_points = []
    for block in filler.placed_blocks:
        r, c = block['r'], block['c']
        bh, bw = block['bh'], block['bw']
        
        # 计算砖块中心的网格坐标
        center_c = c + bw / 2.0  # 列中心（X方向）
        center_r = r + bh / 2.0  # 行中心（Y方向）
        
        # 直接转世界坐标（和网格线同源）
        # 关键：Y轴翻转（网格r=0对应图像顶部，世界坐标y轴向上）
        wx = ox + center_c * grid_reso
        wy = oy + (grid_h - center_r) * grid_reso
        world_points.append((wx, wy))
    
    if not world_points:
        print("⚠️  未找到可铺砖的区域！")
        return []
    
    # 5. MST生成遍历序列
    mst_edges = filler.calculate_mst(world_points)
    # 构建邻接表
    adj = collections.defaultdict(list)
    for p1, p2 in mst_edges:
        adj[p1].append(p2)
        adj[p2].append(p1)
    # DFS生成遍历序列
    sequence = []
    visited = set()
    def dfs(curr_pt):
        visited.add(curr_pt)
        sequence.append(curr_pt)
        for neighbor in adj[curr_pt]:
            if neighbor not in visited:
                dfs(neighbor)
    dfs(world_points[0])

    # 6. 生成带偏航角的目标点（保留两位小数）
    goal_points = []
    for i in range(len(sequence)):
        x, y = sequence[i]
        if i < len(sequence)-1:
            nx, ny = sequence[i+1]
            yaw = math.atan2(ny - y, nx - x)
        else:
            yaw = goal_points[-1][2] if goal_points else 0.0
        
        # 核心修改：保留两位小数
        x_rounded = round(x, 2)
        y_rounded = round(y, 2)
        yaw_rounded = round(yaw, 2)
        
        goal_points.append((x_rounded, y_rounded, yaw_rounded))
    
    # -------------------------- 可视化（纯网格数驱动） --------------------------
    plt.figure(figsize=(14, 12))
    # 底图（仅用于显示，不参与坐标计算）
    plt.imshow(map_params['map_img'], cmap='gray', extent=[
        ox, ox + map_params['map_w_px']*map_params['resolution'],
        oy, oy + map_params['map_h_px']*map_params['resolution']
    ])
    
    # 1. 绘制网格线（纯网格数计算，和坐标同源）
    # X方向网格线（列）
    for c in range(grid_w + 1):
        x = ox + c * grid_reso
        plt.axvline(x=x, color='lightgray', linestyle='-', linewidth=1, alpha=0.9)
    # Y方向网格线（行）
    for r in range(grid_h + 1):
        y = oy + (grid_h - r) * grid_reso  # Y轴翻转
        plt.axhline(y=y, color='lightgray', linestyle='-', linewidth=1, alpha=0.9)
    
    # 2. 网格中心标注0/1（障碍物状态）
    for r in range(grid_h):
        for c in range(grid_w):
            # 网格中心世界坐标
            wx = ox + (c + 0.5) * grid_reso
            wy = oy + (grid_h - r - 0.5) * grid_reso
            # 标注
            plt.text(wx, wy, str(obstacle_grid[r][c]),
                     ha='center', va='center', fontsize=7,
                     color='black', weight='bold')
    
    # 3. 绘制砖块（纯网格数计算）
    for block in filler.placed_blocks:
        r, c = block['r'], block['c']
        bh, bw = block['bh'], block['bw']
        bid = block['id']
        color = filler.block_defs[bid][2]
        
        # 砖块左下角世界坐标（和网格线对齐）
        rect_x = ox + c * grid_reso
        rect_y = oy + (grid_h - (r + bh)) * grid_reso
        # 砖块物理尺寸
        rect_w = bw * grid_reso
        rect_h = bh * grid_reso
        
        # 绘制砖块
        rect = patches.Rectangle((rect_x, rect_y), rect_w, rect_h,
                                 linewidth=1.2, edgecolor='black',
                                 facecolor=color, alpha=0.6)
        plt.gca().add_patch(rect)
    
    # 4. 绘制MST和中心点
    for i, (p1, p2) in enumerate(mst_edges):
        label = 'MST Edges' if i == 0 else ""
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]],
                 'b-', linewidth=2, alpha=0.7, label=label)
    # 中心点
    seq_x = [p[0] for p in sequence]
    seq_y = [p[1] for p in sequence]
    plt.scatter(seq_x, seq_y, c='yellow', s=30, edgecolors='black',
                zorder=4, label='lock Centers')
    # 起点
    plt.scatter(seq_x[0], seq_y[0], c='red', s=150, marker='*',
                zorder=5, label='Start Point')
    
    # 图表配置
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title(f'Tiling Coverage Path (BASE_SIZE={BASE_SIZE}, CELL_SIZE={CELL_SIZE}, Total Blocks: {len(world_points)})')
    plt.legend(loc='upper right')
    plt.axis('equal')
    
    # 保存图像
    os.makedirs(image_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(image_dir, f"grid_based_tiled_{timestamp}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 可视化图像已保存：{save_path}")
    
    return goal_points

# -------------------------- 执行 --------------------------
if __name__ == '__main__':
    targets = generate_target_points()
    print(targets)
    print(f"✅ 生成目标点数量：{len(targets)}")