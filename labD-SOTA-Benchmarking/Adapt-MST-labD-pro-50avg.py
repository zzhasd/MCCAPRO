import random
import numpy as np
import math
import os
import time
from datetime import datetime
import concurrent.futures

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

class GridFiller:
    def __init__(self, grid_size=10, obstacle_num=None, block_types=None, obstacle_positions=None, verbose=True):
        self.grid_size = grid_size
        self.verbose = verbose
        # 初始化网格：0表示空，1表示障碍物，>=2表示已填充
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        if obstacle_num is None:
            obstacle_num = max(1, int(self.grid_size * self.grid_size * 0.1))
        # 生成障碍物
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
            
        self.block_counts = {block_id: 0 for block_id in self.block_types}
        self.placed_blocks = []

    def _get_block_fill_order(self):
        return sorted(
            self.block_types.items(),
            key=lambda item: item[1][0] * item[1][1],
            reverse=True
        )

    def _is_area_available_on_grid(self, grid, x, y, h, w):
        if x + h > self.grid_size or y + w > self.grid_size:
            return False
        for i in range(h):
            for j in range(w):
                if grid[x+i][y+j] != 0:
                    return False
        return True

    def _fill_area_on_grid(self, grid, x, y, h, w, block_id):
        for i in range(h):
            for j in range(w):
                grid[x+i][y+j] = block_id

    def _apply_obstacles(self, obstacles):
        for x, y in obstacles:
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                self.grid[x][y] = 1
        
    def _generate_obstacles(self, obstacle_num):
        max_obstacles = self.grid_size * self.grid_size
        obstacle_num = min(obstacle_num, max_obstacles - 1)
        obstacles = set()
        while len(obstacles) < obstacle_num:
            x = random.randint(0, self.grid_size-1)
            y = random.randint(0, self.grid_size-1)
            obstacles.add((x, y))
        for x, y in obstacles:
            self.grid[x][y] = 1
    
    def _check_symmetry(self):
        obstacles = np.argwhere(self.grid == 1)
        obstacle_set = set((x, y) for x, y in obstacles)
        grid_size = self.grid_size
        
        main_diag_sym = True
        for x, y in obstacles:
            if (y, x) not in obstacle_set:
                main_diag_sym = False
                break
        
        anti_diag_sym = True
        for x, y in obstacles:
            mirror_x = grid_size - 1 - y
            mirror_y = grid_size - 1 - x
            if (mirror_x, mirror_y) not in obstacle_set:
                anti_diag_sym = False
                break
        
        if main_diag_sym and anti_diag_sym:
            return ['top_left_to_bottom_right']
        elif main_diag_sym:
            return ['top_left_to_bottom_right', 'top_right_to_bottom_left']
        elif anti_diag_sym:
            return ['top_left_to_bottom_right', 'bottom_left_to_top_right']
        else:
            return [
                'top_left_to_bottom_right',
                'bottom_right_to_top_left',
                'bottom_left_to_top_right',
                'top_right_to_bottom_left'
            ]
    
    def _get_traversal_order(self, direction, block_h, block_w):
        max_x = self.grid_size - block_h
        max_y = self.grid_size - block_w
        x_range = range(max_x + 1)
        y_range = range(max_y + 1)
        
        if direction == 'top_left_to_bottom_right':
            return [(x, y) for x in x_range for y in y_range]
        elif direction == 'bottom_right_to_top_left':
            return [(x, y) for x in reversed(x_range) for y in reversed(y_range)]
        elif direction == 'bottom_left_to_top_right':
            return [(x, y) for x in reversed(x_range) for y in y_range]
        elif direction == 'top_right_to_bottom_left':
            return [(x, y) for x in x_range for y in reversed(y_range)]
    
    def _fill_with_direction(self, direction):
        grid = self.original_grid.copy()
        placed_blocks = []
        block_counts = {block_id: 0 for block_id in self.block_types}

        for block_id, (h, w, _, _) in self._get_block_fill_order():
            for x, y in self._get_traversal_order(direction, h, w):
                if self._is_area_available_on_grid(grid, x, y, h, w):
                    self._fill_area_on_grid(grid, x, y, h, w, block_id)
                    block_counts[block_id] += 1
                    placed_blocks.append({'x': x, 'y': y, 'h': h, 'w': w, 'id': block_id})
        
        total_blocks = sum(block_counts.values())
        return total_blocks, grid, placed_blocks, block_counts
    
    def fill_grid(self):
        candidate_directions = self._check_symmetry()
        best_result = None
        min_total_blocks = float('inf')
        
        for direction in candidate_directions:
            total_blocks, grid, placed_blocks, block_counts = self._fill_with_direction(direction)
            if total_blocks < min_total_blocks:
                min_total_blocks = total_blocks
                best_result = {
                    'grid': grid,
                    'placed_blocks': placed_blocks,
                    'block_counts': block_counts,
                    'direction': direction
                }
        
        self.grid = best_result['grid']
        self.placed_blocks = best_result['placed_blocks']
        self.block_counts = best_result['block_counts']
    
    def _get_block_center(self, block):
        return (block['y'] + block['w'] / 2.0, block['x'] + block['h'] / 2.0)

    def _blocks_are_edge_adjacent(self, block_a, block_b):
        ax1, ay1 = block_a['x'], block_a['y']
        ax2, ay2 = ax1 + block_a['h'], ay1 + block_a['w']
        bx1, by1 = block_b['x'], block_b['y']
        bx2, by2 = bx1 + block_b['h'], by1 + block_b['w']

        vertical_touch = (ay2 == by1 or by2 == ay1) and (max(ax1, bx1) < min(ax2, bx2))
        horizontal_touch = (ax2 == bx1 or bx2 == ax1) and (max(ay1, by1) < min(ay2, by2))
        return vertical_touch or horizontal_touch

    def calculate_mst(self, blocks):
        n = len(blocks)
        total_length = 0.0
        if n <= 1:
            return [], total_length

        centers = [self._get_block_center(block) for block in blocks]
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if self._blocks_are_edge_adjacent(blocks[i], blocks[j]):
                    p1 = centers[i]
                    p2 = centers[j]
                    dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                    edges.append((dist, i, j))
        edges.sort()

        parent = list(range(n))

        def find(i):
            if parent[i] != i:
                parent[i] = find(parent[i])
            return parent[i]

        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j
                return True
            return False

        mst_edges = []
        edges_count = 0
        for dist, u, v in edges:
            if union(u, v):
                mst_edges.append((centers[u], centers[v]))
                total_length += dist
                edges_count += 1
                if edges_count == n - 1:
                    break
        return mst_edges, total_length

def generate_obstacles_for_size(grid_size, obstacle_num, seed):
    rng = random.Random(seed)
    max_obstacles = grid_size * grid_size - 1
    obstacle_num = max(1, min(obstacle_num, max_obstacles))
    obstacles = set()
    while len(obstacles) < obstacle_num:
        obstacles.add((rng.randint(0, grid_size - 1), rng.randint(0, grid_size - 1)))
    return obstacles

# 定义被多进程调用的单次实验任务
def _run_single_experiment(args):
    grid_size, obstacle_ratio, seed = args
    obstacle_num = max(1, int(grid_size * grid_size * obstacle_ratio))
    
    # 局部生成障碍物
    obstacles = generate_obstacles_for_size(grid_size, obstacle_num, seed)

    # 方案 A
    filler_a = GridFiller(
        grid_size=grid_size, obstacle_num=obstacle_num, 
        block_types=SCHEME_FULL, obstacle_positions=obstacles, verbose=False
    )
    start_a = time.perf_counter()
    filler_a.fill_grid()
    blocks_a = sum(filler_a.block_counts.values())
    mst_a = filler_a.calculate_mst(filler_a.placed_blocks)[1]
    time_a = time.perf_counter() - start_a

    # 方案 B
    filler_b = GridFiller(
        grid_size=grid_size, obstacle_num=obstacle_num, 
        block_types=SCHEME_SIMPLE, obstacle_positions=obstacles, verbose=False
    )
    start_b = time.perf_counter()
    filler_b.fill_grid()
    blocks_b = sum(filler_b.block_counts.values())
    mst_b = filler_b.calculate_mst(filler_b.placed_blocks)[1]
    time_b = time.perf_counter() - start_b

    # 传回所有统计数据
    return (blocks_a, mst_a, time_a, blocks_b, mst_b, time_b)


def run_experiment_for_ratio(min_size, max_size, step, obstacle_ratio, seeds_list):
    sizes = list(range(min_size, max_size + 1, step))
    lines = ["grid_size,A_blocks,A_mst,A_time,B_blocks,B_mst,B_time"]
    num_experiments = len(seeds_list)

    # 使用最大可能的 CPU 核心数，留一个核心防止系统卡顿
    max_workers = max(1, os.cpu_count() - 1)

    for grid_size in sizes:
        sum_blocks_a, sum_mst_a, sum_time_a = 0.0, 0.0, 0.0
        sum_blocks_b, sum_mst_b, sum_time_b = 0.0, 0.0, 0.0

        # 打包任务参数
        tasks = [(grid_size, obstacle_ratio, seed) for seed in seeds_list]

        # 启动多进程池
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # executor.map 会自动将任务分发给多个进程并行执行，并按顺序返回结果
            results = list(executor.map(_run_single_experiment, tasks))

        # 聚合该 grid_size 下所有进程返回的 50 次实验结果
        for res in results:
            b_a, m_a, t_a, b_b, m_b, t_b = res
            sum_blocks_a += b_a
            sum_mst_a += m_a
            sum_time_a += t_a
            sum_blocks_b += b_b
            sum_mst_b += m_b
            sum_time_b += t_b

        # 计算平均值
        avg_blocks_a = sum_blocks_a / num_experiments
        avg_mst_a = sum_mst_a / num_experiments
        avg_time_a = sum_time_a / num_experiments
        
        avg_blocks_b = sum_blocks_b / num_experiments
        avg_mst_b = sum_mst_b / num_experiments
        avg_time_b = sum_time_b / num_experiments

        print(
            f"密度 {int(obstacle_ratio*100)}% | 网格 {grid_size}x{grid_size} (基于50个种子平均): "
            f"A(铺砖={avg_blocks_a:.2f}, MST={avg_mst_a:.2f}, 时间={avg_time_a:.4f}s) | "
            f"B(铺砖={avg_blocks_b:.2f}, MST={avg_mst_b:.2f}, 时间={avg_time_b:.4f}s)"
        )
        
        lines.append(f"{grid_size},{avg_blocks_a:.6f},{avg_mst_a:.6f},{avg_time_a:.6f},{avg_blocks_b:.6f},{avg_mst_b:.6f},{avg_time_b:.6f}")

    return "\n".join(lines)

# 主程序
if __name__ == "__main__":
    # 多进程在 Windows 环境下必须在这个保护块内执行
    # 参数设置
    min_size = 20
    max_size = 200
    step = 10
    
    # 动态获取当前脚本所在目录作为输出路径
    output_dir = os.path.dirname(os.path.abspath(__file__))
    if output_dir == "": 
        output_dir = "."
        
    # 生成时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_path = os.path.join(output_dir, f'experiment_data_{timestamp}.txt')
    
    # 自动循环运行 10%, 15%, 20%
    ratios = [0.10, 0.15, 0.20]
    all_experiments_output = []
    
    for ratio in ratios:
        ratio_int = int(ratio * 100)
        print(f"\n================ 开始运行障碍物密度 {ratio_int}% 的实验 (使用指定的50个种子) ================")
        
        # 传入 EXPERIMENT_SEEDS 列表执行实验
        start_ratio_time = time.perf_counter()
        data_str = run_experiment_for_ratio(min_size, max_size, step, ratio, EXPERIMENT_SEEDS)
        end_ratio_time = time.perf_counter()
        
        print(f"[{ratio_int}% 实验完成] 耗时: {end_ratio_time - start_ratio_time:.2f} 秒")
        
        # 拼接变量名格式
        var_name = f"data_{ratio_int}_str"
        final_output = f'{var_name} = """\n{data_str}\n"""'
        all_experiments_output.append(final_output)
        
    # 保存所有结果到同一个 txt 文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(all_experiments_output))
        
    print(f"\n>> 所有实验完成！平均值数据已成功保存至: {file_path}")