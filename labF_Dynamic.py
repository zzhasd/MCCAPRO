import datetime
import importlib.util
import os
from pathlib import Path
import sys
import warnings
from collections import deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.spatial import Voronoi, voronoi_plot_2d

warnings.filterwarnings('ignore')

# 移除中文字体，使用 matplotlib 默认的学术英文字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False


def _load_tile_first_planner():
    """Load the uploaded FACT-MCCA / TileFirstMCPP implementation as an external planner."""
    base_dir = Path(__file__).resolve().parent
    candidates = [
        base_dir / 'mainline_tile_first_v2_3_2.py',
        base_dir / 'mainline_tile_first_v2_3_2(20260825-081249).py',
    ]
    candidates.extend(sorted(base_dir.glob('mainline_tile_first_v2_3_2*.py')))

    planner_file = next((p for p in candidates if p.exists() and p.resolve() != Path(__file__).resolve()), None)
    if planner_file is None:
        raise FileNotFoundError(
            '找不到 mainline_tile_first_v2_3_2 算法文件。请将 labF_Dynamic.py 与 '
            'mainline_tile_first_v2_3_2.py（或上传的带时间戳版本）放在同一目录。'
        )

    module_name = '_labf_tile_first_v2_3_2'
    spec = importlib.util.spec_from_file_location(module_name, planner_file)
    if spec is None or spec.loader is None:
        raise ImportError(f'无法加载路径规划算法文件: {planner_file}')

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.TileFirstMCPP


TileFirstMCPP = _load_tile_first_planner()


class DynamicMCPP_Weighted:
    """Adapter that keeps the Lab-F dynamic experiment flow but delegates planning to TileFirstMCPP."""

    def __init__(self, map_size, robot_num, obstacle_ratio=0.1):
        self.map_size = map_size
        self.robot_num = robot_num
        self.obstacle_num = int(map_size * map_size * obstacle_ratio)
        self.total_grids = map_size * map_size

        # Initial equal weights 1:1:1
        self.weights = np.ones(robot_num) / robot_num

        # 保留原实验的随机地图生成逻辑
        self.generate_map()
        self._prepare_planner_obstacle_mask()

        self.centroids_xy = []
        self.centroids = np.empty((0, 2), dtype=float)
        self.robot_areas = np.zeros(self.robot_num)
        self.full_assignments = np.full(self.total_grids, -1, dtype=int)

        # 新算法输出结构：铺砖 + 实际闭合路径（不再使用 MST）
        self.all_tiles = []
        self.tile_stats = {}
        self.robot_path_dict = {}
        self.planner = None
        self.plan_result = None

    def generate_map(self):
        self.idx_to_xy = {}
        self.xy_to_idx = {}
        idx = 0
        for x in range(self.map_size):
            for y in range(self.map_size):
                self.idx_to_xy[idx] = (x, y)
                self.xy_to_idx[(x, y)] = idx
                idx += 1

        all_xy = list(self.xy_to_idx.keys())
        obstacle_indices_xy = list(np.random.choice(len(all_xy), self.obstacle_num, replace=False))
        self.obstacle_set = {all_xy[i] for i in obstacle_indices_xy}

        self.is_obstacle = np.zeros(self.total_grids, dtype=bool)
        self.obstacle_mask = np.zeros((self.map_size, self.map_size), dtype=bool)
        for xy in self.obstacle_set:
            self.is_obstacle[self.xy_to_idx[xy]] = True
            self.obstacle_mask[xy[0], xy[1]] = True

        self.free_grid_indices = [idx for idx, val in enumerate(self.is_obstacle) if not val]

    def _prepare_planner_obstacle_mask(self):
        """Keep the original map, but give TileFirstMCPP one connected free-space component.

        The old Lab-F partitioner naturally left unreachable free cells as assignment -2.
        TileFirstMCPP requires connected free space, so only for the planner input we mask
        smaller free-space components. They are restored as -2 in full_assignments.
        """
        free_cells = {(x, y) for x in range(self.map_size) for y in range(self.map_size)
                      if not self.obstacle_mask[x, y]}
        unseen = set(free_cells)
        components = []

        while unseen:
            start = unseen.pop()
            component = {start}
            q = deque([start])
            while q:
                x, y = q.popleft()
                for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                    nb = (x + dx, y + dy)
                    if nb in unseen:
                        unseen.remove(nb)
                        component.add(nb)
                        q.append(nb)
            components.append(component)

        largest_component = max(components, key=len) if components else set()
        self.unreachable_free_set = free_cells - largest_component
        self.planner_obstacle_mask = self.obstacle_mask.copy()
        for x, y in self.unreachable_free_set:
            self.planner_obstacle_mask[x, y] = True

    def partition(self, max_iter=10):
        """Core planner call replacing the previous weighted Voronoi + local MST pipeline."""
        self.planner = TileFirstMCPP(
            map_shape=self.map_size,
            robot_num=self.robot_num,
            obstacle_mask=self.planner_obstacle_mask,
            seed=42,
            partition_iterations=max_iter,
            robot_weights=self.weights,
        )
        self.plan_result = self.planner.solve(method='tile_first')

        # 将新算法的二维 assignment 回写到原实验的一维 full_assignments 结构中
        self.full_assignments = np.full(self.total_grids, -1, dtype=int)
        for x in range(self.map_size):
            for y in range(self.map_size):
                idx = self.xy_to_idx[(x, y)]
                if self.is_obstacle[idx]:
                    self.full_assignments[idx] = -1
                elif (x, y) in self.unreachable_free_set:
                    self.full_assignments[idx] = -2
                else:
                    self.full_assignments[idx] = int(self.planner.assignments[x, y])

        for rid in range(self.robot_num):
            self.robot_areas[rid] = np.count_nonzero(self.planner.assignments == rid)

        # 仅用于保持原快照中的 centroid 显示；不参与新算法的路线规划
        self.centroids_xy = list(self.planner.centroids_xy)
        centroid_points = []
        for cell in self.centroids_xy:
            if cell is None:
                centroid_points.append([np.nan, np.nan])
            else:
                centroid_points.append([cell[0] + 0.5, cell[1] + 0.5])
        self.centroids = np.asarray(centroid_points, dtype=float)

    def generate_tiles_and_paths(self):
        """Expose TileFirstMCPP tiling and routed paths in the Lab-F visualization structures."""
        if self.planner is None or self.plan_result is None:
            raise RuntimeError('请先调用 partition() 完成 TileFirstMCPP 规划。')

        self.all_tiles = list(self.planner.tiles)
        self.tile_stats = {
            rid: {f"{w}x{h}": 0 for w, h in [(4, 4), (4, 2), (2, 4), (2, 2), (2, 1), (1, 2), (1, 1)]}
            for rid in range(self.robot_num)
        }
        for x, y, w, h, rid in self.all_tiles:
            key = f"{w}x{h}"
            if key in self.tile_stats[rid]:
                self.tile_stats[rid][key] += 1

        self.robot_path_dict = {}
        for rid in range(self.robot_num):
            route = self.planner.routes.get(rid, {})
            self.robot_path_dict[rid] = {
                'path': [list(p) for p in route.get('path', [])],
                'centers': [np.asarray(c, dtype=float).tolist() for c in route.get('centers', [])],
                'total_length': float(route.get('length', 0.0)),
            }

    def compute_metrics(self):
        grid_counts = [np.sum(self.full_assignments == i) for i in range(self.robot_num)]
        path_lengths = [self.robot_path_dict[i]['total_length'] for i in range(self.robot_num)]
        return grid_counts, path_lengths

    def visualize(self, step, output_dir, timestamp):
        self.compute_metrics()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        while len(colors) < self.robot_num:
            colors.append(np.random.rand(3,))

        fig, ax = plt.subplots(figsize=(16, 14))

        # 保留原实验的 Voronoi 辅助显示（只用于可视化，不参与路径规划）
        valid_centroids = self.centroids[np.isfinite(self.centroids).all(axis=1)]
        map_bounds = [0, self.map_size, 0, self.map_size]
        boundary_points = np.array([
            [map_bounds[0] - 10, map_bounds[2] - 10], [map_bounds[1] + 10, map_bounds[2] - 10],
            [map_bounds[0] - 10, map_bounds[3] + 10], [map_bounds[1] + 10, map_bounds[3] + 10]
        ])
        try:
            if len(valid_centroids) > 0:
                all_pts = np.vstack([valid_centroids, boundary_points])
                vor = Voronoi(all_pts)
                voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='gray', line_width=1, line_alpha=0.5)
        except Exception:
            pass

        for x in range(self.map_size):
            for y in range(self.map_size):
                idx = self.xy_to_idx[(x, y)]
                if self.is_obstacle[idx]:
                    rect = Rectangle((x, y), 1, 1, facecolor='black', edgecolor='white', linewidth=0.8)
                else:
                    rid = self.full_assignments[idx]
                    if rid >= 0:
                        rect = Rectangle((x, y), 1, 1, facecolor=colors[rid % len(colors)], alpha=0.1,
                                         edgecolor='white', linewidth=0.5)
                    elif rid == -2:
                        rect = Rectangle((x, y), 1, 1, facecolor='dimgray', alpha=0.4,
                                         edgecolor='white', linewidth=0.5, hatch='///')
                    else:
                        rect = Rectangle((x, y), 1, 1, facecolor='lightgray', alpha=0.1,
                                         edgecolor='white', linewidth=0.5)
                ax.add_patch(rect)

        for (x, y, w, h, rid) in self.all_tiles:
            color = colors[rid % len(colors)]
            tile_rect = Rectangle((x, y), w, h, facecolor=color, alpha=0.5,
                                  edgecolor='black', linewidth=1)
            ax.add_patch(tile_rect)

        for i, point in enumerate(self.centroids):
            if not np.isfinite(point).all():
                continue
            cx, cy = point
            color = colors[i % len(colors)]
            ax.scatter(cx, cy, color=color, s=50, marker='o', edgecolors='black', linewidth=1,
                       label=f'Robot {i+1}\nCentroid: {self.centroids_xy[i]}')

        # 绘制 TileFirstMCPP 输出的实际闭合路径，替代原来的 MST 边
        for rid, path_data in self.robot_path_dict.items():
            color = colors[rid % len(colors)]
            path = np.asarray(path_data['path'], dtype=float)
            if len(path) >= 2:
                ax.plot(path[:, 0], path[:, 1], color=color, linewidth=2.5, alpha=1, zorder=5)
            elif len(path) == 1:
                ax.scatter(path[:, 0], path[:, 1], color=color, s=35, zorder=5)

            if path_data['centers']:
                centers = np.asarray(path_data['centers'], dtype=float)
                ax.scatter(centers[:, 0], centers[:, 1], color='white', s=30,
                           edgecolors=color, linewidth=1.5, zorder=6)

        ax.set_xlim(0, self.map_size)
        ax.set_ylim(0, self.map_size)
        ax.grid(True, alpha=0.2)

        ax.set_title(
            f'Step: {step}',
            fontsize=40, pad=5
        )
        ax.figure.subplots_adjust(top=0.95)
        ax.set_aspect('equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        plt.tight_layout()
        save_path = os.path.join(output_dir, f'Snapshot_{timestamp}_Step_{step}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.0)
        print(f"\n📸 快照已保存: {save_path}")
        plt.close()


def run_dynamic_simulation():
    np.random.seed(42)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 修改：直接输出到工作目录下的 labF-dynamic
    output_dir = os.path.join('.', "labF-dynamic")
    os.makedirs(output_dir, exist_ok=True)

    # --- Experiment Configuration ---
    map_size = 100
    robot_num = 3
    max_T = 150
    alpha = 0.1

    solver = DynamicMCPP_Weighted(map_size, robot_num, obstacle_ratio=0.1)

    # Ground Truth Data Generation
    C_true_abs = np.zeros((max_T, robot_num))

    for t in range(max_T):
        C_true_abs[t, 0] = 15.0
        C_true_abs[t, 1] = 35.0
        if t < 50:
            C_true_abs[t, 2] = 50.0
        elif 50 <= t < 100:
            C_true_abs[t, 2] = 15.0
        else:
            C_true_abs[t, 2] = 50.0

    # 用于收集每步的数据，输出CSV
    experiment_data = []

    print(f"🚀 Starting Dynamic Adaptive Simulation | Damping Factor α={alpha} | Time Steps={max_T}")
    print(f"📁 结果将输出至目录: {output_dir}")

    for t in range(max_T):
        # ✅ 这里添加了单行进度条，\r 回车符会让它在同一行不断覆盖刷新
        print(f"\r⏳ 正在计算 Step {t+1}/{max_T} ...", end="", flush=True)

        # 1. Add Gaussian noise to simulate sensor readings
        noise = np.random.normal(0, 2.0, robot_num)
        C_sampled_abs = np.clip(C_true_abs[t] + noise, 1.0, 100.0)

        # 2. Ratios
        C_sampled_ratio = C_sampled_abs / np.sum(C_sampled_abs)

        # 3. EMA Filter
        solver.weights = alpha * C_sampled_ratio + (1 - alpha) * solver.weights
        solver.weights /= np.sum(solver.weights)

        # 4. 调用 TileFirstMCPP 完成多机器人铺砖、分区和路径规划
        solver.partition(max_iter=5)

        # 4.1 记录分区面积占比
        total_assigned_area = np.sum(solver.robot_areas)
        area_ratios = solver.robot_areas / (total_assigned_area + 1e-9)

        # 5. 同步新规划器的铺砖与实际路径结果
        solver.generate_tiles_and_paths()

        # 6. 记录 Path Length（替代原 MST Length）
        path_lengths = [solver.robot_path_dict[i]['total_length'] for i in range(robot_num)]

        # 收集此步数据
        experiment_data.append({
            'Step': t,
            'Sensed_C_0': C_sampled_abs[0], 'Sensed_C_1': C_sampled_abs[1], 'Sensed_C_2': C_sampled_abs[2],
            'Weight_0': solver.weights[0], 'Weight_1': solver.weights[1], 'Weight_2': solver.weights[2],
            'Path_Length_0': path_lengths[0], 'Path_Length_1': path_lengths[1], 'Path_Length_2': path_lengths[2],
            'Area_Ratio_0': area_ratios[0], 'Area_Ratio_1': area_ratios[1], 'Area_Ratio_2': area_ratios[2]
        })

        # 7. 在指定时间步截取快照
        if t in [45, 60, 140]:
            solver.visualize(step=t, output_dir=output_dir, timestamp=timestamp)
            # ✅ 为了防止被 "\r" 覆盖，截取快照后加一个空 print 换行
            print()

    # 循环结束后，将数据保存为 CSV
    df = pd.DataFrame(experiment_data)
    csv_filename = f"experiment_EMA_Simulation_{timestamp}.csv"
    csv_filepath = os.path.join(output_dir, csv_filename)
    df.to_csv(csv_filepath, index=False)

    print(f"\n✅ 实验运行完毕！实验数据已保存至: {csv_filepath}")


if __name__ == '__main__':
    run_dynamic_simulation()
