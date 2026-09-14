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

# Use the default matplotlib English font instead of a CJK font
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
            'Cannot find the mainline_tile_first_v2_3_2 algorithm file. Place labF_Dynamic.py and '
            'mainline_tile_first_v2_3_2.py(or the uploaded timestamped version) in the same directory.'
        )

    module_name = '_labf_tile_first_v2_3_2'
    spec = importlib.util.spec_from_file_location(module_name, planner_file)
    if spec is None or spec.loader is None:
        raise ImportError(f'Unable to load the path-planning algorithm file: {planner_file}')

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

        # Preserve the original random-map generation logic
        self.generate_map()
        self._prepare_planner_obstacle_mask()

        self.centroids_xy = []
        self.centroids = np.empty((0, 2), dtype=float)
        self.robot_areas = np.zeros(self.robot_num)
        self.full_assignments = np.full(self.total_grids, -1, dtype=int)

        # New algorithm output: tiling + actual closed routes (no longer using MST)
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

        # Copy the new algorithm's 2D assignment back into the original 1D full_assignments structure
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

        # Only preserve centroid display in existing snapshots; not used for route planning
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
            raise RuntimeError('Call partition() first to complete TileFirstMCPP planning.')

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

        # Preserve the original Voronoi overlay (visualization only; not used for path planning)
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

        # Draw actual closed routes from TileFirstMCPP instead of the original MST edges
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
        print(f"\n📸 Snapshot saved: {save_path}")
        plt.close()


def run_dynamic_simulation():
    np.random.seed(42)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Change: write directly under the working directory to labF-dynamic
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

    # Collect data from each step and output CSV
    experiment_data = []

    print(f"🚀 Starting Dynamic Adaptive Simulation | Damping Factor α={alpha} | Time Steps={max_T}")
    print(f"📁 Results will be written to: {output_dir}")

    for t in range(max_T):
        # ✅ Use a single-line progress bar; the \r carriage return updates the same line
        print(f"\r⏳ Computing Step {t+1}/{max_T} ...", end="", flush=True)

        # 1. Add Gaussian noise to simulate sensor readings
        noise = np.random.normal(0, 2.0, robot_num)
        C_sampled_abs = np.clip(C_true_abs[t] + noise, 1.0, 100.0)

        # 2. Ratios
        C_sampled_ratio = C_sampled_abs / np.sum(C_sampled_abs)

        # 3. EMA Filter
        solver.weights = alpha * C_sampled_ratio + (1 - alpha) * solver.weights
        solver.weights /= np.sum(solver.weights)

        # 4. Call TileFirstMCPP for multi-robot tiling, partitioning, and path planning
        solver.partition(max_iter=5)

        # 4.1 Record partition area fractions
        total_assigned_area = np.sum(solver.robot_areas)
        area_ratios = solver.robot_areas / (total_assigned_area + 1e-9)

        # 5. Synchronize tiling and actual route results from the new planner
        solver.generate_tiles_and_paths()

        # 6. Record Path Length (replacing the original MST Length)
        path_lengths = [solver.robot_path_dict[i]['total_length'] for i in range(robot_num)]

        # Collect data for this step
        experiment_data.append({
            'Step': t,
            'Sensed_C_0': C_sampled_abs[0], 'Sensed_C_1': C_sampled_abs[1], 'Sensed_C_2': C_sampled_abs[2],
            'Weight_0': solver.weights[0], 'Weight_1': solver.weights[1], 'Weight_2': solver.weights[2],
            'Path_Length_0': path_lengths[0], 'Path_Length_1': path_lengths[1], 'Path_Length_2': path_lengths[2],
            'Area_Ratio_0': area_ratios[0], 'Area_Ratio_1': area_ratios[1], 'Area_Ratio_2': area_ratios[2]
        })

        # 7. Capture snapshots at the specified timesteps
        if t in [45, 60, 140]:
            solver.visualize(step=t, output_dir=output_dir, timestamp=timestamp)
            # ✅ Print a blank line after each snapshot to prevent overwriting by "\r"
            print()

    # After the loop, save data as CSV
    df = pd.DataFrame(experiment_data)
    csv_filename = f"experiment_EMA_Simulation_{timestamp}.csv"
    csv_filepath = os.path.join(output_dir, csv_filename)
    df.to_csv(csv_filepath, index=False)

    print(f"\n✅ Experiment complete! Data saved to: {csv_filepath}")


if __name__ == '__main__':
    run_dynamic_simulation()
