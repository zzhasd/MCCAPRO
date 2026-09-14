import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra, minimum_spanning_tree
import os
import warnings
import pandas as pd
from scipy.spatial import Voronoi, voronoi_plot_2d

warnings.filterwarnings('ignore')

# Use the default matplotlib English font instead of a CJK font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

class DynamicMCPP_Weighted:
    def __init__(self, map_size, robot_num, obstacle_ratio=0.1):
        self.map_size = map_size
        self.robot_num = robot_num
        self.obstacle_num = int(map_size * map_size * obstacle_ratio)
        self.total_grids = map_size * map_size
        
        # Initial equal weights 1:1:1
        self.weights = np.ones(robot_num) / robot_num
        
        self.generate_map()
        self.build_sparse_graph()
        
        self.centroids_xy = []
        self.centroids = []
        self.robot_areas = np.zeros(self.robot_num)
        self.full_assignments = np.full(self.total_grids, -1, dtype=int)
        
        # Initialize structures required for tiling and MST
        self.all_tiles = []
        self.tile_stats = {}
        self.robot_mst_dict = {}

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
        for xy in self.obstacle_set:
            self.is_obstacle[self.xy_to_idx[xy]] = True
            
        self.free_grid_indices = [idx for idx, val in enumerate(self.is_obstacle) if not val]

    def build_sparse_graph(self):
        row, col, data = [], [], []
        for x in range(self.map_size):
            for y in range(self.map_size):
                idx = self.xy_to_idx[(x, y)]
                if self.is_obstacle[idx]: continue
                
                for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                    nx_, ny_ = x + dx, y + dy
                    if 0 <= nx_ < self.map_size and 0 <= ny_ < self.map_size:
                        n_idx = self.xy_to_idx[(nx_, ny_)]
                        if not self.is_obstacle[n_idx]:
                            row.append(idx)
                            col.append(n_idx)
                            data.append(1.0)
                            
        self.adj_matrix = sp.csr_matrix((data, (row, col)), shape=(self.total_grids, self.total_grids))

    def init_centroids(self):
        centroids_xy = []
        step = self.map_size / np.sqrt(self.robot_num) if self.robot_num > 1 else self.map_size/2
        for i in range(self.robot_num):
            x = int(np.clip(round((i % int(np.ceil(np.sqrt(self.robot_num)))) * step + step/2 - 0.5), 0, self.map_size-1))
            y = int(np.clip(round((i // int(np.ceil(np.sqrt(self.robot_num)))) * step + step/2 - 0.5), 0, self.map_size-1))
            
            if (x, y) in self.obstacle_set:
                for dist in range(1, self.map_size):
                    found = False
                    for dx in range(-dist, dist+1):
                        for dy in range(-dist, dist+1):
                            if abs(dx) + abs(dy) != dist: continue
                            nx_, ny_ = x + dx, y + dy
                            if 0 <= nx_ < self.map_size and 0 <= ny_ < self.map_size and (nx_, ny_) not in self.obstacle_set:
                                x, y = nx_, ny_
                                found = True
                                break
                        if found: break
                    if found: break
            centroids_xy.append((x, y))
        return centroids_xy

    def partition(self, max_iter=10):
        # Always start from fixed, evenly spaced initial points to eliminate hysteresis
        centroids_xy = self.init_centroids()
            
        centroids = np.array([[x+0.5, y+0.5] for (x, y) in centroids_xy])
        
        for iteration in range(max_iter):
            all_distances = []
            for xy in centroids_xy:
                dist = dijkstra(csgraph=self.adj_matrix, directed=False, 
                              indices=self.xy_to_idx[xy], return_predecessors=False)
                all_distances.append(dist)
            all_distances = np.array(all_distances)
            
            sqrt_weights = np.sqrt(self.weights).reshape(-1, 1)
            weighted_distances = all_distances / (sqrt_weights + 1e-9)
            
            grid_assignments = np.zeros(len(self.free_grid_indices), dtype=int)
            for free_idx, grid_idx in enumerate(self.free_grid_indices):
                dist_to_centroids = weighted_distances[:, grid_idx]
                valid_mask = (dist_to_centroids < np.inf)
                if not np.any(valid_mask):
                    grid_assignments[free_idx] = -2
                else:
                    grid_assignments[free_idx] = np.where(valid_mask)[0][np.argmin(dist_to_centroids[valid_mask])]
            
            new_centroids_xy = []
            for i in range(self.robot_num):
                region_idx = [self.free_grid_indices[idx] for idx in np.where(grid_assignments == i)[0]]
                if not region_idx:
                    new_centroids_xy.append(centroids_xy[i])
                    continue
                region_xy = [self.idx_to_xy[idx] for idx in region_idx]
                mean_x, mean_y = np.mean([x for x,y in region_xy]), np.mean([y for x,y in region_xy])
                new_x, new_y = int(np.clip(round(mean_x), 0, self.map_size-1)), int(np.clip(round(mean_y), 0, self.map_size-1))
                
                if (new_x, new_y) in self.obstacle_set:
                    for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                        nx, ny = new_x + dx, new_y + dy
                        if 0 <= nx < self.map_size and 0 <= ny < self.map_size and (nx, ny) not in self.obstacle_set:
                            new_x, new_y = nx, ny
                            break
                new_centroids_xy.append((new_x, new_y))
            
            new_centroids = np.array([[x+0.5, y+0.5] for (x, y) in new_centroids_xy])
            if np.all(np.linalg.norm(new_centroids - centroids, axis=1) < 1e-3):
                break
            centroids_xy, centroids = new_centroids_xy, new_centroids
            
        self.centroids_xy = centroids_xy
        self.centroids = centroids
        self.full_assignments = np.full(self.total_grids, -1, dtype=int)
        for free_idx, grid_idx in enumerate(self.free_grid_indices):
            self.full_assignments[grid_idx] = grid_assignments[free_idx]
            
        for i in range(self.robot_num):
            self.robot_areas[i] = np.sum(grid_assignments == i)

    def _tile_in_order(self, region_mask, x_scan_order, y_scan_order, robot_id):
        covered_mask = np.zeros_like(region_mask, dtype=bool)
        tiles = []
        tile_layers = [(4, 4), (4, 2), (2, 4), (2, 2), (2, 1), (1, 2)]
        
        for w, h in tile_layers:
            for x in x_scan_order:
                for y in y_scan_order:
                    if covered_mask[x, y] or not region_mask[x, y]: continue
                    if x + w > self.map_size or y + h > self.map_size: continue
                    
                    can_place = True
                    for px in range(x, x + w):
                        for py in range(y, y + h):
                            if not region_mask[px, py] or covered_mask[px, py]:
                                can_place = False
                                break
                        if not can_place: break
                        
                    if can_place:
                        tiles.append((x, y, w, h, robot_id))
                        for px in range(x, x + w):
                            for py in range(y, y + h):
                                covered_mask[px, py] = True

        for x in x_scan_order:
            for y in y_scan_order:
                if not covered_mask[x,y] and region_mask[x,y]:
                    tiles.append((x, y, 1, 1, robot_id))
                    covered_mask[x,y] = True
                    
        return tiles, len(tiles)

    def generate_tiles_and_mst(self):
        order1_x, order1_y = list(range(self.map_size)), list(range(self.map_size))
        order2_x, order2_y = list(range(self.map_size-1, -1, -1)), list(range(self.map_size-1, -1, -1))
        order3_x, order3_y = list(range(self.map_size)), list(range(self.map_size-1, -1, -1))
        order4_x, order4_y = list(range(self.map_size-1, -1, -1)), list(range(self.map_size))

        self.all_tiles = []
        self.tile_stats = {rid: {f"{w}x{h}": 0 for w, h in [(4,4),(4,2),(2,4),(2,2),(2,1),(1,2),(1,1)]} for rid in range(self.robot_num)}
        self.robot_mst_dict = {rid: {'mst_edges': [], 'centroids': [], 'total_length': 0.0} for rid in range(self.robot_num)}

        for rid in range(self.robot_num):
            region_mask = np.zeros((self.map_size, self.map_size), dtype=bool)
            for x in range(self.map_size):
                for y in range(self.map_size):
                    if not self.is_obstacle[self.xy_to_idx[(x,y)]] and self.full_assignments[self.xy_to_idx[(x,y)]] == rid:
                        region_mask[x,y] = True
            
            t1, c1 = self._tile_in_order(region_mask, order1_x, order1_y, rid)
            t2, c2 = self._tile_in_order(region_mask, order2_x, order2_y, rid)
            t3, c3 = self._tile_in_order(region_mask, order3_x, order3_y, rid)
            t4, c4 = self._tile_in_order(region_mask, order4_x, order4_y, rid)
            
            schemes = [(c1, t1), (c2, t2), (c3, t3), (c4, t4)]
            schemes.sort(key=lambda x: x[0])
            best_tiles = schemes[0][1]
            self.all_tiles.extend(best_tiles)
            
            stats = self.tile_stats[rid]
            for (x,y,w,h,_) in best_tiles:
                key = f"{w}x{h}"
                if key in stats:
                    stats[key] += 1

            if len(best_tiles) <= 1:
                continue

            tile_map = np.full((self.map_size, self.map_size), -1, dtype=int)
            centroids = np.zeros((len(best_tiles), 2))
            
            for i, (x, y, w, h, _) in enumerate(best_tiles):
                tile_map[x:x+w, y:y+h] = i
                centroids[i] = [x + w/2.0, y + h/2.0]
                
            h_edges = np.column_stack((tile_map[:-1, :].ravel(), tile_map[1:, :].ravel()))
            v_edges = np.column_stack((tile_map[:, :-1].ravel(), tile_map[:, 1:].ravel()))
            all_edges = np.vstack((h_edges, v_edges))
            
            mask = (all_edges[:, 0] != -1) & (all_edges[:, 1] != -1) & (all_edges[:, 0] != all_edges[:, 1])
            valid_edges = all_edges[mask]
            
            valid_edges.sort(axis=1)
            unique_edges = np.unique(valid_edges, axis=0)
            
            if len(unique_edges) > 0:
                u = unique_edges[:, 0]
                v = unique_edges[:, 1]
                
                weights = np.linalg.norm(centroids[u] - centroids[v], axis=1)
                
                n_tiles = len(best_tiles)
                graph = coo_matrix((weights, (u, v)), shape=(n_tiles, n_tiles))
                
                mst = minimum_spanning_tree(graph)
                mst_coo = mst.tocoo()
                
                total_len = mst_coo.data.sum()
                mst_edges = [(centroids[r], centroids[c]) for r, c in zip(mst_coo.row, mst_coo.col)]
                
                self.robot_mst_dict[rid]['mst_edges'] = mst_edges
                self.robot_mst_dict[rid]['total_length'] = total_len
                self.robot_mst_dict[rid]['centroids'] = centroids.tolist()

    def compute_metrics(self):
        grid_counts = [np.sum(self.full_assignments == i) for i in range(self.robot_num)]
        mst_lengths = [self.robot_mst_dict[i]['total_length'] for i in range(self.robot_num)]
        return grid_counts, mst_lengths

    def visualize(self, step, output_dir, timestamp):
        self.compute_metrics()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        while len(colors) < self.robot_num:
            colors.append(np.random.rand(3,))
            
        fig, ax = plt.subplots(figsize=(16, 14))
        
        map_bounds = [0, self.map_size, 0, self.map_size]
        boundary_points = np.array([
            [map_bounds[0]-10, map_bounds[2]-10], [map_bounds[1]+10, map_bounds[2]-10],
            [map_bounds[0]-10, map_bounds[3]+10], [map_bounds[1]+10, map_bounds[3]+10]
        ])
        try:
            all_pts = np.vstack([self.centroids, boundary_points])
            vor = Voronoi(all_pts)
            voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='gray', line_width=1, line_alpha=0.5)
        except: pass
        
        for x in range(self.map_size):
            for y in range(self.map_size):
                idx = self.xy_to_idx[(x, y)]
                if self.is_obstacle[idx]:
                    rect = Rectangle((x, y), 1, 1, facecolor='black', edgecolor='white', linewidth=0.8)
                else:
                    rid = self.full_assignments[idx]
                    if rid >= 0:
                        rect = Rectangle((x, y), 1, 1, facecolor=colors[rid % len(colors)], alpha=0.1, edgecolor='white', linewidth=0.5)
                    elif rid == -2:
                        rect = Rectangle((x, y), 1, 1, facecolor='dimgray', alpha=0.4, edgecolor='white', linewidth=0.5, hatch='///')
                    else:
                        rect = Rectangle((x, y), 1, 1, facecolor='lightgray', alpha=0.1, edgecolor='white', linewidth=0.5)
                ax.add_patch(rect)
                
        for (x, y, w, h, rid) in self.all_tiles:
            color = colors[rid % len(colors)]
            tile_rect = Rectangle((x, y), w, h, facecolor=color, alpha=0.5, edgecolor='black', linewidth=1)
            ax.add_patch(tile_rect)
            
        for i, (cx, cy) in enumerate(self.centroids):
            color = colors[i % len(colors)]
            ax.scatter(cx, cy, color=color, s=50, marker='o', edgecolors='black', linewidth=1,
                       label=f'Robot {i+1}\nCentroid: {self.centroids_xy[i]}')
            
        for rid, mst_data in self.robot_mst_dict.items():
            color = colors[rid % len(colors)]
            for (p1, p2) in mst_data['mst_edges']:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=2.5, alpha=1, zorder=5)
            if mst_data['centroids']:
                cx = [c[0] for c in mst_data['centroids']]
                cy = [c[1] for c in mst_data['centroids']]
                ax.scatter(cx, cy, color='white', s=30, edgecolors=color, linewidth=1.5, zorder=6)

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
        print(f"\n📸 Snapshot saved: {save_path}")  # Add a newline before snapshot output
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
        
        # 4. Voronoi Partition
        solver.partition(max_iter=5)

        # 4.1 Record partition area fractions
        total_assigned_area = np.sum(solver.robot_areas)
        area_ratios = solver.robot_areas / (total_assigned_area + 1e-9)
        
        # 5. Generate tiling and compute MST
        solver.generate_tiles_and_mst()
        
        # 6. Record MST length
        mst_lengths = [solver.robot_mst_dict[i]['total_length'] for i in range(robot_num)]

        # Collect data for this step
        experiment_data.append({
            'Step': t,
            'Sensed_C_0': C_sampled_abs[0], 'Sensed_C_1': C_sampled_abs[1], 'Sensed_C_2': C_sampled_abs[2],
            'Weight_0': solver.weights[0], 'Weight_1': solver.weights[1], 'Weight_2': solver.weights[2],
            'MST_Length_0': mst_lengths[0], 'MST_Length_1': mst_lengths[1], 'MST_Length_2': mst_lengths[2],
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