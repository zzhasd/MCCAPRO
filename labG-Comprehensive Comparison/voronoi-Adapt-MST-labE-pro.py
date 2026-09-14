import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
import networkx as nx
import time
import os
import warnings
warnings.filterwarnings('ignore')

# Configure CJK fonts
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class OptimizedMCPP:
    def __init__(self, map_size, robot_num, obstacle_ratio=0.1, max_iter=20):
        self.map_size = map_size
        self.robot_num = robot_num
        self.obstacle_num = int(map_size * map_size * obstacle_ratio)
        self.max_iter = max_iter
        self.total_grids = map_size * map_size
        
        self.generate_map()
        self.build_sparse_graph()
        
        # Retain data for plotting
        self.centroids_xy = []
        self.all_tiles = []
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

    def partition(self):
        from collections import deque 

        centroids_xy = self.init_centroids()
        centroids = np.array([[x+0.5, y+0.5] for (x, y) in centroids_xy])
        
        history_centroids = [] 
        
        for iteration in range(self.max_iter):
            grid_assignments = np.full(self.total_grids, -2, dtype=int)
            
            for obs_idx in self.obstacle_set:
                grid_assignments[self.xy_to_idx[obs_idx]] = -1
                
            queue = deque()
            
            for i, xy in enumerate(centroids_xy):
                idx = self.xy_to_idx[xy]
                grid_assignments[idx] = i
                queue.append((xy[0], xy[1], i)) 
                
            while queue:
                cx, cy, r_id = queue.popleft()
                
                for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                        n_idx = self.xy_to_idx[(nx, ny)]
                        if grid_assignments[n_idx] == -2:
                            grid_assignments[n_idx] = r_id
                            queue.append((nx, ny, r_id))
            
            new_centroids_xy = []
            for i in range(self.robot_num):
                region_idx = np.where(grid_assignments == i)[0]
                
                if len(region_idx) == 0:
                    new_centroids_xy.append(centroids_xy[i])
                    continue
                    
                mean_x = np.mean([self.idx_to_xy[idx][0] for idx in region_idx])
                mean_y = np.mean([self.idx_to_xy[idx][1] for idx in region_idx])
                new_x = int(np.clip(round(mean_x), 0, self.map_size-1))
                new_y = int(np.clip(round(mean_y), 0, self.map_size-1))
                
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
                
            tuple_centroids = tuple(new_centroids_xy)
            if tuple_centroids in history_centroids:
                break
                
            history_centroids.append(tuple_centroids)
            if len(history_centroids) > 3:
                history_centroids.pop(0)

            centroids_xy, centroids = new_centroids_xy, new_centroids
            
        self.centroids_xy = centroids_xy  
        self.full_assignments = grid_assignments

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
        all_tiles = []
        order1_x, order1_y = list(range(self.map_size)), list(range(self.map_size))
        order2_x, order2_y = list(range(self.map_size-1, -1, -1)), list(range(self.map_size-1, -1, -1))
        order3_x, order3_y = list(range(self.map_size)), list(range(self.map_size-1, -1, -1))
        order4_x, order4_y = list(range(self.map_size-1, -1, -1)), list(range(self.map_size))

        for rid in range(self.robot_num):
            region_mask = np.zeros((self.map_size, self.map_size), dtype=bool)
            for x in range(self.map_size):
                for y in range(self.map_size):
                    if (x,y) not in self.obstacle_set and self.full_assignments[self.xy_to_idx[(x,y)]] == rid:
                        region_mask[x,y] = True
            
            tiles1, count1 = self._tile_in_order(region_mask, order1_x, order1_y, rid)
            tiles2, count2 = self._tile_in_order(region_mask, order2_x, order2_y, rid)
            tiles3, count3 = self._tile_in_order(region_mask, order3_x, order3_y, rid)
            tiles4, count4 = self._tile_in_order(region_mask, order4_x, order4_y, rid)
            
            schemes = [(count1, tiles1), (count2, tiles2), (count3, tiles3), (count4, tiles4)]
            schemes.sort(key=lambda x: x[0])
            all_tiles.extend(schemes[0][1])
            
        self.all_tiles = all_tiles
        
        self.robot_mst_dict = {rid: {'mst_edges': [], 'centroids': []} for rid in range(self.robot_num)}
        robot_tiles = {rid: [] for rid in range(self.robot_num)}
        for t in all_tiles: robot_tiles[t[4]].append(t)
        
        for rid, tiles in robot_tiles.items():
            if len(tiles) <= 1: continue
            
            cell_to_idx = {}
            centroids = []
            for i, (x, y, w, h, _) in enumerate(tiles):
                centroids.append(np.array([x+w/2, y+h/2]))
                for px in range(x, x+w):
                    for py in range(y, y+h):
                        cell_to_idx[(px, py)] = i
            
            G = nx.Graph()
            edges_added = set()
            for i, (x, y, w, h, _) in enumerate(tiles):
                G.add_node(i)
                perimeter = [(px, y-1) for px in range(x, x+w)] + \
                            [(px, y+h) for px in range(x, x+w)] + \
                            [(x-1, py) for py in range(y, y+h)] + \
                            [(x+w, py) for py in range(y, y+h)]
                            
                for nx_, ny_ in perimeter:
                    if (nx_, ny_) in cell_to_idx:
                        j = cell_to_idx[(nx_, ny_)]
                        if i != j:
                            edge = (min(i,j), max(i,j))
                            if edge not in edges_added:
                                G.add_edge(i, j, weight=np.linalg.norm(centroids[i] - centroids[j]))
                                edges_added.add(edge)
            
            mst = nx.minimum_spanning_tree(G, weight='weight')
            
            for u, v, data in mst.edges(data=True):
                self.robot_mst_dict[rid]['mst_edges'].append((centroids[u], centroids[v]))
            self.robot_mst_dict[rid]['centroids'] = centroids

    def save_visualization(self, filepath, title):
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        while len(colors) < self.robot_num:
            colors.append(np.random.rand(3,))
            
        fig, ax = plt.subplots(figsize=(10, 10))
        
        for x in range(self.map_size):
            for y in range(self.map_size):
                idx = self.xy_to_idx[(x, y)]
                if self.is_obstacle[idx]:
                    rect = Rectangle((x, y), 1, 1, facecolor='black', edgecolor='none')
                else:
                    rid = self.full_assignments[idx]
                    if rid >= 0:
                        color = colors[rid % len(colors)]
                        rect = Rectangle((x, y), 1, 1, facecolor=color, alpha=0.15, edgecolor='none')
                    elif rid == -2:
                        rect = Rectangle((x, y), 1, 1, facecolor='dimgray', alpha=0.4, hatch='///')
                    else:
                        rect = Rectangle((x, y), 1, 1, facecolor='lightgray', alpha=0.1)
                ax.add_patch(rect)
                
        for (x, y, w, h, rid) in self.all_tiles:
            color = colors[rid % len(colors)]
            tile_rect = Rectangle((x, y), w, h, facecolor=color, alpha=0.4, edgecolor='black', linewidth=0.5)
            ax.add_patch(tile_rect)
            
        for rid, mst_data in self.robot_mst_dict.items():
            color = colors[rid % len(colors)]
            mst_edges = mst_data['mst_edges']
            centroids = mst_data['centroids']
            
            for (p1, p2) in mst_edges:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=2.0, zorder=4)
                
            if len(centroids) > 0:
                cx = [c[0] for c in centroids]
                cy = [c[1] for c in centroids]
                ax.scatter(cx, cy, color='white', s=15, edgecolors=color, linewidth=1, zorder=5)

        for i, (cx, cy) in enumerate(self.centroids_xy):
            color = colors[i % len(colors)]
            ax.scatter(cx + 0.5, cy + 0.5, color=color, s=150, marker='*', edgecolors='black', linewidth=1.5, zorder=6)

        ax.set_xlim(0, self.map_size)
        ax.set_ylim(0, self.map_size)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14)
        ax.set_xticks([]) 
        ax.set_yticks([])
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig) 


def run_stress_test():
    # Define 50 random seeds
    seeds = [
        42, 100, 256, 512, 1024, 2048, 4096, 8192, 12345, 99999,
        7, 13, 29, 63, 127, 255, 511, 777, 1337, 2024,
        3141, 4097, 5003, 6666, 7001, 8191, 9001, 10007, 12011, 15013,
        17021, 19031, 21001, 23003, 25013, 27011, 29009, 31013, 33023, 35023,
        37019, 39041, 41047, 43051, 45053, 47057, 49069, 51071, 53087, 55073
    ]
    
    # 1. Get the absolute directory of the current script (labE-Scalability-Complexity)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Generate a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 3. Generate a timestamped txt filename in the current script directory
    log_file_path = os.path.join(script_dir, f"StressTest_DataLog_{timestamp}.txt")
    
    print(f"📂 Experiment plots and TXT data will be saved automatically to: {script_dir}")
    
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write(f" mCPP Spatial and fleet scalability stress test - data archive ({len(seeds)} random seeds)\n")
        f.write(f" Test seed list: {seeds}\n")
        f.write("="*60 + "\n\n")
        
        # ---------------- Experiment 1: spatial scalability test ----------------
        space_maps = [25, 50, 75, 100, 125, 150, 175, 200]
        space_times_dict = {ms: [] for ms in space_maps} 
        
        title1 = "▶ Experiment 1: spatial scalability stress test (fixed robots=5)"
        print("\n" + "="*60 + "\n" + title1 + "\n" + "="*60)
        f.write("="*60 + "\n" + title1 + "\n" + "="*60 + "\n")
        
        for ms in space_maps:
            for seed in seeds:
                np.random.seed(seed)
                start_t = time.time()
                solver = OptimizedMCPP(map_size=ms, robot_num=5)
                solver.partition()
                solver.generate_tiles_and_mst()
                elapsed = time.time() - start_t
                space_times_dict[ms].append(elapsed)
                
                res_str = f"[*] Map {ms:>3}x{ms:<3} | Seed: {seed:>5} | Elapsed: {elapsed:.3f} s"
                print(res_str)
                f.write(res_str + "\n")
                
                # if seed == seeds[0]:
                #     img_name = os.path.join(script_dir, f"SpaceTest_Map{ms}x{ms}_Robots5_seed{seed}.png")
                #     solver.save_visualization(img_name, f"Space Extension: Map {ms}x{ms}, 5 Robots (Seed {seed})")
            
            avg_time = np.mean(space_times_dict[ms])
            f.write(f"--- Map {ms}x{ms} complete; mean runtime over {len(seeds)} trials: {avg_time:.3f} s ---\n\n")

        # ---------------- Experiment 2: fleet scalability test ----------------
        cluster_robots = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
        cluster_times_dict = {rn: [] for rn in cluster_robots} 
        fixed_map = 100
        
        title2 = f"▶ Experiment 2: fleet scalability stress test (fixed map={fixed_map}x{fixed_map})"
        print("\n" + "="*60 + "\n" + title2 + "\n" + "="*60)
        f.write("\n" + "="*60 + "\n" + title2 + "\n" + "="*60 + "\n")
        
        for rn in cluster_robots:
            for seed in seeds:
                np.random.seed(seed)
                start_t = time.time()
                solver = OptimizedMCPP(map_size=fixed_map, robot_num=rn)
                solver.partition()
                solver.generate_tiles_and_mst()
                elapsed = time.time() - start_t
                cluster_times_dict[rn].append(elapsed)
                
                res_str = f"[*] Robot count: {rn:>2} | Seed: {seed:>5} | Elapsed: {elapsed:.3f} s"
                print(res_str)
                f.write(res_str + "\n")
                
                if seed == seeds[0]:
                    img_name = os.path.join(script_dir, f"ClusterTest_Map{fixed_map}x{fixed_map}_Robots{rn}_seed{seed}.png")
                    solver.save_visualization(img_name, f"Cluster Extension: Map {fixed_map}x{fixed_map}, {rn} Robots (Seed {seed})")
            
            avg_time = np.mean(cluster_times_dict[rn])
            f.write(f"--- Robot count {rn} complete; mean runtime over {len(seeds)} trials: {avg_time:.3f} s ---\n\n")
            
        f.write("\n✅ All stress tests complete; figures and data archived.\n")
        
    # ================= Draw boxplots and retain the fitted curves =================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # -------- Spatial scaling plot --------
    space_data = [space_times_dict[ms] for ms in space_maps]
    space_means = [np.mean(times) for times in space_data]
    
    ax1.boxplot(space_data, positions=space_maps, widths=8, patch_artist=True,
                boxprops=dict(facecolor='#fadbd8', color='#c0392b', alpha=0.7),
                medianprops=dict(color='#e74c3c', linewidth=2))
    
    x1 = np.array(space_maps)
    y1 = np.array(space_means)
    z1 = np.polyfit(x1, y1, 2)
    p1 = np.poly1d(z1)
    x1_fit = np.linspace(min(x1)-10, max(x1)+10, 100)
    ax1.plot(x1_fit, p1(x1_fit), '--', color='#c0392b', alpha=0.8, linewidth=2, label='Fitted mean curve (O(N^2))')
    
    ax1.set_title('Algorithm spatial scalability (Computation vs. Map Size)', fontsize=12)
    ax1.set_xlabel('Map Size (N x N)', fontsize=11)
    ax1.set_ylabel('Compute Time (Seconds)', fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend()

    # -------- Fleet scaling plot --------
    cluster_data = [cluster_times_dict[rn] for rn in cluster_robots]
    cluster_means = [np.mean(times) for times in cluster_data]
    
    ax2.boxplot(cluster_data, positions=cluster_robots, widths=0.8, patch_artist=True,
                boxprops=dict(facecolor='#d4e6f1', color='#2980b9', alpha=0.7),
                medianprops=dict(color='#3498db', linewidth=2))
    
    x2 = np.array(cluster_robots)
    y2 = np.array(cluster_means)
    z2 = np.polyfit(x2, y2, 1)
    p2 = np.poly1d(z2)
    x2_fit = np.linspace(min(x2)-1, max(x2)+1, 100)
    ax2.plot(x2_fit, p2(x2_fit), '--', color='#2980b9', alpha=0.8, linewidth=2, label='Linear fit to means (O(K))')
    
    ax2.set_title('Algorithm fleet scalability (Computation vs. Robot Num)', fontsize=12)
    ax2.set_xlabel('Number of Robots (K)', fontsize=11)
    ax2.set_ylabel('Compute Time (Seconds)', fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    # 4. Save plots beside the script
    plt.savefig(os.path.join(script_dir, "Final_Performance_Report_Boxplot.png"), dpi=200, bbox_inches='tight')

if __name__ == '__main__':
    run_stress_test()