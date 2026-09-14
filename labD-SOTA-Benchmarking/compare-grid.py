import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import matplotlib.lines as mlines  # Import mlines to create custom legend handles

def draw_single_grid(ax, custom_boxes, black_lines, scheme_name):
    """Draw a single grid world on the specified subplot (ax)"""
    
    # 1. Define the color mapping and map
    colors = {
        0: [32/255, 118/255, 180/255],  # Blue (actual map obstacles)
        1: [60/255, 15/255, 100/255],   # Purple (discretized obstacles)
        2: [255/255, 235/255, 60/255]   # Yellow (traversable region)
    }
    
    grid_map = np.array([
        [1, 1, 2, 2, 2, 2, 1, 2],
        [1, 1, 2, 2, 2, 2, 2, 2],
        [1, 1, 2, 2, 2, 2, 1, 1],
        [2, 2, 2, 1, 1, 1, 1, 1],
        [2, 2, 2, 1, 1, 1, 1, 1]
    ])

    # 2. Generate and render the base-map image using vectorized operations
    img = np.zeros((5, 8, 3))
    for key, color in colors.items():
        img[grid_map == key] = color
    ax.imshow(img, extent=[0, 8, 5, 0], zorder=1)
    
    # 3. Draw static simulated obstacles (add in a batch)
    obstacles = [
        patches.Circle((6.25, 0.25), 0.18, color='#2078B4', zorder=2),
        patches.Circle((6.95, 2.25), 0.18, color='#2078B4', zorder=2),
        patches.Rectangle((0.0, 0.0), 1.2, 2.5, color='#2078B4', zorder=2),
        patches.Rectangle((3.2, 3.1), 4.8, 1.9, color='#2078B4', zorder=2)
    ]
    for obs in obstacles:
        ax.add_patch(obs)

    # 4. Configure grid lines
    ax.set_xticks(np.arange(0, 9, 1))
    ax.set_yticks(np.arange(0, 6, 1))
    ax.set_xticks(np.arange(0, 8.5, 0.5), minor=True)
    ax.set_yticks(np.arange(0, 5.5, 0.5), minor=True)

    ax.grid(which='major', color='#B0C460', linestyle='-', linewidth=2, zorder=5)
    ax.grid(which='minor', color='white', linestyle='--', linewidth=0.8, alpha=0.6, zorder=5)
    ax.set_axisbelow(False) # Allow grid lines to appear above other elements

    # 5. Draw the contracted borders
    margin = 0.02 
    for cx, cy, w, h, color in custom_boxes:
        draw_w, draw_h = w - 2 * margin, h - 2 * margin
        ax.add_patch(patches.Rectangle(
            (cx - draw_w / 2, cy - draw_h / 2), draw_w, draw_h,
            fill=False, edgecolor=color, linewidth=2, zorder=6
        ))

    # 6. Draw black search-tree edges, calculate distances, and draw nodes
    all_points = set()
    total_length = 0.0
    for p1, p2 in black_lines:
        # Draw edges
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=6.0, zorder=9)
        # Collect endpoints
        all_points.update([p1, p2])
        # Accumulate length
        total_length += np.hypot(p2[0] - p1[0], p2[1] - p1[1])

    # Draw black dots at junctions
    for x, y in all_points:
        ax.plot(x, y, 'ko', markersize=12, zorder=10)
    
    # 7. Hide axis labels and retain only the grid
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    # ★★★ 8. Show the scheme name and total path length in a top-center legend ★★★
    # Use a black line with black circular markers as the legend handle
    legend_handle = mlines.Line2D([], [], color='k', marker='o', markersize=6, linewidth=2.5,
                                label=f'{scheme_name}\nLength:{total_length:.2f}')

    # Place the legend near the upper center of the axes and store the returned legend object
    # Place the legend in the lower right and configure its spacing
    leg = ax.legend(handles=[legend_handle], 
                    loc='lower right', 
                    prop={'size': 35, 'weight': 'bold'}, 
                    framealpha=1.0,
                    borderpad=0.2,       # [Key] Set the padding between the legend border and its contents (default 0.4)
                    handlelength=1.0,    # [Key] Set the length of the black line handle (default 2.0)
                    handletextpad=0.4    # [Key] Set the gap between the black line and text (default 0.8)
                   )

    # Explicitly set the legend zorder to 20 so it appears above all other plot elements
    leg.set_zorder(20)


def draw_grid_world():
    # 1. Correct the canvas aspect ratio: width 8, height 10 (5*2), matching the data
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    
    # --- First dataset ---
    boxes_1 = [
        [6.5, 1.5, 1.0, 1.0, 'red'], [7.5, 1.5, 1.0, 1.0, 'red'],
        [7.5, 0.5, 1.0, 1.0, 'red'], [2.5, 3.5, 1.0, 1.0, 'red'],  
        [2.5, 4.5, 1.0, 1.0, 'red'], [2.5, 0.5, 1.0, 1.0, 'red'],
        [3.5, 0.5, 1.0, 1.0, 'red'], [4.5, 0.5, 1.0, 1.0, 'red'],
        [5.5, 0.5, 1.0, 1.0, 'red'], [1.0, 4.0, 2.0, 2.0, 'blue'],
        [3.0, 2.0, 2.0, 2.0, 'blue'], [5.0, 2.0, 2.0, 2.0, 'blue'],
    ]
    lines_1 = [
        [(1.0, 4.0), (2.5, 3.5)], [(2.5, 3.5), (2.5, 4.5)],
        [(2.5, 3.5), (3.0, 2.0)], [(3.0, 2.0), (3.5, 0.5)],
        [(2.5, 0.5), (5.5, 0.5)], [(4.5, 0.5), (5.0, 2.0)],
        [(5.0, 2.0), (6.5, 1.5)], [(6.5, 1.5), (7.5, 1.5)],
        [(7.5, 1.5), (7.5, 0.5)]
    ]

    # --- Second dataset ---
    boxes_2 = [
        [6.5, 1.5, 1.0, 1.0, 'red'],  
        [7.5, 1.0, 1.0, 2.0, 'green'],  
        [5.0, 2.5, 2.0, 1.0, 'green'],  
        [3.0, 2.5, 2.0, 1.0, 'green'],
        [2.5, 4.0, 1.0, 2.0, 'green'],
        [1.0, 4.0, 2.0, 2.0, 'blue'],
        [4.0, 1.0, 4.0, 2.0, 'purple'],
    ]
    lines_2 = [
        [(1.0, 4.0), (2.5, 4.0)], [(2.5, 4.0), (3.0, 2.5)],
        [(3.0, 2.5), (4.0, 1.0)], [(4.0, 1.0), (5.0, 2.5)],
        [(5.0, 2.5), (6.5, 1.5)], [(6.5, 1.5), (7.5, 1.0)]
    ]

    # Draw two subplots
    draw_single_grid(axes[0], boxes_1, lines_1, "CPPF")
    draw_single_grid(axes[1], boxes_2, lines_2, "MCCA")

    # 1. Set outer padding pad to 0 to remove margins; use h_pad to separate the stacked subplots
    # h_pad can be adjusted as needed, for example to 1.5, 2.0, or 3.0
    plt.tight_layout(pad=0, h_pad=0.5) 
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'compare-grid.png')
    
    # 2. Remove excess whitespace when saving
    fig.savefig(output_path, dpi=200, bbox_inches='tight', pad_inches=0)

if __name__ == "__main__":
    draw_grid_world()