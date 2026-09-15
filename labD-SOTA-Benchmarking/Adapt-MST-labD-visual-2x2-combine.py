import os
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import numpy as np
import matplotlib.patches as patches
import matplotlib.lines as mlines  # Import mlines to create custom legend handles
from matplotlib.ticker import PercentFormatter, FuncFormatter


# PaperPlaza/IEEE-safe PDF fonts: Type 42 TrueType, embedded by Matplotlib.
plt.rcParams.update({
    'pdf.fonttype': 42,
    'pdf.use14corefonts': False,
    'ps.fonttype': 42,
    'ps.useafm': False,
    'text.usetex': False,
    'font.family': 'DejaVu Sans',
    'font.sans-serif': ['DejaVu Sans'],
    'mathtext.fontset': 'dejavusans',
    'axes.unicode_minus': False,
})


# ================================
# compare-grid.py: Merge the original plotting logic into this script
# ================================
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


def draw_grid_world_to_image():
    """Draw using the original compare-grid.py method and return an in-memory PNG image."""
    # 1. Correct the canvas aspect ratio: width 8, height 10 (5*2), matching the data
    compare_fig, compare_axes = plt.subplots(2, 1, figsize=(8, 10))
    
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
    draw_single_grid(compare_axes[0], boxes_1, lines_1, "CPPF")
    draw_single_grid(compare_axes[1], boxes_2, lines_2, "MCCA")

    # 1. Set outer padding pad to 0 to remove margins; use h_pad to separate the stacked subplots
    # h_pad can be adjusted as needed, for example to 1.5, 2.0, or 3.0
    plt.tight_layout(pad=0, h_pad=0.5)

    # 2. Keep the original compare-grid.py save settings, but write to memory instead of generating or requiring compare-grid.png
    compare_buffer = io.BytesIO()
    compare_fig.savefig(compare_buffer, format='png', dpi=200, bbox_inches='tight', pad_inches=0)
    compare_buffer.seek(0)
    compare_img = plt.imread(compare_buffer, format='png')

    plt.close(compare_fig)
    compare_buffer.close()
    return compare_img


# ================================
# 1. Get the current script directory and read the txt data file
# ================================
script_dir = os.path.dirname(os.path.abspath(__file__))

# ★★★ Set the txt filename here ★★★
txt_filename = "experiment_data_20260502_nvidia.txt" 
txt_filepath = os.path.join(script_dir, txt_filename)

if not os.path.exists(txt_filepath):
    raise FileNotFoundError(f"Data file not found: {txt_filepath}")

with open(txt_filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Extract CSV data blocks for each density using regular expressions
match_10 = re.search(r'data_10_str\s*=\s*"""(.*?)"""', content, re.DOTALL)
match_15 = re.search(r'data_15_str\s*=\s*"""(.*?)"""', content, re.DOTALL)
match_20 = re.search(r'data_20_str\s*=\s*"""(.*?)"""', content, re.DOTALL)

if not (match_10 and match_15 and match_20):
    raise ValueError("Could not extract complete data from the txt file; ensure it uses the format generated by the preceding script!")

# ================================
# 2. Load data into pandas DataFrame
# ================================
df_10 = pd.read_csv(io.StringIO(match_10.group(1).strip()))
df_15 = pd.read_csv(io.StringIO(match_15.group(1).strip()))
df_20 = pd.read_csv(io.StringIO(match_20.group(1).strip()))

datasets = [("10% Obstacles", df_10), ("15% Obstacles", df_15), ("20% Obstacles", df_20)]

# ================================
# 3. First generate the lower-right image in memory using the original compare-grid.py logic, then configure the main figure style
# ================================
compare_img = draw_grid_world_to_image()

plt.style.use('seaborn-v0_8-whitegrid')
colors = [ '#2ca02c', '#9467bd', '#ff7f0e','#1f77b4', '#d62728'] 

# ================================
# 4. Create a 2x2 Figure and Axes
# ================================
fig, axes = plt.subplots(2, 2, figsize=(24, 20), sharey=True) 
axes_flat = axes.flatten()

# fig.suptitle("Reduction Rates of MCCA-Path vs CPPF on Grid Maps", fontsize=35, fontweight='bold', y=0.99)

# ★★★ Define a secondary-axis formatter: convert 1000 to 1k ★★★
def thousands_formatter(x, pos):
    if x == 0:
        return '0'
    return f'{int(x/1000)}k'

# Prepare variables for the shared legend
global_lines = []
global_labels = []

# ================================
# 5. Plotting and calculation logic
# ================================
for i, (title, df) in enumerate(datasets):
    ax = axes_flat[i]
    ax.grid(False)
    
    # Compute reduction rates (preserve the original calculation logic)
    time_reduction = (df['B_time'] - df['A_time']) / df['B_time'] * 100
    blocks_reduction = (df['B_blocks'] - df['A_blocks']) / df['B_blocks'] * 100
    mst_reduction = (df['B_mst'] - df['A_mst']) / df['B_mst'] * 100
    
    # ===================== Draw on the primary axes (ax) A_blocks/B_blocks =====================
    # Draw primary-axis line plots
    line4 = ax.plot(df['grid_size'], df['A_blocks'], marker='o', label='MCCA Blocks', 
            color=colors[3], linewidth=6, markersize=15, linestyle='--')
    line5 = ax.plot(df['grid_size'], df['B_blocks'], marker='o', label='CPPF Blocks', 
            color=colors[4], linewidth=6, markersize=15, linestyle='--')

    # Configure the primary X/Y axes
    ax.set_xlabel(f'Grid Size ({title})', fontsize=50, fontweight='bold')
    ax.set_ylabel('Blocks Count', fontsize=50, fontweight='bold', color='black')  # Primary Y-axis label
    ax.tick_params(axis='both', labelsize=50)
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))  # Primary Y axis: thousands separator
    
    # ===================== Create secondary axes (ax_twin) for the three reduction rates =====================
    ax_twin = ax.twinx()
    ax_twin.grid(False)
    
    # Draw secondary-axis line plots
    line1 = ax_twin.plot(df['grid_size'], time_reduction, marker='s', label='Computation Time Reduction', 
            color=colors[0], linewidth=6, markersize=15, linestyle='-.')
    line2 = ax_twin.plot(df['grid_size'], blocks_reduction, marker='s', label='Blocks Number Reduction', 
            color=colors[1], linewidth=6, markersize=15, linestyle='-.')
    line3 = ax_twin.plot(df['grid_size'], mst_reduction, marker='s', label='Path Length Reduction', 
            color=colors[2], linewidth=6, markersize=15, linestyle='-.')
    
    # Configure the secondary Y axis with a light-green style
    ax_twin.set_ylabel('Reduction Rate (%)', fontsize=50, fontweight='bold', color='green')  # Secondary Y-axis label
    ax_twin.tick_params(axis='y', labelsize=50, labelcolor='green')  # Green tick labels
    ax_twin.yaxis.set_major_formatter(PercentFormatter())  # Secondary Y axis: percentage format
    # Set the right spine of the secondary axes to green
    ax_twin.spines['right'].set_color('green')

    # Save the lines and labels from the first plot for the shared bottom legend
    if i == 0:
        global_lines = line1 + line2 + line3 + line4 + line5
        global_labels = [l.get_label() for l in global_lines]

# ================================
# 6. Handle the lower-right panel and insert the compare-grid plot directly
# ================================
# Run tight_layout first to determine the final positions of all 2x2 subplots
# Use rect=[0, 0.08, 1, 1] to reserve the bottom 8% for the shared legend and prevent overlap
plt.tight_layout(rect=[0, 0.08, 1, 1])

# Remove the original fourth subplot to fully detach its sharey=True binding
ax4 = axes_flat[3]
ax4.remove()

# ★★★ Main change: one shared, bottom-centered legend arranged in two columns ★★★

legend_kwargs = {
    'loc': 'lower center',
    'prop': {'size': 50, 'weight': 'bold'},
    'columnspacing': 0.6,
    'handletextpad': 0.25,
    'borderpad': 0.1,
    'labelspacing': 0.35,
    'frameon': False
}

# ncol=2 arranges the 5 legend entries into two columns:
# Left column: first 3 entries
# Right column: last 2 entries
fig.legend(
    global_lines,
    global_labels,
    ncol=2,
    bbox_to_anchor=(0.5, -0.08),
    **legend_kwargs
)

# compare-grid image already generated in memory; no longer check or read compare-grid.png
img = compare_img

# Get the original image dimensions and calculate its aspect ratio
img_h, img_w = img.shape[:2]
img_aspect = img_w / img_h

# Get the current Figure dimensions and calculate its aspect ratio
fig_w, fig_h = fig.get_size_inches()
fig_aspect = fig_w / fig_h

# ★★★ Adjust the image placement here ★★★
# The legend is at the bottom; placing the image too low may cause overlap, so increase img_bottom
img_left = 0.57    # Fractional offset from the left edge of the canvas (0~1)
img_bottom = 0.09  # Fractional offset from the bottom edge of the canvas (0~1)
img_width = 0.30   # Fraction of the total canvas width occupied by the image (0~1)

# Use the Figure and Image aspect ratios to calculate the relative image height without distortion
img_height = (img_width / img_aspect) * fig_aspect

# Add independent axes using the custom position and calculated height
ax_img = fig.add_axes([img_left, img_bottom, img_width, img_height])

ax_img.imshow(img)

# Hide the image border and axis ticks
ax_img.axis('off')

# ================================
# 7. Save PNG and PDF files
# ================================
output_png = os.path.join(script_dir, 'reduction_rates_2x2_with_bottom_legend.png')
output_pdf = os.path.join(script_dir, 'reduction_rates_2x2_with_bottom_legend.pdf')

fig.savefig(output_png, dpi=200, bbox_inches='tight')
fig.savefig(output_pdf, dpi=200, bbox_inches='tight')
plt.close(fig)

print(f"PNG Figure saved to: {output_png}")
print(f"PDF Figure saved to: {output_pdf}")
