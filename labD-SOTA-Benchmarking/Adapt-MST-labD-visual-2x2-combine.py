import os
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import numpy as np
import matplotlib.patches as patches
import matplotlib.lines as mlines  # 引入 mlines 用于创建自定义图例句柄
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
# compare-grid.py：原有绘图逻辑合并到当前脚本
# ================================
def draw_single_grid(ax, custom_boxes, black_lines, scheme_name):
    """在指定的子图 (ax) 上绘制单个网格世界"""
    
    # 1. 定义颜色映射与地图
    colors = {
        0: [32/255, 118/255, 180/255],  # 蓝色 (真实地图障碍物)
        1: [60/255, 15/255, 100/255],   # 紫色 (网格化障碍物)
        2: [255/255, 235/255, 60/255]   # 黄色 (可通行区域)
    }
    
    grid_map = np.array([
        [1, 1, 2, 2, 2, 2, 1, 2],
        [1, 1, 2, 2, 2, 2, 2, 2],
        [1, 1, 2, 2, 2, 2, 1, 1],
        [2, 2, 2, 1, 1, 1, 1, 1],
        [2, 2, 2, 1, 1, 1, 1, 1]
    ])

    # 2. 向量化生成底图图像数据并绘制
    img = np.zeros((5, 8, 3))
    for key, color in colors.items():
        img[grid_map == key] = color
    ax.imshow(img, extent=[0, 8, 5, 0], zorder=1)
    
    # 3. 绘制静态模拟障碍物 (批量添加)
    obstacles = [
        patches.Circle((6.25, 0.25), 0.18, color='#2078B4', zorder=2),
        patches.Circle((6.95, 2.25), 0.18, color='#2078B4', zorder=2),
        patches.Rectangle((0.0, 0.0), 1.2, 2.5, color='#2078B4', zorder=2),
        patches.Rectangle((3.2, 3.1), 4.8, 1.9, color='#2078B4', zorder=2)
    ]
    for obs in obstacles:
        ax.add_patch(obs)

    # 4. 设置网格线
    ax.set_xticks(np.arange(0, 9, 1))
    ax.set_yticks(np.arange(0, 6, 1))
    ax.set_xticks(np.arange(0, 8.5, 0.5), minor=True)
    ax.set_yticks(np.arange(0, 5.5, 0.5), minor=True)

    ax.grid(which='major', color='#B0C460', linestyle='-', linewidth=2, zorder=5)
    ax.grid(which='minor', color='white', linestyle='--', linewidth=0.8, alpha=0.6, zorder=5)
    ax.set_axisbelow(False) # 解除网格线被强制置底的限制

    # 5. 绘制收缩边框
    margin = 0.02 
    for cx, cy, w, h, color in custom_boxes:
        draw_w, draw_h = w - 2 * margin, h - 2 * margin
        ax.add_patch(patches.Rectangle(
            (cx - draw_w / 2, cy - draw_h / 2), draw_w, draw_h,
            fill=False, edgecolor=color, linewidth=2, zorder=6
        ))

    # 6. 绘制搜索树黑线、计算距离并绘制节点
    all_points = set()
    total_length = 0.0
    for p1, p2 in black_lines:
        # 画线
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=6.0, zorder=9)
        # 收集端点
        all_points.update([p1, p2])
        # 累加长度
        total_length += np.hypot(p2[0] - p1[0], p2[1] - p1[1])

    # 绘制连接处黑点
    for x, y in all_points:
        ax.plot(x, y, 'ko', markersize=12, zorder=10)
    
    # 7. 隐藏坐标轴的标签，只保留网格
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    # ★★★ 8. 在中部顶部通过 legend 显示方案名称和总路径长度 ★★★
    # 创建一条带黑色圆点的黑线作为图例的图标
    legend_handle = mlines.Line2D([], [], color='k', marker='o', markersize=6, linewidth=2.5,
                                label=f'{scheme_name}\nLength:{total_length:.2f}')

    # 将图例放置在轴域的中部偏上，并将返回的图例对象赋值给变量
    # 将图例放置在轴域的右下角，并添加控制紧凑度的参数
    leg = ax.legend(handles=[legend_handle], 
                    loc='lower right', 
                    prop={'size': 35, 'weight': 'bold'}, 
                    framealpha=1.0,
                    borderpad=0.2,       # 【关键】控制图例边框与内部内容之间的留白（默认 0.4）
                    handlelength=1.0,    # 【关键】控制前面那根黑线（句柄）的长度（默认 2.0）
                    handletextpad=0.4    # 【关键】控制黑线与文字之间的间距（默认 0.8）
                   )

    # 显式设置图例的 zorder 为 20，确保其悬浮在所有其他图形元素的最顶层
    leg.set_zorder(20)


def draw_grid_world_to_image():
    """按 compare-grid.py 原有方式绘制，并直接返回内存中的 PNG 图像。"""
    # 1. 修正画布比例：宽度8，高度10 (5*2)，使其与数据比例一致
    compare_fig, compare_axes = plt.subplots(2, 1, figsize=(8, 10))
    
    # --- 第一组数据 ---
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

    # --- 第二组数据 ---
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

    # 绘制两个子图
    draw_single_grid(compare_axes[0], boxes_1, lines_1, "CPPF")
    draw_single_grid(compare_axes[1], boxes_2, lines_2, "MCCA")

    # 1. 设置外边距 pad 为 0 去除四周白边，设置 h_pad 拉开上下子图的间距
    # h_pad 的数值可以根据你的喜好调整，比如 1.5, 2.0, 3.0 等
    plt.tight_layout(pad=0, h_pad=0.5)

    # 2. 保持 compare-grid.py 原有保存参数，但改为写入内存，不再生成/依赖 compare-grid.png
    compare_buffer = io.BytesIO()
    compare_fig.savefig(compare_buffer, format='png', dpi=200, bbox_inches='tight', pad_inches=0)
    compare_buffer.seek(0)
    compare_img = plt.imread(compare_buffer, format='png')

    plt.close(compare_fig)
    compare_buffer.close()
    return compare_img


# ================================
# 1. 获取当前脚本所在的绝对目录并读取 txt 数据文件
# ================================
script_dir = os.path.dirname(os.path.abspath(__file__))

# ★★★ 在这里修改你的 txt 文件名 ★★★
txt_filename = "experiment_data_20260502_nvidia.txt" 
txt_filepath = os.path.join(script_dir, txt_filename)

if not os.path.exists(txt_filepath):
    raise FileNotFoundError(f"找不到数据文件: {txt_filepath}")

with open(txt_filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# 用正则表达式提取不同密度的 CSV 数据块
match_10 = re.search(r'data_10_str\s*=\s*"""(.*?)"""', content, re.DOTALL)
match_15 = re.search(r'data_15_str\s*=\s*"""(.*?)"""', content, re.DOTALL)
match_20 = re.search(r'data_20_str\s*=\s*"""(.*?)"""', content, re.DOTALL)

if not (match_10 and match_15 and match_20):
    raise ValueError("无法从 txt 文件中提取到完整的数据，请确保它是用上一步的代码生成的正确格式！")

# ================================
# 2. 读取数据进入 pandas DataFrame
# ================================
df_10 = pd.read_csv(io.StringIO(match_10.group(1).strip()))
df_15 = pd.read_csv(io.StringIO(match_15.group(1).strip()))
df_20 = pd.read_csv(io.StringIO(match_20.group(1).strip()))

datasets = [("10% Obstacles", df_10), ("15% Obstacles", df_15), ("20% Obstacles", df_20)]

# ================================
# 3. 先按 compare-grid.py 原有逻辑生成右下角图片（内存），再设置主图样式
# ================================
compare_img = draw_grid_world_to_image()

plt.style.use('seaborn-v0_8-whitegrid')
colors = [ '#2ca02c', '#9467bd', '#ff7f0e','#1f77b4', '#d62728'] 

# ================================
# 4. 创建 2x2 Figure 和 Axes
# ================================
fig, axes = plt.subplots(2, 2, figsize=(24, 20), sharey=True) 
axes_flat = axes.flatten()

# fig.suptitle("Reduction Rates of MCCA-Path vs CPPF on Grid Maps", fontsize=35, fontweight='bold', y=0.99)

# ★★★ 定义次坐标轴格式化函数：将 1000 转换为 1k ★★★
def thousands_formatter(x, pos):
    if x == 0:
        return '0'
    return f'{int(x/1000)}k'

# 准备用于保存全局图例内容的变量
global_lines = []
global_labels = []

# ================================
# 5. 绘图与计算逻辑
# ================================
for i, (title, df) in enumerate(datasets):
    ax = axes_flat[i]
    ax.grid(False)
    
    # 计算降低率（保留原有计算逻辑）
    time_reduction = (df['B_time'] - df['A_time']) / df['B_time'] * 100
    blocks_reduction = (df['B_blocks'] - df['A_blocks']) / df['B_blocks'] * 100
    mst_reduction = (df['B_mst'] - df['A_mst']) / df['B_mst'] * 100
    
    # ===================== 主坐标轴(ax) 绘制 A_blocks/B_blocks =====================
    # 绘制主坐标轴折线
    line4 = ax.plot(df['grid_size'], df['A_blocks'], marker='o', label='CPPF Blocks', 
            color=colors[3], linewidth=6, markersize=15, linestyle='--')
    line5 = ax.plot(df['grid_size'], df['B_blocks'], marker='o', label='MCCA Blocks', 
            color=colors[4], linewidth=6, markersize=15, linestyle='--')

    # 主坐标轴X/Y轴设置
    ax.set_xlabel(f'Grid Size ({title})', fontsize=50, fontweight='bold')
    ax.set_ylabel('Blocks Count', fontsize=50, fontweight='bold', color='black')  # 主Y轴标签
    ax.tick_params(axis='both', labelsize=50)
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))  # 主Y轴：千位分隔符
    
    # ===================== 创建次坐标轴(ax_twin) 绘制三个降低率 =====================
    ax_twin = ax.twinx()
    ax_twin.grid(False)
    
    # 绘制次坐标轴折线
    line1 = ax_twin.plot(df['grid_size'], time_reduction, marker='s', label='Computation Time Reduction', 
            color=colors[0], linewidth=6, markersize=15, linestyle='-.')
    line2 = ax_twin.plot(df['grid_size'], blocks_reduction, marker='s', label='Blocks Number Reduction', 
            color=colors[1], linewidth=6, markersize=15, linestyle='-.')
    line3 = ax_twin.plot(df['grid_size'], mst_reduction, marker='s', label='Path Length Reduction', 
            color=colors[2], linewidth=6, markersize=15, linestyle='-.')
    
    # 次坐标轴Y轴设置 + 浅绿色样式
    ax_twin.set_ylabel('Reduction Rate (%)', fontsize=50, fontweight='bold', color='green')  # 次Y轴标签
    ax_twin.tick_params(axis='y', labelsize=50, labelcolor='green')  # 刻度文字绿色
    ax_twin.yaxis.set_major_formatter(PercentFormatter())  # 次Y轴：百分比格式
    # 次坐标轴右侧轴线设置为绿色
    ax_twin.spines['right'].set_color('green')

    # 将第一张图的线条和标签保存，用于在底部生成全局图例
    if i == 0:
        global_lines = line1 + line2 + line3 + line4 + line5
        global_labels = [l.get_label() for l in global_lines]

# ================================
# 6. 处理右下角并直接插入 compare-grid 绘图结果
# ================================
# 先执行 tight_layout 以确定好所有 2x2 子图的最终正确位置
# 加入 rect=[0, 0.08, 1, 1] 使得底部留出 8% 的空间给全局图例，防止重叠
plt.tight_layout(rect=[0, 0.08, 1, 1])

# 删除原有的第四个子图，彻底解除 sharey=True 的绑定
ax4 = axes_flat[3]
ax4.remove()

# ★★★ 核心修改部分：单个底部居中的全局图例，内部排成两列 ★★★

legend_kwargs = {
    'loc': 'lower center',
    'prop': {'size': 50, 'weight': 'bold'},
    'columnspacing': 0.6,
    'handletextpad': 0.25,
    'borderpad': 0.1,
    'labelspacing': 0.35,
    'frameon': False
}

# ncol=2 会把 5 个 legend 排成两列：
# 左列：前 3 个
# 右列：后 2 个
fig.legend(
    global_lines,
    global_labels,
    ncol=2,
    bbox_to_anchor=(0.5, -0.08),
    **legend_kwargs
)

# compare-grid 图片已经在内存中直接生成，不再检查/读取 compare-grid.png
img = compare_img

# 获取图片原始像素尺寸，计算图片宽高比
img_h, img_w = img.shape[:2]
img_aspect = img_w / img_h

# 获取当前 Figure 的尺寸，计算 Figure 宽高比
fig_w, fig_h = fig.get_size_inches()
fig_aspect = fig_w / fig_h

# ★★★ 在这里自由设置图片的悬浮排布属性 ★★★
# 注意由于图例在最底下，如果图片太靠下可能会和图例重叠，可以适当增加 img_bottom
img_left = 0.57    # 距离画布左侧边缘的比例 (0~1)
img_bottom = 0.09  # 距离画布底部边缘的比例 (0~1) 
img_width = 0.30   # 图片占画布总宽度的比例 (0~1)

# 结合 Figure 和 Image 的比例，自动推算占画布的高度比例，以保证图片不被拉伸
img_height = (img_width / img_aspect) * fig_aspect

# 使用自定义坐标和计算出的高度添加一个新的独立坐标轴
ax_img = fig.add_axes([img_left, img_bottom, img_width, img_height])

ax_img.imshow(img)

# 隐藏图片的边框和坐标轴刻度
ax_img.axis('off')

# ================================
# 7. 保存 PNG 和 PDF 文件
# ================================
output_png = os.path.join(script_dir, 'reduction_rates_2x2_with_bottom_legend.png')
output_pdf = os.path.join(script_dir, 'reduction_rates_2x2_with_bottom_legend.pdf')

fig.savefig(output_png, dpi=200, bbox_inches='tight')
fig.savefig(output_pdf, dpi=200, bbox_inches='tight')
plt.close(fig)

print(f"PNG Figure saved to: {output_png}")
print(f"PDF Figure saved to: {output_pdf}")
