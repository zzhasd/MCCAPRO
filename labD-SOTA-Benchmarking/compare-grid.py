import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import matplotlib.lines as mlines  # 引入 mlines 用于创建自定义图例句柄

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


def draw_grid_world():
    # 1. 修正画布比例：宽度8，高度10 (5*2)，使其与数据比例一致
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    
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
    draw_single_grid(axes[0], boxes_1, lines_1, "CPPF")
    draw_single_grid(axes[1], boxes_2, lines_2, "MCCA")

    # 1. 设置外边距 pad 为 0 去除四周白边，设置 h_pad 拉开上下子图的间距
    # h_pad 的数值可以根据你的喜好调整，比如 1.5, 2.0, 3.0 等
    plt.tight_layout(pad=0, h_pad=0.5) 
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'compare-grid.png')
    
    # 2. 保存时依然强制去除多余留白
    fig.savefig(output_path, dpi=200, bbox_inches='tight', pad_inches=0)

if __name__ == "__main__":
    draw_grid_world()