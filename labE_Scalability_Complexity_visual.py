"""Lab E scalability/complexity visualization (v1.0).

Expected project layout:

MCCA-PRO/
├── LAB_DATA/
├── labE_Scalability_Complexity_visual_v1_0.py
└── ...

The plotting style/logic is kept consistent with
voronoi-Adapt-MST-labE-pro-visual.py.  The only workflow changes are:
1. automatically search MCCA-PRO/LAB_DATA recursively;
2. select the newest complete StressTest_DataLog_*.txt;
3. save the visualization beside that source log file.
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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


SPACE_MAPS = [25, 50, 75, 100, 125, 150, 175, 200]
CLUSTER_ROBOTS = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
DEFAULT_NUM_SEEDS = 50


def get_non_outlier_min_max(data_2d):
    """Helper function to calculate y-axis limits excluding outliers"""
    global_min = float('inf')
    global_max = float('-inf')
    for d in data_2d:
        q1 = np.percentile(d, 25)
        q3 = np.percentile(d, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        valid_data = [v for v in d if lower_bound <= v <= upper_bound]
        if valid_data:
            global_min = min(global_min, min(valid_data))
            global_max = max(global_max, max(valid_data))
    return global_min, global_max


def _get_lab_data_dir():
    """LAB_DATA is expected to be beside this visual script in MCCA-PRO."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lab_data_dir = os.path.join(script_dir, "LAB_DATA")
    if not os.path.isdir(lab_data_dir):
        raise FileNotFoundError(
            f"LAB_DATA folder not found. Expected location: {lab_data_dir}"
        )
    return lab_data_dir


def _timestamp_key(path):
    """Prefer the YYYYMMDD_HHMMSS timestamp embedded in the experiment filename."""
    name = os.path.basename(path)
    match = re.search(r"(\d{8}_\d{6})", name)
    timestamp = match.group(1) if match else ""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0
    return timestamp, mtime, name


def _parse_trial_data(log_file_path):
    """Read only per-trial lines, preserving the original Lab E data grouping."""
    with open(log_file_path, "r", encoding="utf-8") as f:
        content = f.read()

    seed_match = re.search(r"\((\d+)个随机种子\)", content)
    num_seeds = int(seed_match.group(1)) if seed_match else DEFAULT_NUM_SEEDS

    space_pattern = re.compile(
        r"\[\*\]\s*地图\s*(\d+)x\s*(\d+)\s*\|\s*Seed:\s*(-?\d+)\s*\|\s*耗时:\s*([0-9.]+)\s*秒"
    )
    cluster_pattern = re.compile(
        r"\[\*\]\s*机器人数量:\s*(\d+)\s*\|\s*Seed:\s*(-?\d+)\s*\|\s*耗时:\s*([0-9.]+)\s*秒"
    )

    space_dict = {ms: [] for ms in SPACE_MAPS}
    cluster_dict = {rn: [] for rn in CLUSTER_ROBOTS}

    for width, height, _seed, elapsed in space_pattern.findall(content):
        width = int(width)
        height = int(height)
        if width == height and width in space_dict:
            space_dict[width].append(float(elapsed))

    for robot_num, _seed, elapsed in cluster_pattern.findall(content):
        robot_num = int(robot_num)
        if robot_num in cluster_dict:
            cluster_dict[robot_num].append(float(elapsed))

    return space_dict, cluster_dict, num_seeds


def _is_complete_labE_log(log_file_path):
    try:
        space_dict, cluster_dict, num_seeds = _parse_trial_data(log_file_path)
    except (OSError, UnicodeError, ValueError):
        return False

    return (
        all(len(space_dict[ms]) == num_seeds for ms in SPACE_MAPS)
        and all(len(cluster_dict[rn]) == num_seeds for rn in CLUSTER_ROBOTS)
    )


def find_latest_labE_log():
    """Find the newest complete Lab E stress-test log under LAB_DATA."""
    lab_data_dir = _get_lab_data_dir()
    candidates = []

    for root, _dirs, files in os.walk(lab_data_dir):
        for filename in files:
            if filename.startswith("StressTest_DataLog_") and filename.endswith(".txt"):
                candidates.append(os.path.join(root, filename))

    if not candidates:
        raise FileNotFoundError(
            f"No StressTest_DataLog_*.txt found under: {lab_data_dir}"
        )

    candidates.sort(key=_timestamp_key, reverse=True)

    skipped = []
    for path in candidates:
        if _is_complete_labE_log(path):
            if skipped:
                print(
                    "Warning: the newest Lab E log is incomplete; "
                    "using the newest complete log instead."
                )
            return path
        skipped.append(path)

    raise ValueError(
        "Lab E log files were found, but none contains a complete set of "
        "spatial and swarm experiments."
    )


def plot_from_log(log_file_path=None):
    # Auto-select newest complete experiment when no explicit path is supplied.
    if log_file_path is None:
        log_file_path = find_latest_labE_log()

    log_file_path = os.path.abspath(log_file_path)
    base_dir = os.path.dirname(log_file_path)

    # 1. 读取TXT数据
    try:
        space_dict, cluster_dict, num_seeds = _parse_trial_data(log_file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {log_file_path}\nPlease check if the file exists.")
        return

    # 定义实验参数
    space_maps = SPACE_MAPS
    cluster_robots = CLUSTER_ROBOTS

    # 检查提取的数据量是否匹配
    expected_count = (len(space_maps) + len(cluster_robots)) * num_seeds
    actual_count = sum(len(v) for v in space_dict.values()) + sum(len(v) for v in cluster_dict.values())
    if actual_count != expected_count:
        print(f"Warning: Extracted data count ({actual_count}) does not match expected ({expected_count})!")
        return

    print(f"Loading newest Lab E data from: {log_file_path}")

    # 2. 数据切片与预处理
    space_data = [space_dict[ms] for ms in space_maps]
    cluster_data = [cluster_dict[rn] for rn in cluster_robots]

    space_means = [np.mean(d) for d in space_data]
    cluster_means = [np.mean(d) for d in cluster_data]

    # 3. 开始绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.1))


    x1 = np.array(space_maps)
    p1 = np.poly1d(np.polyfit(x1, np.array(space_means), 2))
    x1_fit = np.linspace(min(x1)-10, max(x1)+10, 100)
    ax1.plot(x1_fit, p1(x1_fit), '--', color='#c0392b', alpha=0.8, linewidth=4, label='Quadratic fit')

    # -------- Space Graph (Boxplot + Quadratic Fit) --------
    ax1.boxplot(space_data, positions=space_maps, widths=8, patch_artist=True, showfliers=False,
                boxprops=dict(facecolor='#fadbd8', color='#c0392b', alpha=0.8),
                medianprops=dict(color='#e74c3c', linewidth=2))

    # 【修改2】将标题、x轴、y轴字体加粗 (添加 fontweight='bold')
    # ax1.set_title('Spatial Scalability\n (Computation Time vs. Map Size)', fontsize=25, fontweight='bold', pad=15)
    ax1.set_xlabel('Map Size (L x L)', fontsize=25, fontweight='bold')
    ax1.set_ylabel('Computation Time (s)', fontsize=25, fontweight='bold')

    # 【补充】为了视觉统一，将刻度数字也加粗
    plt.setp(ax1.get_xticklabels(), fontweight='bold', size=20)
    plt.setp(ax1.get_yticklabels(), fontweight='bold', size=20)

    # 【修改1】彻底不显示背景网格线
    ax1.grid(False)

    # 调整纵坐标间距
    min_y1, max_y1 = get_non_outlier_min_max(space_data)
    y1_margin = (max_y1 - min_y1) * 0.1
    ax1.set_ylim(max(0, min_y1 - y1_margin), max_y1 + y1_margin)
    ax1.locator_params(axis='y', nbins=10)

    # 【修改3】将 legend 的字体加大并加粗
    ax1.legend(loc='upper left', prop={'size': 20, 'weight': 'bold'})

    x2 = np.array(cluster_robots)
    p2 = np.poly1d(np.polyfit(x2, np.array(cluster_means), 1))
    x2_fit = np.linspace(min(x2)-1, max(x2)+1, 100)
    ax2.plot(x2_fit, p2(x2_fit), '--', color='#2980b9', alpha=0.8, linewidth=4, label='Linear fit')

        # -------- Cluster Graph (Boxplot + Linear Fit) --------
    ax2.boxplot(cluster_data, positions=cluster_robots, widths=0.8, patch_artist=True, showfliers=False,
                boxprops=dict(facecolor='#d4e6f1', color='#2980b9', alpha=0.8),
                medianprops=dict(color='#3498db', linewidth=2))

    # 【修改2】将标题、x轴、y轴字体加粗
    # ax2.set_title('Swarm Scalability\n (Computation Time vs. Robot Num)', fontsize=25, fontweight='bold', pad=15)
    ax2.set_xlabel('Number of Robots (N)', fontsize=25, fontweight='bold')
    ax2.set_ylabel('Computation Time (s)', fontsize=25, fontweight='bold')

    # 【补充】为了视觉统一，将刻度数字也加粗
    plt.setp(ax2.get_xticklabels(), fontweight='bold', size=20)
    plt.setp(ax2.get_yticklabels(), fontweight='bold', size=20)

    # 【修改1】彻底不显示背景网格线
    ax2.grid(False)

    # 调整纵坐标间距
    min_y2, max_y2 = get_non_outlier_min_max(cluster_data)
    y2_margin = (max_y2 - min_y2) * 0.1
    ax2.set_ylim(max(0, min_y2 - y2_margin), max_y2 + y2_margin)
    ax2.locator_params(axis='y', nbins=10)

    # 【修改3】将 legend 的字体加大并加粗
    ax2.legend(loc='upper left', prop={'size': 20, 'weight': 'bold'})

    plt.tight_layout()
    output_img = os.path.join(base_dir, "Final_Performance_Report_Boxplot_EN.png")
    output_pdf = os.path.join(base_dir, "Final_Performance_Report_Boxplot_EN.pdf")

    plt.savefig(output_img, dpi=200, bbox_inches='tight')
    plt.savefig(output_pdf, bbox_inches='tight')

    print(f"Success! PNG saved to: {output_img}")
    print(f"Success! PDF saved to: {output_pdf}")


if __name__ == "__main__":
    plot_from_log()
