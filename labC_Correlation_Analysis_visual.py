"""Lab C correlation-analysis visualization (v1.0).

Expected project layout:

MCCA-PRO/
├── LAB_DATA/
├── labC_Correlation_Analysis_visual_v1_0.py
└── ...

The plotting style/logic is kept consistent with
voronoi-Adapt-MST-labC-visual.py.  The workflow changes are limited to:
1. automatically search MCCA-PRO/LAB_DATA recursively;
2. select the newest experiment_C_Correlation-Analysis_*.csv;
3. recognize current mainline Actual_TSP_Ratio semantics while retaining
   compatibility with historical Actual_MST_Ratio CSV files.
"""

import sys
import os
import csv
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')

# Set font for visualization
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_scatter_and_fit(ax, x_data, y_data, title, y_label, color, corr_val):
    ax.scatter(x_data, y_data, color=color, alpha=0.6, edgecolors='k', s=50, label='Experimental samples')
    # Linear fitting
    z = np.polyfit(x_data, y_data, 1)
    p = np.poly1d(z)
    fit_x = np.linspace(min(x_data), max(x_data), 100)
    ax.plot(fit_x, p(fit_x), color='red', linestyle='--', linewidth=2, label=f'Linear regression\n(y={z[0]:.4f}x+{z[1]:.4f})')
    # Ideal line y=x
    ax.plot([0, max(x_data)], [0, max(x_data)], color='gray', linestyle=':', label='Ideal line (y=x)')

    # 【修改2】将标题、x轴、y轴字体加粗 (添加 fontweight='bold')，并适当放大了字号
    ax.set_title(f'Pearson Correlation: {corr_val:.4f}', fontsize=25, fontweight='bold')
    ax.set_xlabel('Target Weight Ratio', fontsize=25, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=25, fontweight='bold')

    # 【补充修改】将坐标轴的刻度数字也加粗，保持整体视觉统一
    plt.setp(ax.get_xticklabels(), fontsize=20, fontweight='bold')
    plt.setp(ax.get_yticklabels(), fontsize=20, fontweight='bold')

    # 【修改1】彻底不显示背景网格线
    ax.grid(False)

    # 【修改3】将 legend 的字体加大并加粗 (添加 prop={'size': 12, 'weight': 'bold'})
    ax.legend(
        loc='upper left',
        prop={'size': 20, 'weight': 'bold'},
        labelspacing=0.15,
        handlelength=1.2,
        handletextpad=0.25,
        borderpad=0.1,
        borderaxespad=0.1,
        frameon=False
    )


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


def _read_header(csv_file):
    with open(csv_file, mode='r', newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        return next(reader, [])


def _recognized_metric(csv_file):
    """Return TSP or MST based on the actual third-column header."""
    try:
        header = [cell.strip() for cell in _read_header(csv_file)]
    except (OSError, UnicodeError):
        return None

    if len(header) < 3:
        return None
    if header[:2] != ['Target_Weight_Ratio', 'Actual_Area_Ratio']:
        return None
    if header[2] == 'Actual_TSP_Ratio':
        return 'TSP'
    if header[2] == 'Actual_MST_Ratio':
        return 'MST'
    return None


def find_latest_labC_csv():
    """Find newest compatible Lab C CSV under LAB_DATA."""
    lab_data_dir = _get_lab_data_dir()
    candidates = []

    for root, _dirs, files in os.walk(lab_data_dir):
        for filename in files:
            if filename.startswith('experiment_C_Correlation-Analysis_') and filename.endswith('.csv'):
                candidates.append(os.path.join(root, filename))

    if not candidates:
        raise FileNotFoundError(
            f"No experiment_C_Correlation-Analysis_*.csv found under: {lab_data_dir}"
        )

    candidates.sort(key=_timestamp_key, reverse=True)

    for path in candidates:
        if _recognized_metric(path) is not None:
            return path

    raise ValueError(
        "Lab C CSV files were found, but none has a supported header: "
        "Target_Weight_Ratio, Actual_Area_Ratio, Actual_TSP_Ratio/Actual_MST_Ratio"
    )


def run_visualization(csv_file=None):
    if csv_file is None:
        csv_file = find_latest_labC_csv()

    csv_file = os.path.abspath(csv_file)
    if not os.path.exists(csv_file):
        print(f"❌ Error: File '{csv_file}' not found in the current directory.")
        return

    metric_kind = _recognized_metric(csv_file)
    if metric_kind is None:
        print(
            "❌ Error: unsupported CSV header. Expected third column "
            "Actual_TSP_Ratio (current mainline) or Actual_MST_Ratio (legacy)."
        )
        return

    print(f"\n📂 Loading data from {csv_file}...")

    # Load CSV data (skipping the header row)
    data = np.genfromtxt(csv_file, delimiter=',', skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    X = data[:, 0]
    Y_Area = data[:, 1]
    Y_Path = data[:, 2]

    print(f"📦 Total data pairs loaded: {len(X)}")

    # Calculate Pearson correlations
    corr_area, p_area = pearsonr(X, Y_Area)
    corr_path, p_path = pearsonr(X, Y_Path)

    if metric_kind == 'TSP':
        path_title = 'Weight vs Path Length'
        path_ylabel = 'Actual Path Ratio'
        path_print = 'Weight vs Path Length'
    else:
        path_title = 'Weight vs MST Length'
        path_ylabel = 'Actual MST Ratio'
        path_print = 'Weight vs MST Length'

    print(f"\n📊 [Statistical Results]")
    print(f" -> Weight vs Partition Area | Pearson Correlation: {corr_area:.4f} (p-value: {p_area:.2e})")
    print(f" -> {path_print} | Pearson Correlation: {corr_path:.4f} (p-value: {p_path:.2e})\n")

    # ================= 绘制散点图与拟合回归线 =================
    print("📈 Generating plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

    # Plot 1: Weight vs Area
    plot_scatter_and_fit(ax1, X, Y_Area, 'Weight vs Partition Area', 'Actual Area Ratio', '#3498db', corr_area)

    # Plot 2: Weight vs final path metric (TSP for current mainline; MST for legacy CSV)
    plot_scatter_and_fit(ax2, X, Y_Path, path_title, path_ylabel, '#2ecc71', corr_path)

    plt.tight_layout()

    # Save the output figure using the CSV filename as base
    output_base_name = os.path.splitext(csv_file)[0]

    output_img_name = output_base_name + '.png'
    output_pdf_name = output_base_name + '.pdf'

    plt.savefig(output_img_name, dpi=200, bbox_inches='tight')
    plt.savefig(output_pdf_name, bbox_inches='tight')

    print(f"✅ PNG visualization saved to '{output_img_name}'")
    print(f"✅ PDF visualization saved to '{output_pdf_name}'")

    plt.show()


if __name__ == '__main__':
    # With no argument, automatically load the newest compatible Lab C CSV
    # from MCCA-PRO/LAB_DATA.  An explicit CSV path can still be passed if needed.
    full_csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_visualization(full_csv_path)
