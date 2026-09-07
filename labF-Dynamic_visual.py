import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter1d

# ==========================================
# 【需要修改这里】填入实验代码生成的 CSV 文件名
# ==========================================
TARGET_CSV = "experiment_EMA_Simulation_20260825_162604.csv"
# ==========================================

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


def run_visualization():
    output_dir = os.path.join('.', "labF-dynamic")
    csv_path = os.path.join(output_dir, TARGET_CSV)

    if not os.path.exists(csv_path):
        print(f"❌ 找不到数据文件: {csv_path}")
        print("请确认实验代码已经成功运行，并检查本代码中 TARGET_CSV 变量是否配置正确。")
        return

    print(f"📊 正在读取实验数据: {csv_path}")
    df = pd.read_csv(csv_path)

    max_T = len(df)
    robot_num = 3

    # 提取数组供绘图使用（读取逻辑保持不变，仅将 MST_Length 改为 Path_Length）
    history_sampled_C_abs = df[['Sensed_C_0', 'Sensed_C_1', 'Sensed_C_2']].values
    history_weights = df[['Weight_0', 'Weight_1', 'Weight_2']].values
    history_path_lengths = df[['Path_Length_0', 'Path_Length_1', 'Path_Length_2']].values

    # ================= Plotting =================
    # 保持紧凑版面并共享X轴
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # 三个机器人使用不同颜色
    colors = ['#FF6B6B', "#E2AB12", '#45B7D1']

    # 三个机器人使用不同线型
    linestyles = ['-', '--', '-.']

    # 三个机器人使用不同 marker 样式
    markers = ['o', '*', '^']

    labels = ['Robot A', 'Robot B', 'Robot C']

    # 控制 marker 的显示间隔，避免点太密
    mark_every = max(1, max_T // 25)

    # 【放大字体】全局主标题
    # fig.suptitle(
    #     'Full Lifecycle Validation: Sensing, Estimation, and Execution',
    #     fontsize=22,
    #     fontweight='bold',
    #     y=0.98
    # )

    # --- Figure 1: Smoothed Noisy Sensor Input ---
    for i in range(robot_num):
        smoothed_noisy_data = gaussian_filter1d(
            history_sampled_C_abs[:, i],
            sigma=1.2
        )

        ax1.plot(
            range(max_T),
            smoothed_noisy_data,
            color=colors[i],
            linestyle=linestyles[i],
            marker=markers[i],
            markevery=mark_every,
            markersize=10,
            markerfacecolor='white',
            markeredgewidth=1.4,
            linewidth=3.5,
            alpha=0.85,
            label=labels[i]
        )

    # 【放大字体】Y轴标签和刻度
    ax1.set_ylabel('(a) Coverage \nThroughput(cm/s)', fontsize=18, fontweight='bold')
    ax1.axvline(x=50, color='gray', linestyle=':', linewidth=2)
    ax1.axvline(x=100, color='gray', linestyle=':', linewidth=2)
    plt.setp(ax1.get_yticklabels(), fontweight='bold', fontsize=14)
    ax1.grid(False)

    # --- Figure 2: Weight Estimation via EMA ---
    for i in range(robot_num):
        ax2.plot(
            range(max_T),
            history_weights[:, i],
            color=colors[i],
            linestyle=linestyles[i],
            marker=markers[i],
            markevery=mark_every,
            markersize=10,
            markerfacecolor='white',
            markeredgewidth=1.4,
            linewidth=3.5,
            alpha=0.85
        )

    # 【放大字体】Y轴标签和刻度
    ax2.set_ylabel('(b) EMA\nWeight Ratio', fontsize=18, fontweight='bold')
    ax2.axvline(x=50, color='gray', linestyle=':', linewidth=2)
    ax2.axvline(x=100, color='gray', linestyle=':', linewidth=2)

    # 【放大字体】注释文本
    ax2.annotate(
        'Initial Equal Weights',
        xy=(0, 0.33),
        xytext=(5, 0.45),
        arrowprops=dict(
            facecolor='black',
            shrink=0.05,
            width=1.5,
            headwidth=6
        ),
        fontweight='bold',
        fontsize=15
    )

    plt.setp(ax2.get_yticklabels(), fontweight='bold', fontsize=14)
    ax2.grid(False)

    # --- Figure 3: Physical Execution (Path Length) ---
    for i in range(robot_num):
        ax3.plot(
            range(max_T),
            history_path_lengths[:, i],
            color=colors[i],
            linestyle=linestyles[i],
            marker=markers[i],
            markevery=mark_every,
            markersize=10,
            markerfacecolor='white',
            markeredgewidth=1.4,
            linewidth=3.5,
            alpha=0.85
        )

    # 【放大字体】X、Y轴标签和刻度
    ax3.set_xlabel('Time Steps', fontsize=18, fontweight='bold')
    ax3.set_ylabel('(c) Path Length', fontsize=18, fontweight='bold')
    ax3.axvline(x=50, color='gray', linestyle=':', linewidth=2)
    ax3.axvline(x=100, color='gray', linestyle=':', linewidth=2)

    plt.setp(ax3.get_xticklabels(), fontweight='bold', fontsize=14)
    plt.setp(ax3.get_yticklabels(), fontweight='bold', fontsize=14)
    ax3.grid(False)

    # 【放大字体】全局统一图例
    handles, legend_labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc='upper center',
        ncol=3,
        bbox_to_anchor=(0.5, 0.96),
        frameon=False,
        prop={'size': 16, 'weight': 'bold'},
        labelspacing=0.2,
        borderaxespad=0.0
    )

    # 【微调间距】保持宽度不变，缩小 legend 与主图之间的空白
    plt.subplots_adjust(
        hspace=0.15,
        top=0.92,
        bottom=0.1,
        left=0.12,
        right=0.95
    )

    base_name = 'Full_Lifecycle_Validation_English_Compact_Bold_LineStyle_Marker'
    save_path_png = os.path.join(output_dir, f'{base_name}.png')
    save_path_pdf = os.path.join(output_dir, f'{base_name}.pdf')

    # 同目录同时输出 PNG 与 PDF；pad_inches=0 去掉 legend 顶部额外白边
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.savefig(save_path_pdf, bbox_inches='tight', pad_inches=0)
    print(f"✅ 成功！PNG 图表已保存至: {save_path_png}")
    print(f"✅ 成功！PDF 图表已保存至: {save_path_pdf}")


if __name__ == '__main__':
    run_visualization()
