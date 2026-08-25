"""Lab E: Scalability / Complexity stress test for the current mainline planner.

实验流程与 voronoi-Adapt-MST-labE-pro.py 保持一致：
1) 50 个固定随机种子；
2) 实验一：固定 5 台机器人，地图边长 25..200；
3) 实验二：固定 100x100 地图，机器人数量 3..21；
4) 每次记录端到端耗时，按实验条件记录平均耗时；
5) 输出同结构 TXT 日志与 Final_Performance_Report_Boxplot.png。

本文件不实现任何 MCPP 算法步骤，只负责：加载 mainline、创建 solver、调用 solve()、
采集实验时间并输出统计图表。
"""
from __future__ import annotations

import datetime
import importlib.util
import os
import sys
from pathlib import Path
import time
from types import ModuleType
from typing import Dict, List, Sequence, Tuple, Type
import warnings

import matplotlib

# 压力测试通常在无图形界面的服务器/终端运行；避免 GUI 初始化开销与后端问题。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

# 与参考脚本一致的中文字体设置；系统无 SimHei 时 matplotlib 会自动回退。
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# -----------------------------------------------------------------------------
# 实验配置：保持与 voronoi-Adapt-MST-labE-pro.py 一致
# -----------------------------------------------------------------------------
SEEDS: Tuple[int, ...] = (
    42, 100, 256, 512, 1024, 2048, 4096, 8192, 12345, 99999,
    7, 13, 29, 63, 127, 255, 511, 777, 1337, 2024,
    3141, 4097, 5003, 6666, 7001, 8191, 9001, 10007, 12011, 15013,
    17021, 19031, 21001, 23003, 25013, 27011, 29009, 31013, 33023, 35023,
    37019, 39041, 41047, 43051, 45053, 47057, 49069, 51071, 53087, 55073,
)

SPACE_MAPS: Tuple[int, ...] = (25, 50, 75, 100, 125, 150, 175, 200)
SPACE_FIXED_ROBOTS = 5

CLUSTER_ROBOTS: Tuple[int, ...] = (3, 5, 7, 9, 11, 13, 15, 17, 19, 21)
CLUSTER_FIXED_MAP = 100

OBSTACLE_RATIO = 0.10

# mainline 自己负责算法参数；实验脚本只显式传入实验变量和障碍比例。
# 如需固定某个 mainline 版本，可设置环境变量：
#   MCPP_MAINLINE=/absolute/path/to/mainline_xxx.py
MAINLINE_ENV = "MCPP_MAINLINE"


# -----------------------------------------------------------------------------
# mainline 加载：仅做实验依赖解析，不包含算法逻辑
# -----------------------------------------------------------------------------
def _find_mainline_path(script_dir: Path) -> Path:
    env_path = os.environ.get(MAINLINE_ENV)
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{MAINLINE_ENV} 指向的文件不存在: {path}")
        return path

    exact_candidates = (
        script_dir / "mainline.py",
        script_dir / "mainline_tile_first_v2_3_2.py",
    )
    for path in exact_candidates:
        if path.is_file():
            return path.resolve()

    # 兼容带版本号/时间戳/括号的 mainline 文件名（例如本次提供的文件）。
    candidates = sorted(
        (
            p for p in script_dir.glob("mainline*.py")
            if p.is_file() and p.name != Path(__file__).name
        ),
        key=lambda p: (p.stat().st_mtime_ns, p.name),
        reverse=True,
    )
    if candidates:
        return candidates[0].resolve()

    raise FileNotFoundError(
        "未找到 mainline Python 文件。请将 mainline.py / mainline*.py 放在本实验脚本同目录，"
        f"或设置环境变量 {MAINLINE_ENV}=<mainline文件绝对路径>。"
    )


def _load_mainline(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("mcpp_mainline_for_labE", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 mainline: {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclass 等运行时机制会通过 sys.modules 查找模块命名空间。
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_solver_class(module: ModuleType) -> Type:
    # 当前 mainline 的公开类名为 TileFirstMCPP，并保留 FACTMCCA 别名。
    for name in ("TileFirstMCPP", "FACTMCCA"):
        solver_cls = getattr(module, name, None)
        if solver_cls is not None:
            return solver_cls
    raise AttributeError("mainline 中未找到 TileFirstMCPP 或 FACTMCCA")


# -----------------------------------------------------------------------------
# 单次实验：只创建 mainline solver 并调用 solve()
# -----------------------------------------------------------------------------
def _run_one(solver_cls: Type, map_size: int, robot_num: int, seed: int) -> Tuple[float, dict]:
    """返回与参考脚本口径一致的端到端耗时，以及 mainline 的 solve() 结果。

    参考脚本的计时从 solver 构造之前开始，因此这里也把地图生成/初始化包含在 elapsed 中。
    mainline result 中的 total_time 仅表示 solve() 自身时间，可供需要时进一步核查，
    但 Lab E 主日志仍使用 elapsed，保持实验口径一致。
    """
    started = time.perf_counter()
    solver = solver_cls(
        map_shape=map_size,
        robot_num=robot_num,
        obstacle_ratio=OBSTACLE_RATIO,
        seed=seed,
    )
    result = solver.solve()
    elapsed = time.perf_counter() - started
    return float(elapsed), result


# -----------------------------------------------------------------------------
# 绘图：严格沿用参考脚本的两组箱线图 + 拟合曲线结构
# -----------------------------------------------------------------------------
def _save_final_boxplot(
    script_dir: Path,
    space_maps: Sequence[int],
    space_times_dict: Dict[int, List[float]],
    cluster_robots: Sequence[int],
    cluster_times_dict: Dict[int, List[float]],
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # -------- 空间图 --------
    space_data = [space_times_dict[ms] for ms in space_maps]
    space_means = [np.mean(times) for times in space_data]

    ax1.boxplot(
        space_data,
        positions=space_maps,
        widths=8,
        patch_artist=True,
        boxprops=dict(facecolor="#fadbd8", color="#c0392b", alpha=0.7),
        medianprops=dict(color="#e74c3c", linewidth=2),
    )

    x1 = np.asarray(space_maps, dtype=float)
    y1 = np.asarray(space_means, dtype=float)
    z1 = np.polyfit(x1, y1, 2)
    p1 = np.poly1d(z1)
    x1_fit = np.linspace(min(x1) - 10, max(x1) + 10, 100)
    ax1.plot(
        x1_fit,
        p1(x1_fit),
        "--",
        color="#c0392b",
        alpha=0.8,
        linewidth=2,
        label="均值拟合曲线 (O(N^2))",
    )

    ax1.set_title("算法空间扩展性 (Computation vs. Map Size)", fontsize=12)
    ax1.set_xlabel("Map Size (N x N)", fontsize=11)
    ax1.set_ylabel("Compute Time (Seconds)", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.7)
    ax1.legend()

    # -------- 集群图 --------
    cluster_data = [cluster_times_dict[rn] for rn in cluster_robots]
    cluster_means = [np.mean(times) for times in cluster_data]

    ax2.boxplot(
        cluster_data,
        positions=cluster_robots,
        widths=0.8,
        patch_artist=True,
        boxprops=dict(facecolor="#d4e6f1", color="#2980b9", alpha=0.7),
        medianprops=dict(color="#3498db", linewidth=2),
    )

    x2 = np.asarray(cluster_robots, dtype=float)
    y2 = np.asarray(cluster_means, dtype=float)
    z2 = np.polyfit(x2, y2, 1)
    p2 = np.poly1d(z2)
    x2_fit = np.linspace(min(x2) - 1, max(x2) + 1, 100)
    ax2.plot(
        x2_fit,
        p2(x2_fit),
        "--",
        color="#2980b9",
        alpha=0.8,
        linewidth=2,
        label="均值线性拟合 (O(K))",
    )

    ax2.set_title("算法集群扩展性 (Computation vs. Robot Num)", fontsize=12)
    ax2.set_xlabel("Number of Robots (K)", fontsize=11)
    ax2.set_ylabel("Compute Time (Seconds)", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(
        script_dir / "Final_Performance_Report_Boxplot.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


# -----------------------------------------------------------------------------
# Lab E 主实验流程
# -----------------------------------------------------------------------------
def run_stress_test() -> None:
    seeds = list(SEEDS)
    script_dir = Path(__file__).resolve().parent

    mainline_path = _find_mainline_path(script_dir)
    mainline = _load_mainline(mainline_path)
    solver_cls = _resolve_solver_class(mainline)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 实验输出目录：MCCA-PRO/LAB_DATA/labE_时间戳
    output_dir = script_dir / "LAB_DATA" / f"labE_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = output_dir / f"StressTest_DataLog_{timestamp}.txt"

    print(f"📂 实验结果图表和TXT数据将自动保存在目录: {output_dir}")
    print(f"🔗 Mainline: {mainline_path.name}")

    with log_file_path.open("w", encoding="utf-8", buffering=1024 * 1024) as f:
        f.write("=" * 60 + "\n")
        f.write(f" mCPP 时空扩展性压力测试 - 数据留档 ({len(seeds)}个随机种子)\n")
        f.write(f" 测试种子列表: {seeds}\n")
        f.write("=" * 60 + "\n\n")

        # ---------------- 实验一：空间扩展性测试 ----------------
        space_maps = list(SPACE_MAPS)
        space_times_dict: Dict[int, List[float]] = {ms: [] for ms in space_maps}

        title1 = f"▶ 实验一：空间扩展性压力测试 (固定机器人={SPACE_FIXED_ROBOTS})"
        print("\n" + "=" * 60 + "\n" + title1 + "\n" + "=" * 60)
        f.write("=" * 60 + "\n" + title1 + "\n" + "=" * 60 + "\n")

        for ms in space_maps:
            for seed in seeds:
                elapsed, _ = _run_one(solver_cls, ms, SPACE_FIXED_ROBOTS, seed)
                space_times_dict[ms].append(elapsed)

                res_str = f"[*] 地图 {ms:>3}x{ms:<3} | Seed: {seed:>5} | 耗时: {elapsed:.3f} 秒"
                print(res_str)
                f.write(res_str + "\n")

            avg_time = float(np.mean(space_times_dict[ms]))
            f.write(
                f"--- 地图 {ms}x{ms} 测试完成，{len(seeds)}次平均耗时: "
                f"{avg_time:.3f} 秒 ---\n\n"
            )
            # 只在每个实验条件完成后刷盘，避免每次 trial 都 flush 的 I/O 开销。
            f.flush()

        # ---------------- 实验二：集群扩展性测试 ----------------
        cluster_robots = list(CLUSTER_ROBOTS)
        cluster_times_dict: Dict[int, List[float]] = {rn: [] for rn in cluster_robots}
        fixed_map = CLUSTER_FIXED_MAP

        title2 = f"▶ 实验二：集群扩展性压力测试 (固定地图={fixed_map}x{fixed_map})"
        print("\n" + "=" * 60 + "\n" + title2 + "\n" + "=" * 60)
        f.write("\n" + "=" * 60 + "\n" + title2 + "\n" + "=" * 60 + "\n")

        for rn in cluster_robots:
            for seed in seeds:
                elapsed, _ = _run_one(solver_cls, fixed_map, rn, seed)
                cluster_times_dict[rn].append(elapsed)

                res_str = f"[*] 机器人数量: {rn:>2} | Seed: {seed:>5} | 耗时: {elapsed:.3f} 秒"
                print(res_str)
                f.write(res_str + "\n")

            avg_time = float(np.mean(cluster_times_dict[rn]))
            f.write(
                f"--- 机器人数量 {rn} 测试完成，{len(seeds)}次平均耗时: "
                f"{avg_time:.3f} 秒 ---\n\n"
            )
            f.flush()

        f.write("\n✅ 所有压力测试跑完，图像与数据均已存档。\n")

    _save_final_boxplot(
        script_dir,
        space_maps,
        space_times_dict,
        cluster_robots,
        cluster_times_dict,
    )


if __name__ == "__main__":
    run_stress_test()
