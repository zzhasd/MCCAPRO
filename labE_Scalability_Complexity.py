"""Lab E: Scalability / Complexity stress test for the current mainline planner.

Keep the experiment workflow consistent with voronoi-Adapt-MST-labE-pro.py:
1) 50 fixed random seeds;
2) Experiment 1: fix 5 robots and vary the map side length 25..200;
3) Experiment 2: fix a 100x100 map and vary the robot count 3..21;
4) Record end-to-end runtime for each trial and mean runtime for each condition;
5) Write TXT logs with the same structure and Final_Performance_Report_Boxplot.png.

This file implements no MCPP algorithm steps; it only loads mainline, constructs the solver, and calls solve(),
Collect experiment runtimes and produce statistical plots.
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

# Stress tests typically run on headless servers/terminals; avoid GUI initialization overhead and backend issues.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

# Use the reference script font settings; matplotlib falls back automatically if SimHei is unavailable.
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# -----------------------------------------------------------------------------
# Experiment configuration: match voronoi-Adapt-MST-labE-pro.py
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

# mainline manages its own algorithm parameters; the experiment passes only the experimental variables and obstacle ratio.
# To select a specific mainline version, set the environment variable:
#   MCPP_MAINLINE=/absolute/path/to/mainline_xxx.py
MAINLINE_ENV = "MCPP_MAINLINE"


# -----------------------------------------------------------------------------
# mainline loading: resolve experiment dependencies only; no algorithm logic
# -----------------------------------------------------------------------------
def _find_mainline_path(script_dir: Path) -> Path:
    env_path = os.environ.get(MAINLINE_ENV)
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{MAINLINE_ENV} points to a file that does not exist: {path}")
        return path

    exact_candidates = (
        script_dir / "mainline.py",
        script_dir / "mainline_tile_first_v2_3_2.py",
    )
    for path in exact_candidates:
        if path.is_file():
            return path.resolve()

    # Support mainline filenames with version numbers, timestamps, or parentheses (such as the supplied file).
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
        "No mainline Python file found. Place mainline.py / mainline*.py beside this experiment script, "
        f"or set the environment variable {MAINLINE_ENV}=<absolute path to the mainline file>."
    )


def _load_mainline(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("mcpp_mainline_for_labE", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load mainline: {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclass and other runtime mechanisms look up module namespaces through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_solver_class(module: ModuleType) -> Type:
    # The current mainline exposes TileFirstMCPP and retains the FACTMCCA alias.
    for name in ("TileFirstMCPP", "FACTMCCA"):
        solver_cls = getattr(module, name, None)
        if solver_cls is not None:
            return solver_cls
    raise AttributeError("mainline does not contain TileFirstMCPP or FACTMCCA")


# -----------------------------------------------------------------------------
# Single experiment: only construct the mainline solver and call solve()
# -----------------------------------------------------------------------------
def _run_one(solver_cls: Type, map_size: int, robot_num: int, seed: int) -> Tuple[float, dict]:
    """Return end-to-end runtime consistent with the reference script, along with the mainline solve() result.

    The reference script starts timing before solver construction, so elapsed also includes map generation/initialization.
    mainline result total_time measures only solve() itself and can be checked separately if needed,
    The main Lab E log still uses elapsed for consistent measurement.
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
# Plotting: preserve the reference structure of two boxplots with fitted curves
# -----------------------------------------------------------------------------
def _save_final_boxplot(
    script_dir: Path,
    space_maps: Sequence[int],
    space_times_dict: Dict[int, List[float]],
    cluster_robots: Sequence[int],
    cluster_times_dict: Dict[int, List[float]],
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # -------- Spatial scaling plot --------
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
        label="Fitted mean curve (O(N^2))",
    )

    ax1.set_title("Algorithm spatial scalability (Computation vs. Map Size)", fontsize=12)
    ax1.set_xlabel("Map Size (N x N)", fontsize=11)
    ax1.set_ylabel("Compute Time (Seconds)", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.7)
    ax1.legend()

    # -------- Fleet scaling plot --------
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
        label="Linear fit to means (O(K))",
    )

    ax2.set_title("Algorithm fleet scalability (Computation vs. Robot Num)", fontsize=12)
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
# Lab E Main experiment workflow
# -----------------------------------------------------------------------------
def run_stress_test() -> None:
    seeds = list(SEEDS)
    script_dir = Path(__file__).resolve().parent

    mainline_path = _find_mainline_path(script_dir)
    mainline = _load_mainline(mainline_path)
    solver_cls = _resolve_solver_class(mainline)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Experiment output directory: MCCA-PRO/LAB_DATA/labE_<timestamp>
    output_dir = script_dir / "LAB_DATA" / f"labE_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = output_dir / f"StressTest_DataLog_{timestamp}.txt"

    print(f"📂 Experiment plots and TXT data will be saved automatically to: {output_dir}")
    print(f"🔗 Mainline: {mainline_path.name}")

    with log_file_path.open("w", encoding="utf-8", buffering=1024 * 1024) as f:
        f.write("=" * 60 + "\n")
        f.write(f" mCPP Spatial and fleet scalability stress test - data archive ({len(seeds)} random seeds)\n")
        f.write(f" Test seed list: {seeds}\n")
        f.write("=" * 60 + "\n\n")

        # ---------------- Experiment 1: spatial scalability test ----------------
        space_maps = list(SPACE_MAPS)
        space_times_dict: Dict[int, List[float]] = {ms: [] for ms in space_maps}

        title1 = f"▶ Experiment 1: spatial scalability stress test (fixed robots={SPACE_FIXED_ROBOTS})"
        print("\n" + "=" * 60 + "\n" + title1 + "\n" + "=" * 60)
        f.write("=" * 60 + "\n" + title1 + "\n" + "=" * 60 + "\n")

        for ms in space_maps:
            for seed in seeds:
                elapsed, _ = _run_one(solver_cls, ms, SPACE_FIXED_ROBOTS, seed)
                space_times_dict[ms].append(elapsed)

                res_str = f"[*] Map {ms:>3}x{ms:<3} | Seed: {seed:>5} | Elapsed: {elapsed:.3f} s"
                print(res_str)
                f.write(res_str + "\n")

            avg_time = float(np.mean(space_times_dict[ms]))
            f.write(
                f"--- Map {ms}x{ms} complete; mean runtime over {len(seeds)} trials: "
                f"{avg_time:.3f} s ---\n\n"
            )
            # Flush only after each experiment condition to avoid per-trial flush I/O overhead.
            f.flush()

        # ---------------- Experiment 2: fleet scalability test ----------------
        cluster_robots = list(CLUSTER_ROBOTS)
        cluster_times_dict: Dict[int, List[float]] = {rn: [] for rn in cluster_robots}
        fixed_map = CLUSTER_FIXED_MAP

        title2 = f"▶ Experiment 2: fleet scalability stress test (fixed map={fixed_map}x{fixed_map})"
        print("\n" + "=" * 60 + "\n" + title2 + "\n" + "=" * 60)
        f.write("\n" + "=" * 60 + "\n" + title2 + "\n" + "=" * 60 + "\n")

        for rn in cluster_robots:
            for seed in seeds:
                elapsed, _ = _run_one(solver_cls, fixed_map, rn, seed)
                cluster_times_dict[rn].append(elapsed)

                res_str = f"[*] Robot count: {rn:>2} | Seed: {seed:>5} | Elapsed: {elapsed:.3f} s"
                print(res_str)
                f.write(res_str + "\n")

            avg_time = float(np.mean(cluster_times_dict[rn]))
            f.write(
                f"--- Robot count {rn} complete; mean runtime over {len(seeds)} trials: "
                f"{avg_time:.3f} s ---\n\n"
            )
            f.flush()

        f.write("\n✅ All stress tests complete; figures and data archived.\n")

    _save_final_boxplot(
        script_dir,
        space_maps,
        space_times_dict,
        cluster_robots,
        cluster_times_dict,
    )


if __name__ == "__main__":
    run_stress_test()
