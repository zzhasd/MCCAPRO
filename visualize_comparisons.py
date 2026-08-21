"""Render cached before/after partition, tile, and route comparisons.

The expensive planner is executed only when a compatible ``.npz`` cache does
not exist or ``--force-recompute`` is supplied.  Styling-only changes therefore
reuse obstacle maps, assignments, tile layouts, expanded paths, and metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from hybrid_mcpp import HybridMCPP, make_random_map


CACHE_VERSION = 1
REGION_COLORS = [
    "#E64B35", "#4DBBD5", "#00A087", "#F39B7F", "#8491B4",
    "#91D1C2", "#DC0000", "#3C5488", "#7E6148", "#B09C85",
]


def _pack_paths(routes: Dict[int, dict], robots: int) -> Tuple[np.ndarray, np.ndarray]:
    arrays = []
    offsets = [0]
    for rid in range(robots):
        path = np.asarray(routes.get(rid, {}).get("path", []), dtype=np.float32).reshape((-1, 2))
        arrays.append(path)
        offsets.append(offsets[-1] + len(path))
    points = np.vstack(arrays) if any(len(a) for a in arrays) else np.empty((0, 2), dtype=np.float32)
    return points, np.asarray(offsets, dtype=np.int32)


def _solver_arrays(solver: HybridMCPP, metrics: dict, prefix: str) -> dict:
    points, offsets = _pack_paths(solver.routes, solver.k)
    return {
        f"{prefix}_assignments": np.asarray(solver.assignments, dtype=np.int16),
        f"{prefix}_tiles": np.asarray(solver.tiles, dtype=np.int16).reshape((-1, 5)),
        f"{prefix}_centroids": np.asarray(solver.centroids_xy, dtype=np.float32).reshape((-1, 2)),
        f"{prefix}_path_points": points,
        f"{prefix}_path_offsets": offsets,
        f"{prefix}_metrics_json": np.asarray(json.dumps(metrics, ensure_ascii=False)),
    }


def _record_from_cache(data, prefix: str) -> dict:
    return {
        "assignments": np.asarray(data[f"{prefix}_assignments"]),
        "tiles": np.asarray(data[f"{prefix}_tiles"]),
        "centroids": np.asarray(data[f"{prefix}_centroids"]),
        "path_points": np.asarray(data[f"{prefix}_path_points"]),
        "path_offsets": np.asarray(data[f"{prefix}_path_offsets"]),
        "metrics": json.loads(str(data[f"{prefix}_metrics_json"].item())),
    }


def cache_path(cache_dir: Path, n: int, robots: int, seed: int, obstacle_ratio: float) -> Path:
    obs_code = int(round(obstacle_ratio * 10000))
    return cache_dir / f"comparison_n{n}_k{robots}_seed{seed}_obs{obs_code}.npz"


def save_pair_cache(
    path: Path, n: int, robots: int, seed: int, obstacle_ratio: float,
    obstacles: np.ndarray, before: HybridMCPP, before_metrics: dict,
    after: HybridMCPP, after_metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_version": np.asarray(CACHE_VERSION, dtype=np.int16),
        "map_size": np.asarray(n, dtype=np.int16),
        "robots": np.asarray(robots, dtype=np.int16),
        "seed": np.asarray(seed, dtype=np.int64),
        "obstacle_ratio": np.asarray(obstacle_ratio, dtype=np.float64),
        "obstacles": np.asarray(obstacles, dtype=bool),
        **_solver_arrays(before, before_metrics, "before"),
        **_solver_arrays(after, after_metrics, "after"),
    }
    np.savez_compressed(path, **payload)


def load_pair_cache(path: Path, n: int, robots: int, seed: int, obstacle_ratio: float):
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        compatible = (
            int(data["cache_version"]) == CACHE_VERSION
            and int(data["map_size"]) == n
            and int(data["robots"]) == robots
            and int(data["seed"]) == seed
            and abs(float(data["obstacle_ratio"]) - obstacle_ratio) < 1e-12
        )
        if not compatible:
            return None
        return {
            "n": n,
            "robots": robots,
            "obstacles": np.asarray(data["obstacles"]),
            "before": _record_from_cache(data, "before"),
            "after": _record_from_cache(data, "after"),
        }


def solve_or_load_pair(
    n: int, robots: int, seed: int, obstacle_ratio: float,
    cache_dir: Path, force_recompute: bool,
) -> dict:
    path = cache_path(cache_dir, n, robots, seed, obstacle_ratio)
    if not force_recompute:
        cached = load_pair_cache(path, n, robots, seed, obstacle_ratio)
        if cached is not None:
            print(f"cache hit  {path}", flush=True)
            return cached

    print(f"computing  {n}x{n}", flush=True)
    obstacles = make_random_map(n, obstacle_ratio, seed)
    limit = min(10.0, max(2.0, n / 20.0))
    before = HybridMCPP(n, robots, obstacle_mask=obstacles, seed=seed, tiling_time_limit=limit)
    before_metrics = before.solve("baseline")
    after = HybridMCPP(n, robots, obstacle_mask=obstacles, seed=seed, tiling_time_limit=limit)
    after_metrics = after.solve("hybrid")
    save_pair_cache(
        path, n, robots, seed, obstacle_ratio, obstacles,
        before, before_metrics, after, after_metrics,
    )
    print(f"cached     {path}", flush=True)
    return load_pair_cache(path, n, robots, seed, obstacle_ratio)


def background_rgba(record: dict, obstacles: np.ndarray, robots: int) -> np.ndarray:
    image = np.ones((*obstacles.shape, 4), dtype=float)
    for rid in range(robots):
        image[record["assignments"] == rid] = to_rgba(REGION_COLORS[rid % len(REGION_COLORS)], 0.25)
    image[obstacles] = to_rgba("#111111", 1.0)
    return np.transpose(image, (1, 0, 2))


def tile_outline_segments(tiles: np.ndarray) -> np.ndarray:
    """Vectorized rectangle perimeters for a single LineCollection."""
    if not len(tiles):
        return np.empty((0, 2, 2), dtype=float)
    x = tiles[:, 0].astype(float)
    y = tiles[:, 1].astype(float)
    x2 = x + tiles[:, 2]
    y2 = y + tiles[:, 3]
    segments = np.empty((len(tiles) * 4, 2, 2), dtype=float)
    segments[0::4] = np.stack((np.column_stack((x, y)), np.column_stack((x2, y))), axis=1)
    segments[1::4] = np.stack((np.column_stack((x2, y)), np.column_stack((x2, y2))), axis=1)
    segments[2::4] = np.stack((np.column_stack((x2, y2)), np.column_stack((x, y2))), axis=1)
    segments[3::4] = np.stack((np.column_stack((x, y2)), np.column_stack((x, y))), axis=1)
    return segments


def draw_solution(ax, record: dict, obstacles: np.ndarray, robots: int, title: str) -> None:
    n = obstacles.shape[0]
    metrics = record["metrics"]
    ax.imshow(background_rgba(record, obstacles, robots), origin="lower",
              extent=(0, n, 0, n), interpolation="nearest", zorder=0)

    outline_width = max(0.16, 0.72 * (40.0 / n) ** 0.30)
    outlines = LineCollection(
        tile_outline_segments(record["tiles"]), colors=[(0.08, 0.08, 0.08, 0.48)],
        linewidths=outline_width, zorder=2, rasterized=True,
    )
    ax.add_collection(outlines)

    route_width = max(0.32, 1.45 * (40.0 / n) ** 0.35)
    points = record["path_points"]
    offsets = record["path_offsets"]
    for rid in range(robots):
        color = REGION_COLORS[rid % len(REGION_COLORS)]
        path = points[offsets[rid]:offsets[rid + 1]]
        if len(path) >= 2:
            ax.plot(path[:, 0], path[:, 1], color=color, linewidth=route_width,
                    alpha=0.92, solid_capstyle="round", zorder=4)
        if rid < len(record["centroids"]):
            sx, sy = record["centroids"][rid]
            ax.scatter(sx + 0.5, sy + 0.5, s=max(14, 65 * 40 / n), marker="*",
                       color=color, edgecolor="white", linewidth=0.55, zorder=6)

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"{title}\narea CV={metrics['partition_cv']:.4f}, "
        f"tile CV={metrics['tile_cv']:.4f}, path={metrics['path_length']:.1f}, "
        f"tiles={metrics['tile_count']}",
        fontsize=10,
    )


def legend_handles(robots: int):
    handles = [
        Patch(facecolor="#111111", edgecolor="none", label="Obstacle"),
        Line2D([0], [0], color="#222222", alpha=0.6, lw=1.0, label="Tile outline"),
    ]
    for rid in range(robots):
        color = REGION_COLORS[rid % len(REGION_COLORS)]
        handles.append(Line2D([0], [0], color=color, lw=2.0, marker="*", markersize=7,
                              markeredgecolor="white", label=f"Robot {rid + 1}"))
    return handles


def render_pair(pair: dict, output_dir: Path) -> Path:
    n, robots = pair["n"], pair["robots"]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.3), facecolor="white")
    draw_solution(axes[0], pair["before"], pair["obstacles"], robots,
                  f"Before: FIFO Voronoi + greedy + MST walk ({n}x{n})")
    draw_solution(axes[1], pair["after"], pair["obstacles"], robots,
                  f"After: balanced + local MILP + shortest-tour ({n}x{n})")
    fig.legend(handles=legend_handles(robots), loc="lower center", ncol=robots + 2,
               frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    path = output_dir / f"partition_tile_path_comparison_{n}x{n}.png"
    fig.savefig(path, dpi=210, bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)
    print(f"saved      {path}", flush=True)
    return path


def render_overview(pairs: list, output_dir: Path) -> Path:
    robots = pairs[0]["robots"]
    fig, axes = plt.subplots(len(pairs), 2, figsize=(12.5, 5.6 * len(pairs)), facecolor="white")
    if len(pairs) == 1:
        axes = np.asarray([axes])
    for row, pair in enumerate(pairs):
        n = pair["n"]
        draw_solution(axes[row, 0], pair["before"], pair["obstacles"], robots, f"Before - {n}x{n}")
        draw_solution(axes[row, 1], pair["after"], pair["obstacles"], robots, f"After - {n}x{n}")
    fig.legend(handles=legend_handles(robots), loc="lower center", ncol=robots + 2,
               frameon=False, bbox_to_anchor=(0.5, 0.002))
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    path = output_dir / "partition_tile_path_comparison_all_sizes.png"
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)
    print(f"saved      {path}", flush=True)
    return path


def generate(sizes, robots, seed, obstacle_ratio, output_dir, cache_dir, force_recompute):
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = [
        solve_or_load_pair(n, robots, seed, obstacle_ratio, cache_dir, force_recompute)
        for n in sizes
    ]
    paths = [render_pair(pair, output_dir) for pair in pairs]
    paths.append(render_overview(pairs, output_dir))
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[20, 50, 100, 150, 200])
    parser.add_argument("--robots", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--obstacle-ratio", type=float, default=0.10)
    parser.add_argument("--output", type=Path, default=Path("visualizations"))
    parser.add_argument("--cache-dir", type=Path, default=Path("visualization_cache"))
    parser.add_argument("--force-recompute", action="store_true",
                        help="Ignore compatible cache files and rerun all planning stages.")
    args = parser.parse_args()
    generate(
        args.sizes, args.robots, args.seed, args.obstacle_ratio,
        args.output, args.cache_dir, args.force_recompute,
    )


if __name__ == "__main__":
    main()
