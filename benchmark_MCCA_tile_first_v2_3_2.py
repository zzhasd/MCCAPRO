"""Shared utilities for the v2.3.2 tile-first FACT-MCCA experiments."""
from __future__ import annotations

__version__ = "2.3.2"

import csv
from datetime import datetime
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from mainline_tile_first_v2_3_2 import FACTMCCA, make_random_map


REGION_COLORS = [
    "#E64B35", "#4DBBD5", "#00A087", "#F39B7F", "#8491B4",
    "#91D1C2", "#DC0000", "#3C5488", "#7E6148", "#B09C85",
]
VALID_KEYS = (
    "exact_cover", "connected_regions", "route_collision_free", "route_closed",
    "shortcut_segments_legal", "strict_obstacle_no_touch", "all_tile_centers_visited",
)


def timestamped_output(lab_name: str, output_root: Path | str = ".") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(output_root) / f"{lab_name}_{stamp}"
    suffix = 1
    while path.exists():
        path = Path(output_root) / f"{lab_name}_{stamp}_{suffix:02d}"
        suffix += 1
    path.mkdir(parents=True)
    return path


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def valid_result(result: dict) -> bool:
    return all(bool(result[key]) for key in VALID_KEYS)


def planner_kwargs(map_shape, robots: int, mask: np.ndarray, seed: int) -> dict:
    width, height = mask.shape
    characteristic = float(np.sqrt(width * height))
    return {
        "map_shape": map_shape,
        "robot_num": robots,
        "obstacle_mask": mask,
        "seed": seed,
        "partition_iterations": 60,
        "tiling_time_limit": min(6.0, max(1.0, characteristic / 25.0)),
        "candidate_limit": 8,
    }


def run_case(map_shape, robots: int, mask: np.ndarray, seed: int) -> tuple[FACTMCCA, dict, float]:
    planner = FACTMCCA(**planner_kwargs(map_shape, robots, mask, seed))
    started = time.perf_counter()
    result = planner.solve("tile_first")
    wall_runtime = time.perf_counter() - started
    if not valid_result(result):
        raise AssertionError("FACT-MCCA returned an invalid tile-first coverage solution")
    return planner, result, wall_runtime


def tile_segments(tiles: np.ndarray) -> np.ndarray:
    if not len(tiles):
        return np.empty((0, 2, 2), dtype=float)
    x, y = tiles[:, 0].astype(float), tiles[:, 1].astype(float)
    x2, y2 = x + tiles[:, 2], y + tiles[:, 3]
    segments = np.empty((len(tiles) * 4, 2, 2), dtype=float)
    segments[0::4] = np.stack((np.column_stack((x, y)), np.column_stack((x2, y))), axis=1)
    segments[1::4] = np.stack((np.column_stack((x2, y)), np.column_stack((x2, y2))), axis=1)
    segments[2::4] = np.stack((np.column_stack((x2, y2)), np.column_stack((x, y2))), axis=1)
    segments[3::4] = np.stack((np.column_stack((x, y2)), np.column_stack((x, y))), axis=1)
    return segments


def draw_planner_solution(ax, planner: FACTMCCA, result: dict, title: str) -> None:
    width, height, robots = planner.width, planner.height, planner.k
    scale = float(np.sqrt(width * height))
    image = np.ones((width, height, 4), dtype=float)
    for rid in range(robots):
        image[planner.assignments == rid] = to_rgba(REGION_COLORS[rid], 0.25)
    image[planner.obstacles] = to_rgba("#111111", 1.0)
    ax.imshow(np.transpose(image, (1, 0, 2)), origin="lower", extent=(0, width, 0, height),
              interpolation="nearest", zorder=0)
    tiles = np.asarray(planner.tiles, dtype=float).reshape((-1, 5))
    ax.add_collection(LineCollection(
        tile_segments(tiles), colors=[(0.08, 0.08, 0.08, 0.40)],
        linewidths=max(0.14, 0.62 * (40 / scale) ** 0.3), rasterized=True, zorder=2,
    ))
    for rid in range(robots):
        path = np.asarray(planner.routes[rid]["path"], dtype=float)
        if len(path) > 1:
            ax.plot(path[:, 0], path[:, 1], color=REGION_COLORS[rid],
                    linewidth=max(0.3, 1.35 * (40 / scale) ** 0.35),
                    alpha=0.94, rasterized=True, zorder=4)
    cv_improvement = (
        100.0 * (result["initial_path_cv"] - result["raw_path_cv"])
        / result["initial_path_cv"] if result["initial_path_cv"] > 0 else 0.0
    )
    accepted = sum(result["portal_shortcuts_accepted"])
    tested = sum(result["portal_shortcuts_tested"])
    ax.set_title(
        f"{title} (global tile-first)\n"
        f"CV init→partition→shortcut: {result['initial_path_cv']:.4f}→"
        f"{result['raw_path_cv']:.4f}→{result['path_cv']:.4f} "
        f"(partition −{cv_improvement:.2f}%)\n"
        f"path={result['path_length']:.1f}, tiles={result['tile_count']}, "
        f"shortcuts={accepted}/{tested}", fontsize=9,
    )
    ax.set(xlim=(0, width), ylim=(0, height), aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])


def path_legend(robots: int) -> list:
    handles = [
        Patch(facecolor="#111111", edgecolor="none", label="Obstacle"),
        Line2D([0], [0], color="#222222", alpha=0.6, label="Tile outline"),
    ]
    for rid in range(robots):
        handles.append(Line2D([0], [0], color=REGION_COLORS[rid], label=f"Robot {rid + 1}"))
    return handles


def save_figure(fig, output_dir: Path, stem: str, dpi: int = 240) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{stem}.pdf", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
