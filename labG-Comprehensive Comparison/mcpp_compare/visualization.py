from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .grid import GridMap
from .metrics import dfs_coverage_path, path_distance, stc_contour_paths
from .planners import PlanResult
from .user_mst import Point, TileLayer, TiledMST, compute_tiled_mst, tiled_mst_component_walks


ROBOT_PALETTE = [
    "#2563EB",
    "#D97706",
    "#059669",
    "#DC2626",
    "#7C3AED",
    "#0891B2",
    "#DB2777",
    "#525252",
]

@dataclass(frozen=True)
class VisualizationRecord:
    scenario: str
    method: str
    image: Path
    path_length: float | None
    path_mode: str


def save_path_figure(
    grid: GridMap,
    result: PlanResult,
    output_dir: Path,
    scenario: str,
    path_style: str = "dfs",
) -> VisualizationRecord:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{scenario}_{_slug(result.method)}_paths.png"
    tiled_mst = _tiled_mst_for_result(grid, result)
    final_paths, path_lengths_by_robot, path_mode = final_paths_from_result(grid, result, path_style, tiled_mst)
    path_length = float(sum(path_lengths_by_robot.values()))

    fig, ax = _new_path_axes(grid)
    if result.extra.get("projection_kind") == "monitoring_cells":
        _draw_partition_background(ax, grid, result)
    else:
        _draw_grid_background(ax, grid)
    _draw_robot_paths(ax, final_paths)
    _draw_roots(ax, result)
    _draw_robot_legend(ax, final_paths)
    coverage_model = result.extra.get("sensor_model") or result.extra.get("cell_model")
    title_suffix = f"{coverage_model} | " if coverage_model else ""
    ax.set_title(f"{result.method} | {title_suffix}final path = {path_length:.1f}", fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(image_path, bbox_inches="tight")
    plt.close(fig)

    return VisualizationRecord(
        scenario=scenario,
        method=result.method,
        image=image_path,
        path_length=path_length,
        path_mode=path_mode,
    )


def save_voronoi_mst_figure(
    grid: GridMap,
    result: PlanResult,
    output_dir: Path,
    scenario: str,
) -> VisualizationRecord:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{scenario}_{_slug(result.method)}_voronoi_mst_path.png"
    tiled_mst = _tiled_mst_for_result(grid, result)
    if tiled_mst is None:
        raise ValueError(f"{result.method} does not expose the script tile MST needed for this visualization")

    final_paths, path_lengths_by_robot, path_mode = final_paths_from_result(grid, result, "stc", tiled_mst)
    path_length = float(sum(path_lengths_by_robot.values()))

    fig, ax = _new_path_axes(grid)
    _draw_partition_background(ax, grid, result)
    _draw_tile_boundaries(ax, tiled_mst)
    _draw_tile_mst_edges(ax, tiled_mst)
    _draw_round_trip_paths(ax, final_paths)
    _draw_roots(ax, result)
    _draw_voronoi_mst_legend(ax, result)
    ax.set_title(
        f"{result.method} | partition + tile-MST + round-trip path = {path_length:.1f}",
        fontsize=12,
        pad=10,
    )
    fig.tight_layout()
    fig.savefig(image_path, bbox_inches="tight")
    plt.close(fig)

    return VisualizationRecord(
        scenario=scenario,
        method=result.method,
        image=image_path,
        path_length=path_length,
        path_mode=path_mode,
    )


def final_paths_from_result(
    grid: GridMap,
    result: PlanResult,
    path_style: str,
    tiled_mst: TiledMST | None,
) -> tuple[dict[int, list[list[Point]]], dict[int, float], str]:
    robot_count = max(len(result.roots), max(result.assignments.values(), default=-1) + 1)
    paths_by_robot: dict[int, list[list[Point]]] = {}
    lengths_by_robot: dict[int, float] = {}

    if tiled_mst is not None:
        for rid in range(robot_count):
            segments = tiled_mst_component_walks(tiled_mst, rid)
            paths_by_robot[rid] = segments
            lengths_by_robot[rid] = float(2.0 * tiled_mst.lengths_by_robot.get(rid, 0.0))
        return paths_by_robot, lengths_by_robot, "tile_mst_dfs_roundtrip"

    if result.paths is not None:
        for rid, path in enumerate(result.paths):
            plotted = _plot_points_from_result_path(
                path,
                float(result.extra.get("path_coordinate_offset", 0.5)),
            )
            paths_by_robot[rid] = [plotted]
            lengths_by_robot[rid] = path_distance(path)
        return paths_by_robot, lengths_by_robot, str(result.extra.get("path_mode", "official_path"))

    for rid in range(robot_count):
        cells = {cell for cell, assigned in result.assignments.items() if assigned == rid}
        if path_style == "stc":
            paths = stc_contour_paths(grid, cells, result.roots[rid] if rid < len(result.roots) else None)
            paths_by_robot[rid] = paths
            lengths_by_robot[rid] = float(sum(path_distance(path) for path in paths))
            mode = "stc_contour_path"
            continue

        root = result.roots[rid] if rid < len(result.roots) else None
        path = dfs_coverage_path(grid, cells, root)
        paths_by_robot[rid] = [[(float(x) + 0.5, float(y) + 0.5) for x, y in path]]
        lengths_by_robot[rid] = float(max(len(path) - 1, 0))
        mode = "dfs_coverage_path"
    return paths_by_robot, lengths_by_robot, mode


def _new_path_axes(grid: GridMap):
    fig_size = max(6.0, min(11.0, grid.width * 0.34))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=180)
    ax.set_aspect("equal")
    ax.set_xlim(0, grid.width)
    ax.set_ylim(0, grid.height)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(0, grid.width + 1, 1))
    ax.set_yticks(np.arange(0, grid.height + 1, 1))
    ax.grid(color="#E5E7EB", linewidth=0.45)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig, ax


def _draw_grid_background(ax, grid: GridMap) -> None:
    for x in range(grid.width):
        for y in range(grid.height):
            cell = (x, y)
            if cell in grid.obstacles:
                face = "#111827"
                edge = "#FFFFFF"
            else:
                face = "#F9FAFB"
                edge = "#E5E7EB"
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor=face, edgecolor=edge, linewidth=0.35, zorder=1))


def _draw_partition_background(ax, grid: GridMap, result: PlanResult) -> None:
    for x in range(grid.width):
        for y in range(grid.height):
            cell = (x, y)
            if cell in grid.obstacles:
                face = "#111827"
                edge = "#FFFFFF"
                alpha = 1.0
            else:
                rid = result.assignments.get(cell, -1)
                face = ROBOT_PALETTE[rid % len(ROBOT_PALETTE)] if rid >= 0 else "#F9FAFB"
                edge = "#FFFFFF"
                alpha = 0.28 if rid >= 0 else 1.0
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor=face, edgecolor=edge, linewidth=0.35, alpha=alpha, zorder=1))


def _draw_tile_boundaries(ax, tiled_mst: TiledMST) -> None:
    for x, y, width, height, _ in tiled_mst.tiles:
        ax.add_patch(
            Rectangle(
                (x, y),
                width,
                height,
                facecolor="none",
                edgecolor="#111827",
                linewidth=0.35,
                alpha=0.32,
                zorder=2,
            )
        )


def _draw_tile_mst_edges(ax, tiled_mst: TiledMST) -> None:
    for rid, edges in tiled_mst.edges_by_robot.items():
        color = ROBOT_PALETTE[rid % len(ROBOT_PALETTE)]
        for first, second in edges:
            ax.plot(
                [first[0], second[0]],
                [first[1], second[1]],
                color=color,
                linewidth=2.5,
                alpha=0.82,
                solid_capstyle="round",
                zorder=3,
            )


def _draw_round_trip_paths(ax, final_paths: dict[int, list[list[Point]]]) -> None:
    for rid, segments in final_paths.items():
        color = ROBOT_PALETTE[rid % len(ROBOT_PALETTE)]
        for path in segments:
            if len(path) < 2:
                continue
            xs = [point[0] for point in path]
            ys = [point[1] for point in path]
            ax.plot(
                xs,
                ys,
                color="#FFFFFF",
                linewidth=4.3,
                alpha=0.92,
                linestyle=(0, (3, 2)),
                solid_capstyle="round",
                zorder=4,
            )
            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=1.45,
                alpha=0.98,
                linestyle=(0, (3, 2)),
                solid_capstyle="round",
                zorder=5,
            )


def _draw_robot_paths(ax, final_paths: dict[int, list[list[Point]]]) -> None:
    for rid, segments in final_paths.items():
        color = ROBOT_PALETTE[rid % len(ROBOT_PALETTE)]
        for path in segments:
            if len(path) < 2:
                continue
            xs = [point[0] for point in path]
            ys = [point[1] for point in path]
            ax.plot(xs, ys, color="#FFFFFF", linewidth=3.8, alpha=0.92, solid_capstyle="round", zorder=4)
            ax.plot(xs, ys, color=color, linewidth=1.65, alpha=0.98, solid_capstyle="round", zorder=5)


def _draw_roots(ax, result: PlanResult) -> None:
    for rid, root in enumerate(result.roots):
        ax.scatter(
            root[0] + 0.5,
            root[1] + 0.5,
            s=70,
            marker="*",
            c=ROBOT_PALETTE[rid % len(ROBOT_PALETTE)],
            edgecolors="#111827",
            linewidths=0.8,
            zorder=6,
        )


def _draw_robot_legend(ax, final_paths: dict[int, list[list[Point]]]) -> None:
    legend_handles = [
        Line2D([0], [0], color=ROBOT_PALETTE[rid % len(ROBOT_PALETTE)], lw=2.2, label=f"Robot {rid + 1}")
        for rid in sorted(final_paths)
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)


def _draw_voronoi_mst_legend(ax, result: PlanResult) -> None:
    robot_ids = sorted({rid for rid in result.assignments.values() if rid >= 0})
    legend_handles: list[Rectangle | Line2D] = [
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor=ROBOT_PALETTE[rid % len(ROBOT_PALETTE)],
            alpha=0.28,
            edgecolor="none",
            label=f"Robot {rid + 1} partition",
        )
        for rid in robot_ids
    ]
    legend_handles.extend(
        [
            Line2D([0], [0], color="#111827", lw=2.3, label="Tile MST"),
            Line2D([0], [0], color="#111827", lw=1.8, linestyle=(0, (3, 2)), label="Round-trip path"),
        ]
    )
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)


def _tiled_mst_for_result(grid: GridMap, result: PlanResult) -> TiledMST | None:
    if result.extra.get("mst_mode") != "script_tile_mst":
        return None
    robot_count = max(len(result.roots), max(result.assignments.values(), default=-1) + 1)
    return compute_tiled_mst(grid, result.assignments, robot_count, _tile_layers_for_result(result))


def _tile_layers_for_result(result: PlanResult) -> tuple[TileLayer, ...] | None:
    value = result.extra.get("tile_layers")
    if value is None:
        return None
    return tuple((int(width), int(height)) for width, height in value)  # type: ignore[union-attr]


def _plot_points_from_result_path(
    path: list[tuple[float, float]],
    offset: float,
) -> list[Point]:
    if not path:
        return []
    return [(float(point[0]) + offset, float(point[1]) + offset) for point in path]


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return slug or "method"
