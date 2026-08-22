"""Lab D: five-robot v2.3.2 tile-first path figures across map scales."""
from __future__ import annotations

__version__ = "2.3.2"

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from benchmark_MCCA_tile_first_v2_3_2 import (
    draw_planner_solution, make_random_map, path_legend, run_case,
    save_figure, timestamped_output, write_csv, write_json,
)


PROFILES = {
    "quick": [20, 50],
    "verify": [20, 50, 150, 200],
    "paper": [20, 50, 100, 150, 200],
}


def _parse_shape(value: str) -> tuple[int, int]:
    normalized = value.lower().replace("×", "x")
    parts = normalized.split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("shape must be WIDTHxHEIGHT, e.g. 120x80")
    width, height = map(int, parts)
    if width < 2 or height < 2:
        raise argparse.ArgumentTypeError("width and height must both be >= 2")
    return width, height


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="paper")
    parser.add_argument("--sizes", help="Comma-separated square-size override")
    parser.add_argument("--shapes", help="Comma-separated WIDTHxHEIGHT override; takes precedence over --sizes")
    parser.add_argument("--robots", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--obstacle-ratio", type=float, default=0.10)
    parser.add_argument("--output-root", type=Path, default=Path("."))
    args = parser.parse_args()
    sizes = PROFILES[args.profile] if not args.sizes else [int(v) for v in args.sizes.split(",")]
    shapes = ([_parse_shape(v) for v in args.shapes.split(",")]
              if args.shapes else [(size, size) for size in sizes])
    output = timestamped_output("LAB_DATA/labD_path_planning", args.output_root)
    records, rows = [], []

    for width, height in shapes:
        shape = (width, height)
        map_arg = width if width == height else shape
        mask = make_random_map(map_arg, args.obstacle_ratio, args.seed)
        planner, result, wall_runtime = run_case(map_arg, args.robots, mask, args.seed)
        records.append((planner, result))
        rows.append({
            "map_width": width,
            "map_height": height,
            "robots": args.robots,
            "seed": args.seed,
            "free_cells": result["free_cells"],
            "initial_path_cv": result["initial_path_cv"],
            "raw_path_cv": result["raw_path_cv"],
            "path_cv": result["path_cv"],
            "makespan": result["makespan"],
            "path_length": result["path_length"],
            "legacy_final_path_length": result["legacy_final_path_length"],
            "legacy_final_path_cv": result["legacy_final_path_cv"],
            "unconstrained_path_length": result["unconstrained_path_length"],
            "unconstrained_path_cv": result["unconstrained_path_cv"],
            "balance_guard_applied": result["balance_guard_applied"],
            "tile_count": result["tile_count"],
            "accepted_tile_transfers": result["accepted_tile_transfers"],
            "partition_cv_improvement_pct": 100.0 * (
                result["initial_path_cv"] - result["raw_path_cv"]
            ) / result["initial_path_cv"] if result["initial_path_cv"] > 0 else 0.0,
            "shortcut_gain": sum(result["shortcut_gains"]),
            "portal_shortcuts_tested": sum(result["portal_shortcuts_tested"]),
            "portal_shortcuts_accepted": sum(result["portal_shortcuts_accepted"]),
            "tiling_time": result["tiling_time"],
            "partition_time": result["partition_time"],
            "routing_time": result["routing_time"],
            "validation_time": result["validation_time"],
            "core_time": result["core_time"],
            "total_time": result["total_time"],
            "timing_overhead": result["timing_overhead"],
            "wall_runtime": wall_runtime,
            # Legacy Lab-D column retained for downstream CSV compatibility.
            "runtime": result["total_time"],
            **{key: result[key] for key in (
                "exact_cover", "connected_regions", "route_collision_free", "route_closed",
                "shortcut_segments_legal", "strict_obstacle_no_touch",
                "all_tile_centers_visited",
            )},
        })

        fig, ax = plt.subplots(figsize=(7.2, 7.5), facecolor="white")
        draw_planner_solution(ax, planner, result, f"FACT-MCCA {width}×{height}")
        fig.legend(handles=path_legend(args.robots), loc="lower center", ncol=4,
                   frameon=False, bbox_to_anchor=(0.5, 0.005), fontsize=8.5)
        fig.tight_layout(rect=(0, 0.075, 1, 1))
        save_figure(fig, output, f"paths_{width}x{height}")
        print(
            f"shape={width}x{height}: "
            f"max_path={max(result['path_lengths'], default=0.0):.3f}, "
            f"total_path={result['path_length']:.3f}, "
            f"raw_cv={result['raw_path_cv']:.5f}, "
            f"final_cv={result['path_cv']:.5f}, "
            f"shortcuts={sum(result['portal_shortcuts_accepted'])}/"
            f"{sum(result['portal_shortcuts_tested'])}, "
            f"core_time={result['core_time']:.3f}s, "
            f"total_time={result['total_time']:.3f}s",
            flush=True,
        )

    columns = 3 if len(records) >= 5 else 2
    rows_count = (len(records) + columns - 1) // columns
    fig, axes = plt.subplots(rows_count, columns, figsize=(5.5 * columns, 5.65 * rows_count),
                             facecolor="white", squeeze=False)
    for ax, (planner, result) in zip(axes.flat, records):
        draw_planner_solution(ax, planner, result, f"FACT-MCCA {planner.width}×{planner.height}")
    for ax in axes.flat[len(records):]:
        ax.axis("off")
    fig.legend(handles=path_legend(args.robots), loc="lower center", ncol=7,
               frameon=False, bbox_to_anchor=(0.5, 0.003), fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, output, "paths_all_scales", dpi=210)
    write_csv(output / "metrics.csv", rows)
    write_json(output / "summary.json", {
        f"{row['map_width']}x{row['map_height']}": {
            "free_cells": row["free_cells"],
            "initial_path_cv": row["initial_path_cv"],
            "partitioned_raw_path_cv": row["raw_path_cv"],
            "final_shortcut_path_cv": row["path_cv"],
            "final_makespan": row["makespan"],
            "accepted_tile_transfers": row["accepted_tile_transfers"],
            "partition_cv_improvement_pct": row["partition_cv_improvement_pct"],
            "shortcut_gain": row["shortcut_gain"],
            "portal_shortcuts_tested": row["portal_shortcuts_tested"],
            "portal_shortcuts_accepted": row["portal_shortcuts_accepted"],
            "tiling_time": row["tiling_time"],
            "partition_time": row["partition_time"],
            "routing_time": row["routing_time"],
            "validation_time": row["validation_time"],
            "core_time": row["core_time"],
            "total_time": row["total_time"],
            "timing_overhead": row["timing_overhead"],
            "wall_runtime": row["wall_runtime"],
        }
        for row in rows
    })
    write_json(output / "config.json", {**vars(args), "sizes_resolved": sizes, "shapes_resolved": shapes})
    print(f"outputs: {output.resolve()}")


if __name__ == "__main__":
    main()
