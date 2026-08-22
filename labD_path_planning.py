"""Lab D: five-robot FACT-MCCA path figures across map scales."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from benchmark_MCCA import (
    draw_planner_solution, make_random_map, path_legend, run_portfolio_case,
    save_figure, timestamped_output, write_csv, write_json,
)


PROFILES = {
    "quick": {"sizes": [20, 50], "budget": 1.0},
    "verify": {"sizes": [20, 50, 150, 200], "budget": 6.0},
    "paper": {"sizes": [20, 50, 100, 150, 200], "budget": 6.0},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="paper")
    parser.add_argument("--sizes", help="Comma-separated override")
    parser.add_argument("--robots", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--obstacle-ratio", type=float, default=0.10)
    parser.add_argument("--budget", type=float)
    parser.add_argument("--output-root", type=Path, default=Path("."))
    args = parser.parse_args()
    profile = PROFILES[args.profile]
    sizes = profile["sizes"] if not args.sizes else [int(v) for v in args.sizes.split(",")]
    budget = profile["budget"] if args.budget is None else args.budget
    output = timestamped_output("labD_path_planning", args.output_root)
    records, rows = [], []
    for size in sizes:
        mask = make_random_map(size, args.obstacle_ratio, args.seed)
        planner, result, beam_rows = run_portfolio_case(
            size, args.robots, mask, args.seed, budget,
        )
        records.append((planner, result))
        portfolio_row = next(row for row in beam_rows if row["variant"] == "four_beam_portfolio")
        beam_runtime_columns = {
            f"{row['variant']}_end_to_end_time": row["equivalent_end_to_end_runtime"]
            for row in beam_rows
            if row["variant"] not in ("pipeline_only", "four_beam_portfolio")
        }
        rows.append({
            "map_size": size, "robots": args.robots, "seed": args.seed,
            "free_cells": result["free_cells"], "initial_makespan": result["initial_makespan"],
            "makespan": result["makespan"], "path_length": result["path_length"],
            "tile_count": result["tile_count"], "selected_beam": result["selected_refinement_beam"],
            "improvement_pct": 100.0 * (
                result["initial_makespan"] - result["makespan"]
            ) / result["initial_makespan"],
            "four_beam_end_to_end_time": portfolio_row["equivalent_end_to_end_runtime"],
            "runtime": result["total_time"], **beam_runtime_columns,
            **{key: result[key] for key in (
                "exact_cover", "connected_regions", "rooted_regions",
                "route_collision_free", "depot_closed",
            )},
        })
        fig, ax = plt.subplots(figsize=(7.2, 7.5), facecolor="white")
        draw_planner_solution(ax, planner, result, f"FACT-MCCA {size}×{size}")
        fig.legend(handles=path_legend(args.robots), loc="lower center", ncol=4,
                   frameon=False, bbox_to_anchor=(0.5, 0.005), fontsize=8.5)
        fig.tight_layout(rect=(0, 0.075, 1, 1))
        save_figure(fig, output, f"paths_{size}x{size}")
        print(
            f"size={size}: max_path={result['maxpath']:.3f}, "
            f"total_path={result['totalpath']:.3f}",
            flush=True,
        )
    columns = 3 if len(records) >= 5 else 2
    rows_count = (len(records) + columns - 1) // columns
    fig, axes = plt.subplots(rows_count, columns, figsize=(5.5 * columns, 5.65 * rows_count),
                             facecolor="white", squeeze=False)
    for ax, (planner, result) in zip(axes.flat, records):
        draw_planner_solution(ax, planner, result, f"FACT-MCCA {planner.n}×{planner.n}")
    for ax in axes.flat[len(records):]:
        ax.axis("off")
    fig.legend(handles=path_legend(args.robots), loc="lower center", ncol=7,
               frameon=False, bbox_to_anchor=(0.5, 0.003), fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, output, "paths_all_scales", dpi=210)
    write_csv(output / "metrics.csv", rows)
    write_json(output / "summary.json", {
        str(row["map_size"]): {
            "free_cells": row["free_cells"],
            "selected_beam": row["selected_beam"],
            "initial_makespan": row["initial_makespan"],
            "final_makespan": row["makespan"],
            "improvement_pct": row["improvement_pct"],
            "four_beam_end_to_end_time": row["four_beam_end_to_end_time"],
        }
        for row in rows
    })
    write_json(output / "config.json", {**vars(args), "sizes_resolved": sizes,
                                         "budget_resolved": budget})


if __name__ == "__main__":
    main()
