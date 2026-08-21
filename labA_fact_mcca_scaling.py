"""Lab A: FACT-MCCA beam ablation, runtime decomposition, and scaling."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import statistics

import matplotlib.pyplot as plt
import numpy as np

from benchmark_MCCA import (
    make_random_map, run_portfolio_case, save_figure, timestamped_output,
    write_csv, write_json,
)


PROFILES = {
    "quick": {"sizes": [20, 50], "seeds": [42], "budget": 1.0},
    "verify": {"sizes": [20, 50], "seeds": [42], "budget": 6.0},
    "paper": {"sizes": [20, 50, 100, 150, 200], "seeds": [42, 2024, 8191], "budget": 6.0},
}


def ints(text: str | None, fallback: list[int]) -> list[int]:
    return fallback if not text else [int(value) for value in text.split(",")]


def summarize(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[(int(row["map_size"]), row["variant"])].append(row)
    summary = {}
    for (size, variant), group in sorted(groups.items()):
        reductions = [
            100 * (float(row["initial_makespan"]) - float(row["makespan"]))
            / float(row["initial_makespan"])
            for row in group
        ]
        summary[f"{size}:{variant}"] = {
            "cases": len(group),
            "mean_makespan": statistics.mean(float(row["makespan"]) for row in group),
            "mean_reduction_pct": statistics.mean(reductions),
            "mean_exclusive_runtime": statistics.mean(float(row["exclusive_runtime"]) for row in group),
            "mean_end_to_end_runtime": statistics.mean(
                float(row["equivalent_end_to_end_runtime"]) for row in group
            ),
        }
    return summary


def plot(rows: list[dict], output: Path) -> None:
    variants = [
        "assignment_geometry", "route_aware_geometry", "dual_guided_fact",
        "contracted_fact_lns", "four_beam_portfolio",
    ]
    labels = ["Assignment", "Route-aware", "Dual FACT", "Contracted FACT", "Portfolio"]
    sizes = sorted({int(row["map_size"]) for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), facecolor="white")
    for variant, label in zip(variants, labels):
        gains, runtimes = [], []
        for size in sizes:
            group = [row for row in rows if int(row["map_size"]) == size and row["variant"] == variant]
            gains.append(statistics.mean(
                100 * (float(row["initial_makespan"]) - float(row["makespan"]))
                / float(row["initial_makespan"]) for row in group
            ))
            runtimes.append(statistics.mean(float(row["equivalent_end_to_end_runtime"]) for row in group))
        axes[0].plot(sizes, gains, marker="o", label=label)
        axes[1].plot(sizes, runtimes, marker="o", label=label)
    axes[0].set(xlabel="Map side length", ylabel="Makespan reduction (%)",
                title="Optimization gain")
    axes[1].set(xlabel="Map side length", ylabel="Equivalent end-to-end runtime (s)",
                title="Planning time")
    for ax in axes:
        ax.grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, output, "fact_mcca_scaling")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="paper")
    parser.add_argument("--sizes", help="Comma-separated override")
    parser.add_argument("--seeds", help="Comma-separated override")
    parser.add_argument("--robots", type=int, default=5)
    parser.add_argument("--obstacle-ratio", type=float, default=0.10)
    parser.add_argument("--budget", type=float)
    parser.add_argument("--output-root", type=Path, default=Path("."))
    args = parser.parse_args()
    profile = PROFILES[args.profile]
    sizes = ints(args.sizes, profile["sizes"])
    seeds = ints(args.seeds, profile["seeds"])
    budget = profile["budget"] if args.budget is None else args.budget
    output = timestamped_output("labA_fact_mcca_scaling", args.output_root)
    rows = []
    for size in sizes:
        for seed in seeds:
            mask = make_random_map(size, args.obstacle_ratio, seed)
            _, result, case_rows = run_portfolio_case(
                size, args.robots, mask, seed, budget,
            )
            rows.extend(case_rows)
            print(
                f"size={size} seed={seed}: {result['initial_makespan']:.3f} -> "
                f"{result['makespan']:.3f}, winner={result['selected_refinement_beam']}",
                flush=True,
            )
    write_csv(output / "raw.csv", rows)
    write_json(output / "summary.json", summarize(rows))
    write_json(output / "config.json", vars(args) | {"sizes_resolved": sizes, "seeds_resolved": seeds,
                                                       "budget_resolved": budget})
    plot(rows, output)
    print(f"outputs: {output.resolve()}")


if __name__ == "__main__":
    main()
