"""Reproducible baseline-vs-hybrid benchmark for 20..200 grids."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics

import matplotlib.pyplot as plt

from hybrid_mcpp import HybridMCPP, make_random_map


FIELDS = [
    "method", "map_size", "robots", "seed", "free_cells", "partition_sizes",
    "partition_cv", "tile_count", "tile_counts", "tile_cv", "path_length",
    "path_lengths", "path_cv", "partition_time",
    "tiling_time", "routing_time", "total_time", "exact_cover",
    "connected_regions",
]


def run(sizes, seeds, robots, obstacle_ratio, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in sizes:
        for seed in seeds:
            mask = make_random_map(n, obstacle_ratio, seed)
            limit = min(10.0, max(2.0, n / 20.0))
            for method in ("baseline", "hybrid"):
                solver = HybridMCPP(
                    n, robots, obstacle_mask=mask, seed=seed,
                    tiling_time_limit=limit, balance_tolerance=0.01,
                )
                result = solver.solve(method)
                rows.append(result)
                print(
                    f"{n:3d} seed={seed:5d} {method:8s} "
                    f"CV={result['partition_cv']:.5f} tiles={result['tile_count']:5d} "
                    f"path={result['path_length']:9.2f} time={result['total_time']:7.2f}s",
                    flush=True,
                )

    csv_path = output_dir / "benchmark_raw.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            row = row.copy()
            for key in ("partition_sizes", "tile_counts", "path_lengths"):
                row[key] = json.dumps(row[key])
            writer.writerow({key: row[key] for key in FIELDS})

    summary = summarize(rows, sizes)
    json_path = output_dir / "benchmark_summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    plot(summary, output_dir / "benchmark_comparison.png")
    return rows, summary


def summarize(rows, sizes):
    summary = {"by_size": {}, "overall": {}}
    paired_improvements = {"cv": [], "tiles": [], "path": []}
    for n in sizes:
        subset = [r for r in rows if r["map_size"] == n]
        item = {}
        for method in ("baseline", "hybrid"):
            group = [r for r in subset if r["method"] == method]
            item[method] = {}
            for metric in ("partition_cv", "tile_cv", "path_cv", "tile_count", "path_length", "total_time"):
                vals = [float(r[metric]) for r in group]
                item[method][metric] = {
                    "mean": statistics.mean(vals),
                    "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                }
        for metric, short in (("partition_cv", "cv"), ("tile_count", "tiles"), ("path_length", "path")):
            b = item["baseline"][metric]["mean"]
            h = item["hybrid"][metric]["mean"]
            item[f"{short}_reduction_pct"] = 100.0 * (b - h) / b
        item["time_ratio"] = item["hybrid"]["total_time"]["mean"] / item["baseline"]["total_time"]["mean"]
        summary["by_size"][str(n)] = item

    # Paired per-map reductions avoid letting the 200x200 cases dominate.
    keys = sorted({(r["map_size"], r["seed"]) for r in rows})
    for n, seed in keys:
        b = next(r for r in rows if r["map_size"] == n and r["seed"] == seed and r["method"] == "baseline")
        h = next(r for r in rows if r["map_size"] == n and r["seed"] == seed and r["method"] == "hybrid")
        paired_improvements["cv"].append(100 * (b["partition_cv"] - h["partition_cv"]) / b["partition_cv"] if b["partition_cv"] else 0)
        paired_improvements["tiles"].append(100 * (b["tile_count"] - h["tile_count"]) / b["tile_count"])
        paired_improvements["path"].append(100 * (b["path_length"] - h["path_length"]) / b["path_length"])
    for metric, vals in paired_improvements.items():
        summary["overall"][f"{metric}_reduction_pct_mean"] = statistics.mean(vals)
        summary["overall"][f"{metric}_reduction_pct_median"] = statistics.median(vals)
        summary["overall"][f"{metric}_wins"] = sum(v > 0 for v in vals)
        summary["overall"][f"{metric}_cases"] = len(vals)
    summary["overall"]["max_hybrid_time"] = max(r["total_time"] for r in rows if r["method"] == "hybrid")
    summary["overall"]["all_exact_cover"] = all(r["exact_cover"] for r in rows)
    summary["overall"]["all_connected"] = all(r["connected_regions"] for r in rows)
    return summary


def plot(summary, path):
    sizes = [int(x) for x in summary["by_size"]]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    specs = [
        ("partition_cv", "Partition CV", False),
        ("tile_count", "Tile count", False),
        ("path_length", "Closed path length", False),
        ("total_time", "Total runtime (s)", True),
    ]
    for ax, (metric, title, logy) in zip(axes.ravel(), specs):
        for method, color, marker in (("baseline", "#C44E52", "o"), ("hybrid", "#4C72B0", "s")):
            means = [summary["by_size"][str(n)][method][metric]["mean"] for n in sizes]
            stds = [summary["by_size"][str(n)][method][metric]["std"] for n in sizes]
            ax.errorbar(sizes, means, yerr=stds, label=method, color=color, marker=marker, capsize=3)
        ax.set_title(title)
        ax.set_xlabel("Grid side length N")
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[20, 50, 100, 150, 200])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2024, 8191])
    parser.add_argument("--robots", type=int, default=5)
    parser.add_argument("--obstacle-ratio", type=float, default=0.10)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    _, summary = run(args.sizes, args.seeds, args.robots, args.obstacle_ratio, args.output)
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
