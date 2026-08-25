"""Lab B: paired comparison with the unmodified official LS-MCPP source."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from benchmark_MCCA import (
    FACTMCCA, lsmcpp_provenance, make_random_map, paired_statistics,
    read_official_lsmcpp_instance, relaunch_with_lsmcpp_if_needed,
    run_fixed_footprint_mcca, run_official_lsmcpp, save_figure,
    timestamped_output, write_csv, write_json, write_lsmcpp_instance,
)


PROFILES = {
    "quick": {"seeds": [0], "iterations": 300, "budget": 2.0, "official": []},
    "verify": {"seeds": [0, 1, 2], "iterations": 3000, "budget": 20.0, "official": []},
    "paper": {"seeds": list(range(30)), "iterations": 3000, "budget": 20.0,
              "official": ["floor_medium"]},
}


def plot(rows: list[dict], output: Path) -> None:
    configs = [
        ("Length (post-hoc for LS master)", "lsmcpp_length_posthoc_makespan", "mcca_length_makespan"),
        ("Turn-aware primary objective", "lsmcpp_turn_makespan", "mcca_turn_makespan"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), facecolor="white")
    for ax, (title, xcol, ycol) in zip(axes, configs):
        random_rows = [row for row in rows if row["instance_type"] == "random"]
        official_rows = [row for row in rows if row["instance_type"] == "official"]
        x = np.asarray([float(row[xcol]) for row in random_rows])
        y = np.asarray([float(row[ycol]) for row in random_rows])
        stats = paired_statistics(x, y)
        all_values = list(x) + list(y)
        for row in official_rows:
            all_values.extend((float(row[xcol]), float(row[ycol])))
        lo, hi = min(all_values) - 3, max(all_values) + 3
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="0.4")
        ax.scatter(x, y, s=34, alpha=0.75,
                   label=f"{len(random_rows)} random 20×20 cases")
        for row in official_rows:
            ax.scatter(float(row[xcol]), float(row[ycol]), marker="*", s=110,
                       label=f"Official {row['instance']}")
        pvalue = stats["wilcoxon_one_sided_p"]
        ptext = "n/a" if not np.isfinite(pvalue) else f"{pvalue:.2e}"
        ax.text(
            0.03, 0.97,
            f"aggregate improvement={stats['aggregate_improvement_pct']:.2f}%\n"
            f"W/T/L={stats['wins']}/{stats['ties']}/{stats['losses']}, p={ptext}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
        )
        ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Official LS-MCPP makespan",
               ylabel="MCCA makespan", title=title)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.18)
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    save_figure(fig, output, "lsmcpp_paired_comparison")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="paper")
    parser.add_argument("--lsmcpp-root", type=Path,
                        default=Path(".external_baselines/LS-MCPP"))
    parser.add_argument("--seeds", help="Comma-separated override")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--budget", type=float)
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--robots", type=int, default=3)
    parser.add_argument("--obstacle-ratio", type=float, default=0.10)
    parser.add_argument("--official", nargs="*")
    parser.add_argument("--output-root", type=Path, default=Path("."))
    parser.add_argument("--ls-runtime", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.lsmcpp_root.resolve()
    relaunch_with_lsmcpp_if_needed(Path(__file__).resolve(), root)
    profile = PROFILES[args.profile]
    seeds = profile["seeds"] if not args.seeds else [int(v) for v in args.seeds.split(",")]
    iterations = profile["iterations"] if args.iterations is None else args.iterations
    budget = profile["budget"] if args.budget is None else args.budget
    official_names = profile["official"] if args.official is None else args.official
    output = timestamped_output("labB_lsmcpp_comparison", args.output_root)
    cases = []
    for seed in seeds:
        mask = make_random_map(args.size, args.obstacle_ratio, seed)
        starts = FACTMCCA(args.size, args.robots, obstacle_mask=mask, seed=seed).robot_starts_xy
        name = f"mcca_fixed_n{args.size}_k{args.robots}_s{seed}"
        cases.append(("random", name, args.size, seed, mask, starts,
                      write_lsmcpp_instance(root, name, mask, starts)))
    for index, name in enumerate(official_names):
        mask, starts, instance = read_official_lsmcpp_instance(root, name)
        cases.append(("official", name, mask.shape[0], index, mask, starts, instance))
    rows = []
    for kind, name, size, seed, mask, starts, instance in cases:
        ls = run_official_lsmcpp(root, instance, iterations, seed)
        length = run_fixed_footprint_mcca(
            size, len(starts), mask, starts, seed, budget, turn_time_90=0.0,
        )
        turn = run_fixed_footprint_mcca(
            size, len(starts), mask, starts, seed, budget, turn_time_90=0.5,
        )
        row = {
            "instance_type": kind, "instance": name, "size": size, "seed": seed,
            "robots": len(starts), "free_cells": int(np.count_nonzero(~mask)), **ls,
            "mcca_length_makespan": length["makespan"],
            "mcca_length_runtime": length["wall_runtime"],
            "mcca_turn_makespan": turn["makespan"],
            "mcca_turn_runtime": turn["wall_runtime"],
            "mcca_valid": all(length[key] and turn[key] for key in (
                "exact_cover", "connected_regions", "rooted_regions",
                "route_collision_free", "depot_closed",
            )),
        }
        rows.append(row)
        print(f"{name}: turn {ls['lsmcpp_turn_makespan']:.2f} -> {turn['makespan']:.2f}",
              flush=True)
    random_rows = [row for row in rows if row["instance_type"] == "random"]
    summary = {
        "turn_primary": paired_statistics(
            (row["lsmcpp_turn_makespan"] for row in random_rows),
            (row["mcca_turn_makespan"] for row in random_rows),
        ),
        "length_posthoc": paired_statistics(
            (row["lsmcpp_length_posthoc_makespan"] for row in random_rows),
            (row["mcca_length_makespan"] for row in random_rows),
        ),
    }
    write_csv(output / "raw.csv", rows)
    write_json(output / "summary.json", summary)
    write_json(output / "provenance.json", lsmcpp_provenance(root))
    write_json(output / "config.json", {**vars(args), "seeds_resolved": seeds,
                                         "iterations_resolved": iterations,
                                         "budget_resolved": budget})
    plot(rows, output)
    print(f"outputs: {output.resolve()}")


if __name__ == "__main__":
    main()
