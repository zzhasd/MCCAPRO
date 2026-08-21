"""Lab C: exact FACT lower/upper certificates with depot-cut ablation."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from benchmark_MCCA import FACTMCCA, make_random_map, save_figure, timestamped_output, write_csv, write_json
from mainline import solve_exact_fact_bounds


PROFILES = {
    "quick": {"sizes": [4], "seeds": [0], "time_limit": 2.0},
    "verify": {"sizes": [5], "seeds": [0, 1, 2], "time_limit": 12.0},
    "paper": {"sizes": [4, 5], "seeds": [0, 1, 2], "time_limit": 12.0},
}


def solve_case(size: int, seed: int, robots: int, ratio: float,
               time_limit: float, separated: bool) -> dict:
    mask = make_random_map(size, ratio, seed)
    starts = FACTMCCA(size, robots, obstacle_mask=mask, seed=seed).robot_starts_xy
    started = time.perf_counter()
    answer = solve_exact_fact_bounds(
        mask, starts, time_limit=time_limit, mip_rel_gap=0.0,
        strengthen_connectivity=False,
        separate_connectivity_cuts=separated,
        cut_rounds=4, cuts_per_round=20,
    )
    runtime = time.perf_counter() - started
    lower, upper = answer["lower"], answer["upper"]
    return {
        "size": size, "seed": seed, "robots": robots,
        "variant": "dynamic_depot_cut" if separated else "base_compact",
        "free_cells": int(np.count_nonzero(~mask)),
        "lower_bound": answer["lower_bound"], "upper_bound": answer["upper_bound"],
        "certificate_ratio": answer["certificate_ratio"],
        "certified_two_approx": answer["certified_two_approx"],
        "lower_optimal": lower["optimal"], "upper_optimal": upper["optimal"],
        "lower_gap": lower["mip_gap"], "upper_gap": upper["mip_gap"],
        "separated_cuts": lower["separated_connectivity_cuts"] + upper["separated_connectivity_cuts"],
        "runtime": runtime,
    }


def plot(rows: list[dict], output: Path) -> None:
    cases = sorted({(int(row["size"]), int(row["seed"])) for row in rows})
    fig, ax = plt.subplots(figsize=(max(6.4, len(cases) * 1.1), 4.5), facecolor="white")
    for index, case in enumerate(cases):
        base = next(float(row["certificate_ratio"]) for row in rows
                    if (int(row["size"]), int(row["seed"])) == case and row["variant"] == "base_compact")
        cut = next(float(row["certificate_ratio"]) for row in rows
                   if (int(row["size"]), int(row["seed"])) == case and row["variant"] == "dynamic_depot_cut")
        ax.plot([index - 0.17, index + 0.17], [base, cut], marker="o", linewidth=1.6)
    ax.axhline(2.0, color="0.4", linestyle="--", label="2-approximation threshold")
    ax.set_xticks(range(len(cases)), [f"{n}×{n}\nseed {s}" for n, s in cases])
    ax.set_ylabel("Executable upper / certified lower")
    ax.set_title("FACT certificate tightening: base → dynamic depot-cut")
    ax.grid(axis="y", alpha=0.18); ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output, "fact_certificates")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="paper")
    parser.add_argument("--sizes", help="Comma-separated override")
    parser.add_argument("--seeds", help="Comma-separated override")
    parser.add_argument("--robots", type=int, default=2)
    parser.add_argument("--obstacle-ratio", type=float, default=0.08)
    parser.add_argument("--time-limit", type=float)
    parser.add_argument("--output-root", type=Path, default=Path("."))
    args = parser.parse_args()
    profile = PROFILES[args.profile]
    sizes = profile["sizes"] if not args.sizes else [int(v) for v in args.sizes.split(",")]
    seeds = profile["seeds"] if not args.seeds else [int(v) for v in args.seeds.split(",")]
    limit = profile["time_limit"] if args.time_limit is None else args.time_limit
    output = timestamped_output("labC_fact_certificates", args.output_root)
    rows = []
    for size in sizes:
        for seed in seeds:
            for separated in (False, True):
                try:
                    row = solve_case(size, seed, args.robots, args.obstacle_ratio, limit, separated)
                    rows.append(row)
                    print(f"{size}x{size} seed={seed} {row['variant']}: {row['certificate_ratio']:.3f}",
                          flush=True)
                except (RuntimeError, ValueError) as exc:
                    rows.append({"size": size, "seed": seed, "robots": args.robots,
                                 "variant": "dynamic_depot_cut" if separated else "base_compact",
                                 "status": f"failed: {exc}"})
    successful = [row for row in rows if "certificate_ratio" in row]
    write_csv(output / "raw.csv", rows)
    write_json(output / "config.json", {**vars(args), "sizes_resolved": sizes,
                                         "seeds_resolved": seeds, "time_limit_resolved": limit})
    if len(successful) == len(rows):
        plot(successful, output)
    print(f"outputs: {output.resolve()}")


if __name__ == "__main__":
    main()
