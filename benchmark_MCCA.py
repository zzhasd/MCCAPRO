"""Shared reproducibility utilities for the FACT-MCCA experiment labs.

All public lab scripts write into a fresh ``<lab>_YYYYMMDD_HHMMSS`` directory.
This module contains no experiment-specific command-line entry point; it only
provides the common planner runners, official LS-MCPP adapter, serialization,
statistics, and path-rendering helpers used by the lab files.
"""

from __future__ import annotations

from contextlib import nullcontext, redirect_stdout
import csv
from datetime import datetime
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from scipy.stats import bootstrap, wilcoxon
import yaml

from mainline import FACTMCCA, TILE_SHAPES, make_random_map


REGION_COLORS = [
    "#E64B35", "#4DBBD5", "#00A087", "#F39B7F", "#8491B4",
    "#91D1C2", "#DC0000", "#3C5488", "#7E6148", "#B09C85",
]
VALID_KEYS = (
    "exact_cover", "connected_regions", "rooted_regions",
    "route_collision_free", "depot_closed",
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
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def valid_result(result: dict) -> bool:
    return all(bool(result[key]) for key in VALID_KEYS)


def planner_kwargs(
    n: int,
    robots: int,
    mask: np.ndarray,
    seed: int,
    budget: float,
    *,
    contracted: bool = True,
    starts: Sequence[tuple[int, int]] | None = None,
    fixed_footprint: bool = False,
    turn_time_90: float = 0.0,
    max_iterations: int = 4,
    candidate_limit: int = 8,
) -> dict:
    return {
        "map_size": n,
        "robot_num": robots,
        "obstacle_mask": mask,
        "robot_starts": starts,
        "seed": seed,
        "partition_iterations": 60,
        "tiling_time_limit": min(6.0, max(1.0, n / 25.0)),
        "refinement_time_limit": budget,
        "refinement_max_iterations": max_iterations,
        "refinement_candidate_limit": candidate_limit,
        "enable_contracted_fact_beam": contracted,
        "tile_shapes": () if fixed_footprint else TILE_SHAPES,
        "service_time_per_tile": 0.0 if fixed_footprint else 1.0,
        "turn_time_90": turn_time_90,
    }


def run_portfolio_case(
    n: int,
    robots: int,
    mask: np.ndarray,
    seed: int,
    budget: float,
    *,
    max_iterations: int = 4,
    candidate_limit: int = 8,
) -> tuple[FACTMCCA, dict, list[dict]]:
    planner = FACTMCCA(**planner_kwargs(
        n, robots, mask, seed, budget,
        contracted=True,
        max_iterations=max_iterations,
        candidate_limit=candidate_limit,
    ))
    started = time.perf_counter()
    result = planner.solve("coupled")
    wall_runtime = time.perf_counter() - started
    if not valid_result(result):
        raise AssertionError("FACT-MCCA returned an invalid coverage solution")
    initial_route_time = max(0.0, result["routing_time"] - result["refinement_time"])
    pipeline_time = result["partition_time"] + result["tiling_time"] + initial_route_time
    common = {
        "map_size": n,
        "robots": robots,
        "seed": seed,
        "free_cells": result["free_cells"],
        "initial_makespan": result["initial_makespan"],
        "portfolio_makespan": result["makespan"],
        "selected_beam": result["selected_refinement_beam"],
        "pipeline_time": pipeline_time,
        "portfolio_wall_time": wall_runtime,
        "partition_time": result["partition_time"],
        "tiling_time": result["tiling_time"],
        "initial_route_time": initial_route_time,
        "refinement_time": result["refinement_time"],
    }
    rows = [{
        **common,
        "variant": "pipeline_only",
        "makespan": result["initial_makespan"],
        "exclusive_runtime": pipeline_time,
        "equivalent_end_to_end_runtime": pipeline_time,
        "iterations": 0,
        "accepted_modes": "{}",
        "selected": False,
    }]
    for beam in result["refinement_beams"]:
        rows.append({
            **common,
            "variant": beam["beam"],
            "makespan": beam["makespan"],
            "exclusive_runtime": beam["runtime"],
            "equivalent_end_to_end_runtime": pipeline_time + beam["runtime"],
            "iterations": beam["iterations"],
            "accepted_modes": json.dumps(beam["accepted_modes"], sort_keys=True),
            "selected": beam["beam"] == result["selected_refinement_beam"],
        })
    rows.append({
        **common,
        "variant": "four_beam_portfolio",
        "makespan": result["makespan"],
        "exclusive_runtime": result["refinement_time"],
        "equivalent_end_to_end_runtime": wall_runtime,
        "iterations": result["refinement_iterations"],
        "accepted_modes": "{}",
        "selected": True,
    })
    return planner, result, rows


def run_fixed_footprint_mcca(
    n: int,
    robots: int,
    mask: np.ndarray,
    starts: Sequence[tuple[int, int]],
    seed: int,
    budget: float,
    *,
    turn_time_90: float,
    max_iterations: int = 10,
    candidate_limit: int = 12,
) -> dict:
    kwargs = planner_kwargs(
        n, robots, mask, seed, budget,
        contracted=False,
        starts=starts,
        fixed_footprint=True,
        turn_time_90=turn_time_90,
        max_iterations=max_iterations,
        candidate_limit=candidate_limit,
    )
    # Preserve the configuration used by the archived 30-seed LS-MCPP audit.
    kwargs["partition_iterations"] = 120
    planner = FACTMCCA(**kwargs)
    started = time.perf_counter()
    result = planner.solve("coupled")
    result["wall_runtime"] = time.perf_counter() - started
    if not valid_result(result):
        raise AssertionError("Fixed-footprint MCCA returned an invalid solution")
    return result


def lsmcpp_available() -> bool:
    return importlib.util.find_spec("lsmcpp") is not None


def default_lsmcpp_python(root: Path) -> Path:
    return root.parent / "lsmcpp-venv" / "Scripts" / "python.exe"


def relaunch_with_lsmcpp_if_needed(script: Path, root: Path, marker: str = "--ls-runtime") -> None:
    if lsmcpp_available() or marker in sys.argv:
        return
    interpreter = default_lsmcpp_python(root)
    if not interpreter.exists():
        raise RuntimeError(
            "Official LS-MCPP is not importable and its isolated interpreter "
            f"was not found at {interpreter}"
        )
    completed = subprocess.run(
        [str(interpreter), str(script), *sys.argv[1:], marker],
        cwd=str(script.parent),
        check=False,
    )
    raise SystemExit(completed.returncode)


def write_lsmcpp_instance(
    root: Path, name: str, mask: np.ndarray, starts: Sequence[tuple[int, int]],
) -> Path:
    n = mask.shape[0]
    map_dir = root / "data" / "gridmaps"
    instance_dir = root / "data" / "instances"
    map_dir.mkdir(parents=True, exist_ok=True)
    instance_dir.mkdir(parents=True, exist_ok=True)
    map_path = map_dir / f"{name}.map"
    instance_path = instance_dir / f"{name}.mcpp"
    rows = ["".join("@" if mask[x, y] else "." for x in range(n)) for y in range(n)]
    map_path.write_text(
        f"type octile\nheight {n}\nwidth {n}\nmap\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    roots = [int(x) * n + int(y) for x, y in starts]
    instance_path.write_text(
        "\n".join((
            f"map: {name}", "weighted: false", "incomplete: true",
            "root: [" + ", ".join(map(str, roots)) + "]", "weight_seed: 0",
        )) + "\n",
        encoding="utf-8",
    )
    return instance_path


def read_official_lsmcpp_instance(root: Path, name: str) -> tuple[np.ndarray, list, Path]:
    instance_path = root / "data" / "instances" / f"{name}.mcpp"
    instance = yaml.safe_load(instance_path.read_text(encoding="utf-8"))
    if bool(instance.get("weighted", False)):
        raise ValueError(f"{name}: weighted objectives are not comparable")
    map_path = root / "data" / "gridmaps" / f"{instance['map']}.map"
    lines = map_path.read_text(encoding="utf-8").splitlines()
    height, width = int(lines[1].split()[1]), int(lines[2].split()[1])
    if width != height:
        raise ValueError(f"{name}: mainline currently requires square maps")
    mask = np.ones((width, height), dtype=bool)
    for y, line in enumerate(lines[4:4 + height]):
        for x, value in enumerate(line[:width]):
            mask[x, y] = value != "."
    raw_roots = instance["root"]
    starts = ([divmod(int(root), height) for root in raw_roots]
              if isinstance(raw_roots[0], int)
              else [(int(x), int(y)) for x, y in raw_roots])
    if len(set(starts)) != len(starts):
        raise ValueError(f"{name}: duplicate depots are outside the current problem definition")
    return mask, starts, instance_path


def run_official_lsmcpp(
    root: Path,
    instance_path: Path,
    iterations: int,
    seed: int,
    *,
    verbose: bool = False,
) -> dict:
    from lsmcpp.benchmark.instance import MCPP
    from lsmcpp.local_search import LocalSearchMCPP
    from lsmcpp.planners import MFC_planner
    from lsmcpp.pool import PoolType, PrioType, SampleType
    from lsmcpp.benchmark.solution import Solution

    old_cwd = Path.cwd()
    output_context = nullcontext() if verbose else redirect_stdout(io.StringIO())
    end_to_end_started = time.perf_counter()
    try:
        os.chdir(root)
        with output_context:
            mcpp = MCPP.read_instance(str(instance_path))
            init_started = time.perf_counter()
            init = MFC_planner(mcpp)
            init_runtime = time.perf_counter() - init_started
            planner = LocalSearchMCPP(
                mcpp, init, PrioType.CompositeHeur, PoolType.VertexEdgewise,
                verbose=verbose,
            )
            solution, reported_search_runtime = planner.run(
                M=iterations,
                S=max(1, iterations // 20),
                alpha=float(np.exp(np.log(0.2) / max(1, iterations))),
                gamma=0.01,
                sample_type=SampleType.RouletteWheel,
                seed=seed,
            )
    finally:
        os.chdir(old_cwd)
    end_to_end_runtime = time.perf_counter() - end_to_end_started
    graph = mcpp._G_legacy
    length_costs = [Solution.path_cost_legacy(graph, path) for path in solution.Pi]
    covered = set().union(*(set(path) for path in solution.Pi))
    roots = [mcpp.legacy_vertex(root_id) for root_id in mcpp.R]
    edges_valid = all(
        graph.has_edge(path[i], path[i + 1])
        for path in solution.Pi for i in range(len(path) - 1)
    )
    depot_closed = all(
        path and path[0] == root and path[-1] == root
        for path, root in zip(solution.Pi, roots)
    )
    return {
        "lsmcpp_initial_turn_makespan": float(init.tau),
        "lsmcpp_turn_makespan": float(solution.tau),
        "lsmcpp_length_posthoc_makespan": float(max(length_costs)),
        "lsmcpp_reported_search_runtime": float(reported_search_runtime),
        "lsmcpp_init_runtime": init_runtime,
        "lsmcpp_end_to_end_runtime": end_to_end_runtime,
        "lsmcpp_complete": bool(set(graph.nodes).issubset(covered)),
        "lsmcpp_edges_valid": bool(edges_valid),
        "lsmcpp_depot_closed": bool(depot_closed),
    }


def lsmcpp_provenance(root: Path) -> dict:
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True, encoding="utf-8"
        ).strip()
    return {
        "repository": git("remote", "get-url", "origin"),
        "branch": git("branch", "--show-current"),
        "commit": git("rev-parse", "HEAD"),
        "comparison_scope": "official MFC + local search; PBS deconfliction not run",
        "primary_objective": "translation plus 90-degree turn cost 0.5",
        "length_metric_status": "post-hoc metric, not the master-branch search objective",
    }


def paired_statistics(reference: Iterable[float], ours: Iterable[float]) -> dict:
    reference = np.asarray(list(reference), dtype=float)
    ours = np.asarray(list(ours), dtype=float)
    diff = reference - ours
    if len(diff) == 1 or np.allclose(diff, 0):
        pvalue = float("nan")
    else:
        pvalue = float(wilcoxon(diff, alternative="greater").pvalue)
    if len(diff) >= 2:
        ci = bootstrap((diff,), np.mean, confidence_level=0.95,
                       n_resamples=5000, random_state=0).confidence_interval
        ci_values = [float(ci.low), float(ci.high)]
    else:
        ci_values = [float("nan"), float("nan")]
    return {
        "cases": int(len(diff)),
        "aggregate_improvement_pct": float(100 * (1 - ours.sum() / reference.sum())),
        "wins": int(np.sum(diff > 1e-9)),
        "ties": int(np.sum(np.abs(diff) <= 1e-9)),
        "losses": int(np.sum(diff < -1e-9)),
        "wilcoxon_one_sided_p": pvalue,
        "mean_difference_bootstrap_95ci": ci_values,
    }


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
    n, robots = planner.n, planner.k
    image = np.ones((n, n, 4), dtype=float)
    for rid in range(robots):
        image[planner.assignments == rid] = to_rgba(REGION_COLORS[rid], 0.25)
    image[planner.obstacles] = to_rgba("#111111", 1.0)
    ax.imshow(np.transpose(image, (1, 0, 2)), origin="lower", extent=(0, n, 0, n),
              interpolation="nearest", zorder=0)
    tiles = np.asarray(planner.tiles, dtype=float).reshape((-1, 5))
    ax.add_collection(LineCollection(
        tile_segments(tiles), colors=[(0.08, 0.08, 0.08, 0.40)],
        linewidths=max(0.14, 0.62 * (40 / n) ** 0.3), rasterized=True, zorder=2,
    ))
    for rid in range(robots):
        path = np.asarray(planner.routes[rid]["path"], dtype=float)
        if len(path) > 1:
            ax.plot(path[:, 0], path[:, 1], color=REGION_COLORS[rid],
                    linewidth=max(0.3, 1.35 * (40 / n) ** 0.35),
                    alpha=0.94, rasterized=True, zorder=4)
        sx, sy = planner.centroids_xy[rid]
        ax.scatter(sx + 0.5, sy + 0.5, s=max(16, 72 * 40 / n), marker="*",
                   color=REGION_COLORS[rid], edgecolor="white", linewidth=0.6, zorder=6)
    beam = result["selected_refinement_beam"].replace("_", " ")
    improvement = (
        100.0 * (result["initial_makespan"] - result["makespan"])
        / result["initial_makespan"] if result["initial_makespan"] > 0 else 0.0
    )
    ax.set_title(
        f"{title} (winner: {beam})\n"
        f"makespan={result['initial_makespan']:.1f}→{result['makespan']:.1f} "
        f"(−{improvement:.2f}%), path={result['path_length']:.1f}, "
        f"tiles={result['tile_count']}", fontsize=9,
    )
    ax.set(xlim=(0, n), ylim=(0, n), aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])


def path_legend(robots: int) -> list:
    handles = [
        Patch(facecolor="#111111", edgecolor="none", label="Obstacle"),
        Line2D([0], [0], color="#222222", alpha=0.6, label="Tile outline"),
    ]
    for rid in range(robots):
        handles.append(Line2D(
            [0], [0], color=REGION_COLORS[rid], marker="*", markeredgecolor="white",
            label=f"Robot {rid + 1}",
        ))
    return handles


def save_figure(fig, output_dir: Path, stem: str, dpi: int = 240) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{stem}.pdf", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
