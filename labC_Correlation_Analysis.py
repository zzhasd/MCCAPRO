"""Lab C: correlation analysis controller for the current mainline planner.

This file intentionally contains no partition / tiling / TSP algorithm implementation.
It only generates experiment parameters, calls the planner's public ``solve()`` entry,
extracts per-robot metrics, and writes the Lab-C CSV.

Current metric semantics:
    Target_Weight_Ratio
    Actual_Area_Ratio
    Actual_TSP_Ratio
"""
from __future__ import annotations

import csv
import datetime
import importlib.util
import inspect
import os
import sys
from pathlib import Path
import time
from types import ModuleType
from typing import Any, Mapping, Sequence, Type

import numpy as np


# -----------------------------------------------------------------------------
# Experiment configuration (keeps the original Lab-C outer experiment design)
# -----------------------------------------------------------------------------
EXPERIMENT_RANDOM_SEED = 42
NUM_EXPERIMENTS = 100
MAP_SIZE_MIN = 20
MAP_SIZE_MAX = 200
ROBOT_NUM_MIN = 3
ROBOT_NUM_MAX = 7
WEIGHT_MIN = 0.1
WEIGHT_MAX = 1.0
OBSTACLE_RATIO = 0.10

MAINLINE_ENV = "MCPP_MAINLINE"


# -----------------------------------------------------------------------------
# Mainline discovery/loading only; no algorithm implementation lives here.
# -----------------------------------------------------------------------------
def _find_mainline_path(project_root: Path) -> Path:
    env_path = os.environ.get(MAINLINE_ENV)
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{MAINLINE_ENV} points to a missing file: {path}")
        return path

    # Prefer the MCCA-PRO root.  The parent fallback keeps the script usable with
    # the current layout shown in the user's terminal (mainline_new.py one level up).
    search_dirs = (project_root, project_root.parent)
    exact_names = (
        "mainline.py",
        "mainline_new.py",
        "mainline_tile_first_v2_3_2.py",
    )

    for directory in search_dirs:
        for name in exact_names:
            path = directory / name
            if path.is_file():
                return path.resolve()

    for directory in search_dirs:
        candidates = sorted(
            (p for p in directory.glob("mainline*.py") if p.is_file()),
            key=lambda p: (p.stat().st_mtime_ns, p.name),
            reverse=True,
        )
        if candidates:
            return candidates[0].resolve()

    raise FileNotFoundError(
        "No mainline*.py was found in MCCA-PRO or its parent directory. "
        f"You can also set {MAINLINE_ENV}=<absolute mainline path>."
    )


def _load_mainline(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("mcpp_mainline_for_labC", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load mainline: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_solver_class(module: ModuleType) -> Type:
    # Prefer known public class names, including the current HybridMCPP interface.
    for name in ("TileFirstMCPP", "FACTMCCA", "HybridMCPP"):
        solver_cls = getattr(module, name, None)
        if inspect.isclass(solver_cls) and callable(getattr(solver_cls, "solve", None)):
            return solver_cls

    # Defensive compatibility fallback: select a class defined by mainline that
    # exposes solve().  This avoids coupling Lab C to a future class rename.
    candidates = []
    for _, obj in vars(module).items():
        if (
            inspect.isclass(obj)
            and getattr(obj, "__module__", None) == module.__name__
            and callable(getattr(obj, "solve", None))
        ):
            candidates.append(obj)

    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        names = ", ".join(cls.__name__ for cls in candidates)
        raise AttributeError(f"Multiple solver classes with solve() found in mainline: {names}")
    raise AttributeError("No solver class with a public solve() method was found in mainline")


def _constructor_kwargs(
    solver_cls: Type,
    map_size: int,
    robot_num: int,
    target_weights: Sequence[float],
    planner_seed: int,
) -> dict[str, Any]:
    """Adapt experiment inputs to the public constructor names used by mainline."""
    sig = inspect.signature(solver_cls.__init__)
    params = sig.parameters
    kwargs: dict[str, Any] = {}

    if "map_shape" in params:
        kwargs["map_shape"] = map_size
    elif "map_size" in params:
        kwargs["map_size"] = map_size
    else:
        raise TypeError(
            f"{solver_cls.__name__}.__init__() has neither 'map_shape' nor 'map_size': {sig}"
        )

    if "robot_num" in params:
        kwargs["robot_num"] = robot_num
    elif "num_robots" in params:
        kwargs["num_robots"] = robot_num
    else:
        raise TypeError(
            f"{solver_cls.__name__}.__init__() has neither 'robot_num' nor 'num_robots': {sig}"
        )

    if "robot_weights" in params:
        kwargs["robot_weights"] = target_weights
    elif "weights" in params:
        kwargs["weights"] = target_weights
    else:
        raise TypeError(
            f"{solver_cls.__name__}.__init__() has neither 'robot_weights' nor 'weights': {sig}"
        )

    if "obstacle_ratio" in params:
        kwargs["obstacle_ratio"] = OBSTACLE_RATIO
    if "seed" in params:
        kwargs["seed"] = planner_seed
    elif "random_seed" in params:
        kwargs["random_seed"] = planner_seed

    return kwargs


def _as_result_mapping(result: Any) -> Mapping[str, Any]:
    return result if isinstance(result, Mapping) else {}


def _extract_weight_ratios(
    result: Mapping[str, Any], solver: Any, target_weights: Sequence[float], robot_num: int
) -> np.ndarray:
    for source in (
        result.get("robot_weights"),
        result.get("weights"),
        getattr(solver, "robot_weights", None),
        getattr(solver, "weights", None),
        target_weights,
    ):
        if source is None:
            continue
        arr = np.asarray(source, dtype=float).reshape(-1)
        if arr.size == robot_num:
            total = float(arr.sum())
            return arr / total if total > 0.0 else np.zeros(robot_num, dtype=float)
    raise KeyError("Unable to obtain per-robot target weights from mainline")


def _extract_partition_sizes(
    result: Mapping[str, Any], solver: Any, robot_num: int
) -> np.ndarray:
    for source in (
        result.get("partition_sizes"),
        result.get("robot_areas"),
        getattr(solver, "robot_areas", None),
        getattr(solver, "partition_sizes", None),
    ):
        if source is None:
            continue
        arr = np.asarray(source, dtype=float).reshape(-1)
        if arr.size == robot_num:
            return arr

    # Metric-only fallback: count final assignment labels if mainline exposes them.
    # This does not run or reproduce the partition algorithm; it only reads its output.
    assignments = getattr(solver, "assignments", None)
    if assignments is None:
        assignments = getattr(solver, "full_assignments", None)
    if assignments is not None:
        arr = np.asarray(assignments)
        return np.asarray([np.count_nonzero(arr == rid) for rid in range(robot_num)], dtype=float)

    raise KeyError(
        "Unable to obtain per-robot partition areas. Expected result['partition_sizes'] "
        "or an equivalent mainline output."
    )


def _extract_tsp_ratios(
    result: Mapping[str, Any], solver: Any, robot_num: int
) -> np.ndarray:
    # If mainline already returns final path shares, they are exactly the desired ratio.
    shares = result.get("path_shares")
    if shares is not None:
        arr = np.asarray(shares, dtype=float).reshape(-1)
        if arr.size == robot_num:
            total = float(arr.sum())
            return arr / total if total > 0.0 else np.zeros(robot_num, dtype=float)

    length_sources = (
        result.get("path_lengths"),
        result.get("tsp_lengths"),
        result.get("robot_path_lengths"),
        getattr(solver, "path_lengths", None),
        getattr(solver, "tsp_lengths", None),
        getattr(solver, "robot_path_lengths", None),
        getattr(solver, "robot_tsp_lengths", None),
    )
    for source in length_sources:
        if source is None:
            continue
        arr = np.asarray(source, dtype=float).reshape(-1)
        if arr.size == robot_num:
            total = float(arr.sum())
            return arr / total if total > 0.0 else np.zeros(robot_num, dtype=float)

    # Current tile-first mainline stores each final route length in routes[rid]['length'].
    routes = getattr(solver, "routes", None)
    if isinstance(routes, Mapping):
        try:
            lengths = np.asarray(
                [float(routes[rid]["length"]) for rid in range(robot_num)], dtype=float
            )
            total = float(lengths.sum())
            return lengths / total if total > 0.0 else np.zeros(robot_num, dtype=float)
        except (KeyError, TypeError, ValueError):
            pass

    raise KeyError(
        "Unable to obtain final per-robot TSP path lengths/shares from mainline. "
        "Expected path_lengths/path_shares or routes[rid]['length']."
    )


# -----------------------------------------------------------------------------
# One experiment: construct mainline solver -> solve() -> collect metrics only.
# -----------------------------------------------------------------------------
def _run_one_experiment(
    solver_cls: Type,
    map_size: int,
    robot_num: int,
    target_weights: Sequence[float],
    planner_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    started = time.perf_counter()

    kwargs = _constructor_kwargs(
        solver_cls=solver_cls,
        map_size=map_size,
        robot_num=robot_num,
        target_weights=target_weights,
        planner_seed=planner_seed,
    )
    solver = solver_cls(**kwargs)
    result_obj = solver.solve()
    result = _as_result_mapping(result_obj)

    weight_ratios = _extract_weight_ratios(result, solver, target_weights, robot_num)

    partition_sizes = _extract_partition_sizes(result, solver, robot_num)
    total_area = float(partition_sizes.sum())
    area_ratios = (
        partition_sizes / total_area
        if total_area > 0.0
        else np.zeros(robot_num, dtype=float)
    )

    tsp_ratios = _extract_tsp_ratios(result, solver, robot_num)

    elapsed = time.perf_counter() - started
    return weight_ratios, area_ratios, tsp_ratios, float(elapsed)


# -----------------------------------------------------------------------------
# Lab-C experiment controller
# -----------------------------------------------------------------------------
def run_correlation_analysis_experiment() -> None:
    # Current real layout from the terminal:
    # MCCA-PRO/labC_Correlation_Analysis.py
    project_root = Path(__file__).resolve().parent

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "LAB_DATA" / f"labC_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_filepath = output_dir / f"experiment_C_Correlation-Analysis_{timestamp}.csv"

    mainline_path = _find_mainline_path(project_root)
    mainline = _load_mainline(mainline_path)
    solver_cls = _resolve_solver_class(mainline)

    # Keep Lab-C experiment-parameter randomness isolated from planner randomness.
    # This prevents mainline's internal RNG consumption from changing later Lab-C cases.
    rng = np.random.RandomState(EXPERIMENT_RANDOM_SEED)

    all_target_weight_ratios: list[float] = []
    all_actual_area_ratios: list[float] = []
    all_actual_tsp_ratios: list[float] = []

    print(f"🚀 Starting {NUM_EXPERIMENTS} multi-robot load balancing experiments...")
    print(f"📂 Results will be saved to: {output_filepath}")
    print(f"🔗 Mainline: {mainline_path}")
    print(f"🧩 Solver class: {solver_cls.__name__}{inspect.signature(solver_cls.__init__)}\n")

    for exp_id in range(1, NUM_EXPERIMENTS + 1):
        map_size = int(rng.randint(MAP_SIZE_MIN, MAP_SIZE_MAX + 1))
        robot_num = int(rng.randint(ROBOT_NUM_MIN, ROBOT_NUM_MAX + 1))

        raw_weights = rng.uniform(WEIGHT_MIN, WEIGHT_MAX, robot_num)
        target_weights = raw_weights / np.sum(raw_weights)

        # Separate deterministic planner seed: does not consume the Lab-C parameter RNG.
        planner_seed = EXPERIMENT_RANDOM_SEED + exp_id - 1

        weight_ratios, area_ratios, tsp_ratios, elapsed = _run_one_experiment(
            solver_cls=solver_cls,
            map_size=map_size,
            robot_num=robot_num,
            target_weights=target_weights,
            planner_seed=planner_seed,
        )

        all_target_weight_ratios.extend(weight_ratios.tolist())
        all_actual_area_ratios.extend(area_ratios.tolist())
        all_actual_tsp_ratios.extend(tsp_ratios.tolist())

        print(
            f"[*] Experiment {exp_id:>3}/{NUM_EXPERIMENTS} | "
            f"Map {map_size:>3}x{map_size:<3} | "
            f"Robots {robot_num} | Time: {elapsed:.2f}s"
        )

    with output_filepath.open(mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Target_Weight_Ratio",
                "Actual_Area_Ratio",
                "Actual_TSP_Ratio",
            ]
        )
        writer.writerows(
            zip(
                all_target_weight_ratios,
                all_actual_area_ratios,
                all_actual_tsp_ratios,
            )
        )

    print(f"\n✅ Experiments completed. Data has been saved to:\n{output_filepath}")
    print("👉 You can now run the visualization script using this CSV file.")


if __name__ == "__main__":
    run_correlation_analysis_experiment()
