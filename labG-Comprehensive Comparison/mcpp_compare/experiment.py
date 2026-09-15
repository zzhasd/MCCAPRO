from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .grid import GridMap
from .metrics import evaluate_partition, evaluate_paths
from .mcca_pro import MCCAPROPlanner
from .path_audit import audit_path_closure
from .official_adapters import (
    OfficialLSMCPPPlanner,
    OfficialMFCPlanner,
    OfficialMSTCStarPlanner,
    official_repo_statuses,
)
from .planners import (
    AdaptiveVoronoiMSTPlanner,
    DARPAdaptiveMSTPlanner,
    DARPBoustrophedonPlanner,
    DARPPlanner,
    DARPThreeTileMSTPlanner,
    Planner,
    VoronoiThreeTileMSTPlanner,
)
from .sensor_planners import SCoPPGridPlanner
from .report import write_report
from .visualization import save_path_figure, save_voronoi_mst_figure


SCENARIOS = [
    {
        "name": "medium-obstacles",
        "width": 30,
        "height": 30,
        "obstacle_ratio": 0.15,
        "robots": 4,
        "seeds": [11, 13, 15, 17, 19, 21, 23, 25, 27, 29],
    }
]

PARTITION_SUMMARY_METRICS = [
    "area_error_l1",
    "max_area_error",
    "load_balance_cv",
    "geodesic_compactness",
    "geodesic_silhouette",
    "disconnected_robots",
    "component_count",
    "runtime_ms",
]

PATH_SUMMARY_METRICS = [
    "path_length_sum",
    "max_robot_path_length",
    "max_robot_execution_time_s",
    "path_load_balance_cv",
    "turn_count_sum",
    "max_robot_turn_count",
    "turn_load_balance_cv",
    "planning_runtime_ms",
]


def run_all(output_dir: str | Path = "outputs", include_visualizations: bool = True) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    static_paths = run_static_suite(output_path)
    visualization_paths = (
        run_visualization_suite(output_path)
        if include_visualizations
        else write_visualization_placeholders(output_path)
    )
    report_path = write_report(output_path, visualizations_enabled=include_visualizations)
    return {**static_paths, **visualization_paths, "report": report_path}


def run_static_suite(output_path: Path) -> dict[str, Path]:
    status_csv = output_path / "official_repo_status.csv"
    pd.DataFrame([status.__dict__ for status in official_repo_statuses()]).to_csv(status_csv, index=False)
    partition_rows = []
    path_rows = []
    closure_rows = []
    failed_rows = []
    for scenario in SCENARIOS:
        for seed in scenario["seeds"]:
            print(f"Static suite: {scenario['name']}, seed={seed}; {len(path_rows)} runs completed", flush=True)
            grid = GridMap.random_with_filled_disconnected(
                scenario["width"],
                scenario["height"],
                scenario["obstacle_ratio"],
                seed,
            )
            robot_count = scenario["robots"]
            weights = np.ones(robot_count) / robot_count
            planners: list[Planner] = [
                MCCAPROPlanner(seed=seed),
                AdaptiveVoronoiMSTPlanner(),
                VoronoiThreeTileMSTPlanner(),
                DARPAdaptiveMSTPlanner(),
                DARPThreeTileMSTPlanner(),
                DARPBoustrophedonPlanner(),
                SCoPPGridPlanner(),
                DARPPlanner(),
            ]
            if (output_path.parents[0] / "third_party" / "official" / "MSTC_Star").exists():
                planners.extend([OfficialMFCPlanner(), OfficialMSTCStarPlanner(cut_off_opt=True)])
            if (output_path.parents[0] / "third_party" / "official" / "LS-MCPP").exists():
                planners.append(OfficialLSMCPPPlanner(iterations=80, seed=seed))
            for planner in planners:
                try:
                    result = planner.plan(grid, robot_count, weights)
                except Exception as exc:
                    failed_rows.append(
                        {
                            "scenario": scenario["name"],
                            "seed": seed,
                            "method": planner.name,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                partition_rows.append(asdict(evaluate_partition(grid, result, weights, scenario["name"], seed)))
                path_style = _path_style_for_result(result.method)
                path_rows.append(asdict(evaluate_paths(grid, result, scenario["name"], seed, path_style)))
                closure_rows.extend(audit_path_closure(grid, result, scenario["name"], seed, path_style))

    partition_df = pd.DataFrame(partition_rows)
    path_df = pd.DataFrame(path_rows)
    partition_csv = output_path / "partition_metrics.csv"
    path_csv = output_path / "path_metrics.csv"
    closure_csv = output_path / "path_closure_audit.csv"
    partition_df.to_csv(partition_csv, index=False)
    path_df.to_csv(path_csv, index=False)
    pd.DataFrame(closure_rows).to_csv(closure_csv, index=False)

    partition_summary = _summary_with_variation(
        partition_df,
        PARTITION_SUMMARY_METRICS,
        sort_by=["area_error_l1_mean", "runtime_ms_mean"],
    )
    path_summary = _summary_with_variation(
        path_df,
        PATH_SUMMARY_METRICS,
        sort_by=["max_robot_execution_time_s_mean", "max_robot_path_length_mean", "path_length_sum_mean"],
    )
    partition_summary_csv = output_path / "partition_summary.csv"
    path_summary_csv = output_path / "path_summary.csv"
    failed_csv = output_path / "failed_runs.csv"
    partition_summary.to_csv(partition_summary_csv, index=False)
    path_summary.to_csv(path_summary_csv, index=False)
    pd.DataFrame(failed_rows, columns=["scenario", "seed", "method", "error"]).to_csv(failed_csv, index=False)
    return {
        "official_repo_status": status_csv,
        "partition_metrics": partition_csv,
        "path_metrics": path_csv,
        "path_closure_audit": closure_csv,
        "partition_summary": partition_summary_csv,
        "path_summary": path_summary_csv,
        "failed_runs": failed_csv,
    }


def run_visualization_suite(output_path: Path) -> dict[str, Path]:
    config = SCENARIOS[0]
    seed = config["seeds"][0]
    scenario = f"{config['name']}-{config['width']}x{config['height']}-k{config['robots']}-seed{seed}"
    grid = GridMap.random_with_filled_disconnected(
        config["width"],
        config["height"],
        config["obstacle_ratio"],
        seed=seed,
    )
    robot_count = config["robots"]
    weights = np.ones(robot_count) / robot_count
    planners: list[Planner] = [
        MCCAPROPlanner(seed=seed),
        AdaptiveVoronoiMSTPlanner(),
        VoronoiThreeTileMSTPlanner(),
        DARPAdaptiveMSTPlanner(),
        DARPThreeTileMSTPlanner(),
        DARPBoustrophedonPlanner(),
        SCoPPGridPlanner(),
        DARPPlanner(),
    ]
    if (output_path.parents[0] / "third_party" / "official" / "MSTC_Star").exists():
        planners.extend([OfficialMFCPlanner(), OfficialMSTCStarPlanner(cut_off_opt=True)])
    if (output_path.parents[0] / "third_party" / "official" / "LS-MCPP").exists():
        planners.append(OfficialLSMCPPPlanner(iterations=80, seed=seed))
    visual_dir = output_path / "visualizations"
    records = []
    failed_rows = []
    for planner in planners:
        try:
            result = planner.plan(grid, robot_count, weights)
            path_style = _path_style_for_result(result.method)
            if result.extra.get("mst_mode") == "script_tile_mst":
                records.append(save_voronoi_mst_figure(grid, result, visual_dir, scenario))
            else:
                records.append(save_path_figure(grid, result, visual_dir, scenario, path_style))
        except Exception as exc:
            failed_rows.append(
                {
                    "scenario": scenario,
                    "method": planner.name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    summary_csv = visual_dir / "visualization_summary.csv"
    failed_csv = visual_dir / "visualization_failed.csv"
    pd.DataFrame(
        [
            {
                "scenario": record.scenario,
                "method": record.method,
                "image": str(record.image),
                "path_length": record.path_length,
                "path_mode": record.path_mode,
            }
            for record in records
        ],
        columns=["scenario", "method", "image", "path_length", "path_mode"],
    ).to_csv(summary_csv, index=False)
    pd.DataFrame(failed_rows, columns=["scenario", "method", "error"]).to_csv(failed_csv, index=False)
    return {"visualization_summary": summary_csv, "visualization_failed": failed_csv}


def write_visualization_placeholders(output_path: Path) -> dict[str, Path]:
    visual_dir = output_path / "visualizations"
    visual_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = visual_dir / "visualization_summary.csv"
    failed_csv = visual_dir / "visualization_failed.csv"
    pd.DataFrame(columns=["scenario", "method", "image", "path_length", "path_mode"]).to_csv(summary_csv, index=False)
    pd.DataFrame(columns=["scenario", "method", "error"]).to_csv(failed_csv, index=False)
    return {"visualization_summary": summary_csv, "visualization_failed": failed_csv}


def _path_style_for_result(method: str) -> str:
    return "stc"


def _summary_with_variation(
    df: pd.DataFrame,
    metrics: list[str],
    sort_by: list[str],
) -> pd.DataFrame:
    agg_spec = {}
    for metric in metrics:
        agg_spec[f"{metric}_mean"] = (metric, "mean")
        agg_spec[f"{metric}_std"] = (metric, "std")
    summary = df.groupby(["scenario", "method"], as_index=False).agg(**agg_spec)
    for metric in metrics:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        summary[std_col] = summary[std_col].fillna(0.0)
        summary[f"{metric}_mean_pm_std"] = summary.apply(
            lambda row: f"{row[mean_col]:.4f} +/- {row[std_col]:.4f}",
            axis=1,
        )
    return summary.sort_values(["scenario", *sort_by])
