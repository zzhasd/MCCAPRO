from __future__ import annotations

from pathlib import Path

import pandas as pd

from .metrics import ROBOT_MAX_SPEED_M_S, ROBOT_TRACK_WIDTH_M


SOURCES = [
    ("SCoPP-QLB", "ICRA 2021 arXiv", "https://arxiv.org/abs/2103.14709"),
    ("SCoPP-QLB", "official GitHub", "https://github.com/adamslab-ub/SCoPP"),
    ("LS-MCPP / ESTC", "AAAI 2024 paper", "https://ojs.aaai.org/index.php/AAAI/article/view/29707"),
    ("LS-MCPP-official", "official GitHub", "https://github.com/reso1/LS-MCPP"),
    ("MSTC*-official", "paper", "https://arxiv.org/abs/2108.04632"),
    ("MSTC*-official", "official GitHub", "https://github.com/reso1/MSTC_Star"),
    ("MFC", "paper page", "https://idm-lab.org/bib/abstracts/Koen10s.html"),
    ("MFC-official", "MSTC_Star repository implementation", "https://github.com/reso1/MSTC_Star"),
    ("MIP-MCPP", "official GitHub, downloaded but not reported", "https://github.com/reso1/MIP_MCPP"),
    ("MIP-MCPP", "paper, downloaded but not reported", "https://arxiv.org/abs/2306.17609"),
    ("DARP", "official/GUI GitHub", "https://github.com/athakapo/DARP"),
    ("Execution-time model", "Balkcom & Mason IJRR 2002", "https://journals.sagepub.com/doi/10.1177/027836402320556403"),
    ("Execution-time model", "Balkcom & Mason ICRA 2000", "https://publications.ri.cmu.edu/time-optimal-trajectories-for-bounded-velocity-differential-drive-robots/"),
    ("Execution-time model", "Dynamic Window Approach", "https://doi.org/10.1109/100.580977"),
    ("Execution-time model", "Timed Elastic Band", "https://www.vde-verlag.de/proceedings-en/453418014.html"),
]


def write_report(output_path: Path, visualizations_enabled: bool = True) -> Path:
    paths = pd.read_csv(output_path / "path_summary.csv")
    seeds = sorted(pd.read_csv(output_path / "path_metrics.csv")["seed"].unique().tolist())
    repo_status = pd.read_csv(output_path / "official_repo_status.csv")
    failed_runs = pd.read_csv(output_path / "failed_runs.csv")
    closure_audit = pd.read_csv(output_path / "path_closure_audit.csv")
    visualization = pd.read_csv(output_path / "visualizations" / "visualization_summary.csv")

    report = output_path / "comparison_report.md"
    report.write_text(
        "\n".join(
            [
                "# MCPP Static Coverage Path Comparison Report",
                "",
                "## Summary",
                "",
                "This report retains traditional cell-visiting MCPP baselines and adds MCCA-PRO v2.3.2 by directly calling `mainline_tile_first_v2_3_2.py`, along with SCoPP-style planning over discrete monitoring cells.",
                "",
                f"The experiment uses one scenario: `medium-obstacles`, a `30 x 30` map, initial obstacle ratio `0.15`, `4` robots, and random seeds `{seeds}`. After map generation, only the largest connected free component is retained; smaller disconnected free components are filled as obstacles.",
                "",
                "SCoPP does not force each waypoint to represent a 4x4 free-cell footprint. The grid represents the discrete monitoring cells obtained after SCoPP selects cell size from UAV FOV/height, followed by quick load-balanced assignment, connectivity repair, and routing within each region.",
                "",
                f"All summary results are `mean +/- std` across {len(seeds)} random seeds. Final comparisons use path-level metrics: total path length, maximum per-robot path length, maximum per-robot kinematic execution time, path load balance, total turns, maximum per-robot turns, turn load balance, and planning runtime. Partition CSV files are also written; colored SCoPP regions represent monitoring-cell assignments.",
                "",
                "## Comparison scope",
                "",
                _method_boundary_table(),
                "",
                "## Metric definitions",
                "",
                "- `path_length_sum`: Sum of all robot path lengths; lower is better.",
                "- `max_robot_path_length`: Longest individual robot path, an approximation of makespan; lower is better.",
                f"- `max_robot_execution_time_s`: Estimate execution time for each final robot path as `path length / {ROBOT_MAX_SPEED_M_S:g} + total turn angle / (2 * {ROBOT_MAX_SPEED_M_S:g} / {ROBOT_TRACK_WIDTH_M:g})`, then take the maximum across robots, in seconds. All robots use the R1 speed and track width in paper Fig. 3 under an ideal differential-drive approximation. One coordinate unit is interpreted as 1 m; total turn angle is cumulative absolute heading change in radians. Counter-rotating in-place turns are assumed, without acceleration, slip, or waiting.",
                "- `path_load_balance_cv`: Coefficient of variation of robot path lengths; lower indicates more balanced path loads.",
                "- `turn_count_sum`: Total direction changes across all robot paths, including right-angle turns and U-turns; lower indicates smoother trajectories.",
                "- `max_robot_turn_count`: Maximum number of turns assigned to one robot; lower is better.",
                "- `turn_load_balance_cv`: Coefficient of variation of robot turn counts; lower indicates a more balanced turning workload.",
                "- `planning_runtime_ms`: Path-planning computation time in milliseconds, not mission execution time. Use the official planner runtime when returned; otherwise measure the planner core call, including DFS/STC/round-trip path generation needed to produce final paths. Excludes kinematic execution-time estimation, visualization, and metric aggregation.",
                "",
                "## Official repository status",
                "",
                _markdown_table(repo_status, ["name", "path", "available", "note"]),
                "",
                _failed_runs_section(failed_runs),
                "",
                "## Path closure audit",
                "",
                "The closure audit checks the actual metric path for every seed and robot. `mst_dfs_roundtrip_2x_mst` means DFS traverses each MST edge and returns along it, so the path explicitly returns to the root and its length equals `2 × MST`.",
                "",
                _closure_audit_table(closure_audit),
                "",
                "## Static final-path comparison",
                "",
                _markdown_table(
                    paths,
                    [
                        "scenario",
                        "method",
                        "path_length_sum_mean_pm_std",
                        "max_robot_path_length_mean_pm_std",
                        "max_robot_execution_time_s_mean_pm_std",
                        "path_load_balance_cv_mean_pm_std",
                        "turn_count_sum_mean_pm_std",
                        "max_robot_turn_count_mean_pm_std",
                        "turn_load_balance_cv_mean_pm_std",
                        "planning_runtime_ms_mean_pm_std",
                    ],
                ),
                "",
                "## Same-map visualizations",
                "",
                (
                    "Visualizations show each method separately on the first seed `11` of `medium-obstacles`, rather than overlaying multiple methods. MCCA-PRO shows the tile-first TSP paths returned by the mainline algorithm; Your-Voronoi-Adaptive-MST retains its Voronoi partitions, tile-MST, and tree round trips; SCoPP shows monitoring-cell assignments and regional paths; traditional methods show final coverage paths."
                    if visualizations_enabled
                    else "This run used `--skip-visualizations`, so only metrics and the report are refreshed; visualization PNG files are not regenerated."
                ),
                "",
                _markdown_table(
                    visualization,
                    ["method", "path_length", "path_mode", "image"],
                ),
                "",
                "## Interpretation",
                "",
                _mcca_result_line(paths, len(seeds)),
                "2. Traditional MCPP methods remain as reference methods requiring every cell to be visited.",
                "3. SCoPP-QLB is a baseline for discrete monitoring-cell assignment and routing; it does not use a 4x4 footprint.",
                "4. SCoPP-QLB paths stay within each robot's assigned region to avoid crossing other robot regions.",
                "",
                "## References",
                "",
                _sources_list(),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return report


def _method_boundary_table() -> str:
    rows = [
        ["Method", "Integration approach", "Final path source"],
        [
            "MCCA-PRO-v2.3.2",
            "Directly calls mainline_tile_first_v2_3_2.py at the repository root, using exactly the same map, robot count, weights, and random seed as other methods",
            "Connected tile partitions and depot-free obstacle-safe TSP paths returned by the mainline algorithm",
        ],
        [
            "Your-Voronoi-Adaptive-MST",
            "Local implementation: weighted geodesic Voronoi partitioning followed by 4x4/subrectangle tiling",
            "tile adjacency MST with DFS round-trip routing",
        ],
        [
            "Your-Voronoi-Adaptive-MST-3Tiles",
            "Retains weighted geodesic Voronoi partitioning; tile_layers = [(4,4), (2,2), (1,1)]",
            "tile adjacency MST with DFS round-trip routing",
        ],
        [
            "Your-DARP-Adaptive-MST",
            "Replaces the Your-Voronoi-Adaptive-MST partition with local DARP-style partitioning; retains adaptive tiling with seven tile shapes",
            "tile adjacency MST with DFS round-trip routing",
        ],
        [
            "Your-DARP-Adaptive-MST-3Tiles",
            "Local DARP-style partitioning followed by tile_layers = [(4,4), (2,2), (1,1)]",
            "tile adjacency MST with DFS round-trip routing",
        ],
        [
            "DARP-Boustrophedon-W4",
            "Local DARP-style partitioning followed by boustrophedon coverage with swath width 4",
            "Closed route that retraces the regional boustrophedon swaths to the start (twice the open-path length)",
        ],
        [
            "SCoPP-QLB",
            "clean-room grid QLB; treats free cells as the discretized SCoPP monitoring cells",
            "Load-balanced assignment, connectivity repair, and regional nearest-neighbor routing, followed by retracing the route to the start to form a closed tour",
        ],
        [
            "DARP",
            "Local clean-room DARP-style partitioning baseline",
            "STC contour path within each partition",
        ],
        [
            "MFC-official",
            "Calls the reso1/MSTC_Star repository implementation of mcpp/mfc_planner.py",
            "Paths returned by the official implementation",
        ],
        [
            "MSTC*-official",
            "Calls the reso1/MSTC_Star repository implementation of mcpp/mstc_star_planner.py",
            "Paths returned by the official implementation",
        ],
        [
            "LS-MCPP-official",
            "Calls reso1/LS-MCPP with a local NumPy 2 compatibility patch",
            "Paths returned by official local search",
        ],
        [
            "MIP-MCPP",
            "Repository downloaded but not included in the results table",
            "Requires Gurobi/gurobipy before integrating the official solver",
        ],
    ]
    return _rows_to_markdown(rows)


def _failed_runs_section(df: pd.DataFrame) -> str:
    if df.empty:
        return "No planners failed in the latest experiment."
    return "\n".join(
        [
            "The following planners failed and were excluded from the summary: ",
            "",
            _markdown_table(df, ["scenario", "seed", "method", "error"]),
        ]
    )


def _mcca_result_line(paths: pd.DataFrame, seed_count: int) -> str:
    rows = paths[paths["method"] == "MCCA-PRO-v2.3.2"]
    if rows.empty:
        return "1. MCCA-PRO-v2.3.2 has no successful results in this run."
    row = rows.iloc[0]
    comparison = ""
    baseline = paths[paths["method"] == "Your-DARP-Adaptive-MST-3Tiles"]
    if not baseline.empty and baseline.iloc[0]["max_robot_execution_time_s_mean"] > 0:
        reduction = 100.0 * (
            1.0 - row["max_robot_execution_time_s_mean"]
            / baseline.iloc[0]["max_robot_execution_time_s_mean"]
        )
        comparison = (
            f" The execution-time reduction versus Your-DARP-Adaptive-MST-3Tiles "
            f"(DARP-CPPF in manuscript Table I) is {reduction:.1f}%."
        )
    return (
        f"1. MCCA-PRO-v2.3.2 has a total final-path length over the {seed_count} shared random seeds of "
        f"`{row['path_length_sum_mean_pm_std']}`, maximum individual path length "
        f"`{row['max_robot_path_length_mean_pm_std']}`, maximum individual kinematic execution time "
        f"`{row['max_robot_execution_time_s_mean_pm_std']}`, path load-balance CV "
        f"`{row['path_load_balance_cv_mean_pm_std']}`, planning time "
        f"`{row['planning_runtime_ms_mean_pm_std']} ms`." + comparison
    )


def _closure_audit_table(df: pd.DataFrame) -> str:
    summary = df.groupby("method", as_index=False).agg(
        path_count=("explicitly_closed", "size"),
        open_path_count=("explicitly_closed", lambda values: int((~values.astype(bool)).sum())),
        max_closure_gap=("closure_gap", "max"),
        path_mode=("path_mode", lambda values: ", ".join(sorted(set(map(str, values))))),
    )
    return _markdown_table(
        summary,
        ["method", "path_count", "open_path_count", "max_closure_gap", "path_mode"],
    )


def _sources_list() -> str:
    return "\n".join(f"- {method}: [{label}]({url})" for method, label, url in SOURCES)


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    data = df[columns].copy()
    for column in data.columns:
        if pd.api.types.is_float_dtype(data[column]):
            data[column] = data[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    rows = [columns]
    rows.extend(data.astype(str).values.tolist())
    return _rows_to_markdown(rows)


def _rows_to_markdown(rows: list[list[str]]) -> str:
    rows = [[str(cell) for cell in row] for row in rows]
    header = "| " + " | ".join(rows[0]) + " |"
    separator = "| " + " | ".join(["---"] * len(rows[0])) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join([header, separator, *body])
