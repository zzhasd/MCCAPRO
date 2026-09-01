from __future__ import annotations

from pathlib import Path

import pandas as pd


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
    repo_status = pd.read_csv(output_path / "official_repo_status.csv")
    failed_runs = pd.read_csv(output_path / "failed_runs.csv")
    closure_audit = pd.read_csv(output_path / "path_closure_audit.csv")
    visualization = pd.read_csv(output_path / "visualizations" / "visualization_summary.csv")

    report = output_path / "comparison_report.md"
    report.write_text(
        "\n".join(
            [
                "# MCPP 静态覆盖路径对比报告",
                "",
                "## 摘要",
                "",
                "本报告保留传统逐格访问类 MCPP baseline，并加入直接调用 `mainline_tile_first_v2_3_2.py` 的 MCCA-PRO v2.3.2，以及 SCoPP 风格的离散监测单元规划。",
                "",
                "当前实验只保留一个场景：`medium-obstacles`，地图 `30 x 30`，初始障碍比例 `0.15`，机器人数量 `4`，随机种子 `[11, 13, 15, 17, 19, 21]`。地图生成后会保留最大自由连通块，并把其他不连通的小自由块填成障碍。",
                "",
                "SCoPP 不把一个 waypoint 强行当成 4x4 free-cell footprint。当前 grid 被视作 SCoPP 在 UAV FOV/高度确定 cell size 之后得到的离散 monitoring cells，然后做 quick load-balanced 分配、连通修复和区域内路径规划。",
                "",
                "汇总表中的每个结果均为 `mean +/- std`，波动来自 6 个随机种子。最终对比使用路径层面的指标：最终路径总长度、最大单机路径长度、最大单机运动学执行时间、路径负载均衡、转弯总量、最大单机转弯量、转弯负载均衡和规划计算时间。分区 CSV 仍会输出，SCoPP 的彩色区域表示 monitoring-cell 分配。",
                "",
                "## 对比边界",
                "",
                _method_boundary_table(),
                "",
                "## 指标定义",
                "",
                "- `path_length_sum`：所有机器人最终路径长度之和，越低越好。",
                "- `max_robot_path_length`：最长单机器人路径长度，近似 makespan，越低越好。",
                "- `max_robot_execution_time_s`：先对每个机器人的最终路径按 `路径长度 / 0.4 + 累计转角 / (2 * 0.4 / 0.2314)` 估计执行时间，再取机器人中的最大值，单位为秒；转弯按两侧轮反向旋转的差速模型计算。",
                "- `path_load_balance_cv`：各机器人路径长度的变异系数，越低表示路径负载越均衡。",
                "- `turn_count_sum`：所有机器人路径方向变化次数之和，包含直角转弯和掉头，越低表示轨迹越平顺。",
                "- `max_robot_turn_count`：单个机器人承担的最大转弯次数，越低越好。",
                "- `turn_load_balance_cv`：各机器人转弯次数的变异系数，越低表示转弯负担越均衡。",
                "- `planning_runtime_ms`：路径规划算法的计算时间，单位为毫秒，不是机器人任务执行时间；若官方代码返回 planner runtime，则直接采用官方返回值，否则测量 planner 核心调用段，并把为了得到最终路径而执行的 DFS/STC/round-trip path generation 计入；不包含运动学执行时间估计、可视化和指标汇总。",
                "",
                "## 官方仓库状态",
                "",
                _markdown_table(repo_status, ["name", "path", "available", "note"]),
                "",
                _failed_runs_section(failed_runs),
                "",
                "## 路径闭环审计",
                "",
                "闭环审计逐条检查每个种子、每台机器人的实际指标路径。`mst_dfs_roundtrip_2x_mst` 表示 DFS 遍历每条 MST 边后原路返回，因此路径显式回到根节点，长度严格等于 `2 × MST`。",
                "",
                _closure_audit_table(closure_audit),
                "",
                "## 静态最终路径对比",
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
                "## 同图可视化",
                "",
                (
                    "可视化不生成多方法叠加图，而是在 `medium-obstacles` 的第一个随机种子 `11` 上分别输出每个方法的最终路径。MCCA-PRO 显示主线算法返回的 tile-first TSP 路径；Your-Voronoi-Adaptive-MST 保留其 Voronoi 分区、tile-MST 和绕树往返路径；SCoPP 显示 monitoring-cell 分配和区域内路径；传统方法显示最终覆盖路径。"
                    if visualizations_enabled
                    else "本次运行使用了 `--skip-visualizations`，因此只刷新指标和报告，不重新生成可视化 PNG。"
                ),
                "",
                _markdown_table(
                    visualization,
                    ["method", "path_length", "path_mode", "image"],
                ),
                "",
                "## 结论口径",
                "",
                _mcca_result_line(paths),
                "2. 传统 MCPP 方法仍作为必须逐格访问的参照组保留。",
                "3. SCoPP-QLB 是离散监测单元分配与路径规划 baseline，不使用 4x4 footprint。",
                "4. SCoPP-QLB 的路径被限制在本机器人分配区域内，避免穿过其他机器人的区域。",
                "",
                "## 参考来源",
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
        ["方法", "当前接入口径", "最终路径来源"],
        [
            "MCCA-PRO-v2.3.2",
            "直接调用仓库根目录 mainline_tile_first_v2_3_2.py；使用与其他方法完全相同的地图、机器人数量、权重和随机种子",
            "主线算法返回的连通 tile 分区与 depot-free obstacle-safe TSP 路径",
        ],
        [
            "Your-Voronoi-Adaptive-MST",
            "本地实现；加权测地 Voronoi 分区后铺 4x4/子矩形 tile",
            "tile 邻接 MST 的 DFS 往返路径",
        ],
        [
            "Your-Voronoi-Adaptive-MST-3Tiles",
            "保留加权测地 Voronoi 分区；tile_layers = [(4,4), (2,2), (1,1)]",
            "tile 邻接 MST 的 DFS 往返路径",
        ],
        [
            "Your-DARP-Adaptive-MST",
            "将 Your-Voronoi-Adaptive-MST 的分区替换为本地 DARP-style 分区；保留七种砖自适应铺砖",
            "tile 邻接 MST 的 DFS 往返路径",
        ],
        [
            "Your-DARP-Adaptive-MST-3Tiles",
            "本地 DARP-style 分区后使用 tile_layers = [(4,4), (2,2), (1,1)]",
            "tile 邻接 MST 的 DFS 往返路径",
        ],
        [
            "DARP-Boustrophedon-W4",
            "本地 DARP-style 分区后使用牛耕法，作业宽幅为 4",
            "区域内宽幅牛耕覆盖后沿原路径返回起点的闭环路径（长度为开放路径的两倍）",
        ],
        [
            "SCoPP-QLB",
            "clean-room grid QLB；把 free cells 视为 SCoPP 离散化后的 monitoring cells",
            "负载均衡分配、连通修复后的区域内最近邻路径，并沿原路径返回起点形成闭环",
        ],
        [
            "DARP",
            "本地 clean-room DARP-style 分区基线",
            "分区内 STC contour path",
        ],
        [
            "MFC-official",
            "调用 reso1/MSTC_Star 仓库中的 mcpp/mfc_planner.py",
            "官方实现返回路径",
        ],
        [
            "MSTC*-official",
            "调用 reso1/MSTC_Star 仓库中的 mcpp/mstc_star_planner.py",
            "官方实现返回路径",
        ],
        [
            "LS-MCPP-official",
            "调用 reso1/LS-MCPP，并使用本地 NumPy 2 兼容补丁",
            "官方局部搜索返回路径",
        ],
        [
            "MIP-MCPP",
            "仓库已下载但未纳入结果表",
            "需要 Gurobi/gurobipy 后再桥接官方求解器",
        ],
    ]
    return _rows_to_markdown(rows)


def _failed_runs_section(df: pd.DataFrame) -> str:
    if df.empty:
        return "最新实验中没有 planner 运行失败。"
    return "\n".join(
        [
            "以下 planner 运行失败，已从汇总中排除：",
            "",
            _markdown_table(df, ["scenario", "seed", "method", "error"]),
        ]
    )


def _mcca_result_line(paths: pd.DataFrame) -> str:
    rows = paths[paths["method"] == "MCCA-PRO-v2.3.2"]
    if rows.empty:
        return "1. MCCA-PRO-v2.3.2 本轮没有成功结果。"
    row = rows.iloc[0]
    return (
        "1. MCCA-PRO-v2.3.2 在 6 个统一随机种子上的最终路径总长为 "
        f"`{row['path_length_sum_mean_pm_std']}`，最长单机路径为 "
        f"`{row['max_robot_path_length_mean_pm_std']}`，最大单机运动学执行时间为 "
        f"`{row['max_robot_execution_time_s_mean_pm_std']}`，路径负载均衡 CV 为 "
        f"`{row['path_load_balance_cv_mean_pm_std']}`，规划时间为 "
        f"`{row['planning_runtime_ms_mean_pm_std']} ms`。"
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
