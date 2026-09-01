from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mcpp_compare.experiment import SCENARIOS, _path_style_for_result
from mcpp_compare.grid import GridMap
from mcpp_compare.mcca_pro import MCCAPROPlanner
from mcpp_compare.path_audit import audit_path_closure
from mcpp_compare.official_adapters import (
    OfficialLSMCPPPlanner,
    OfficialMFCPlanner,
    OfficialMSTCStarPlanner,
)
from mcpp_compare.planners import (
    AdaptiveVoronoiMSTPlanner,
    DARPAdaptiveMSTPlanner,
    DARPBoustrophedonPlanner,
    DARPPlanner,
    DARPThreeTileMSTPlanner,
    VoronoiThreeTileMSTPlanner,
)
from mcpp_compare.sensor_planners import SCoPPGridPlanner


def planner_suite(seed: int):
    return [
        MCCAPROPlanner(seed=seed),
        AdaptiveVoronoiMSTPlanner(),
        VoronoiThreeTileMSTPlanner(),
        DARPAdaptiveMSTPlanner(),
        DARPThreeTileMSTPlanner(),
        DARPBoustrophedonPlanner(),
        SCoPPGridPlanner(),
        DARPPlanner(),
        OfficialMFCPlanner(),
        OfficialMSTCStarPlanner(cut_off_opt=True),
        OfficialLSMCPPPlanner(iterations=80, seed=seed),
    ]


def main() -> None:
    rows = []
    for scenario in SCENARIOS:
        for seed in scenario["seeds"]:
            grid = GridMap.random_with_filled_disconnected(
                scenario["width"], scenario["height"], scenario["obstacle_ratio"], seed
            )
            weights = np.ones(scenario["robots"]) / scenario["robots"]
            for planner in planner_suite(seed):
                result = planner.plan(grid, scenario["robots"], weights)
                rows.extend(
                    audit_path_closure(
                        grid,
                        result,
                        scenario["name"],
                        seed,
                        _path_style_for_result(result.method),
                    )
                )

    output = Path("outputs") / "path_closure_audit.csv"
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    summary = frame.groupby("method", as_index=False).agg(
        paths=("explicitly_closed", "size"),
        open_paths=("explicitly_closed", lambda values: int((~values).sum())),
        max_closure_gap=("closure_gap", "max"),
    )
    print(summary.to_string(index=False))
    print(f"\nDetailed audit: {output}")


if __name__ == "__main__":
    main()
