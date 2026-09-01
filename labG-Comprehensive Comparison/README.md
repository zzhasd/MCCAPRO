# MCPP Comprehensive Comparison

This workspace contains a reproducible comparison scaffold for grid-based
multi-robot coverage path planning (MCPP). The original
`voronoi-Adapt-MST.py` script is kept intact; new comparison code lives in
`mcpp_compare/`.

## Run

```powershell
python run_comparison.py --output-dir outputs
python run_comparison.py --output-dir outputs --skip-visualizations
python -m unittest discover -s tests
```

The main report is written to `outputs/comparison_report.md`. Raw and
aggregated static metrics are written as CSV files in the same directory.
Same-map final-path figures are written to `outputs/visualizations/`.
The default experiment suite uses one scenario: `medium-obstacles`, `30 x 30`,
obstacle ratio `0.15`, `4` robots, and seeds `[11, 13, 15, 17, 19, 21]`.
After random obstacle sampling, disconnected non-largest free components are
filled as obstacles.
Per-seed path metrics include path length, load balance, turn counts, estimated
kinematic execution time, and planning runtime. Summary CSV files report each
metric as mean, standard deviation, and a formatted `mean +/- std` column
across the seed set. `max_robot_execution_time_s` is computed per robot from
that robot's final path shape and then maximized across robots; it uses
`path_length / 0.4 + total_turn_angle / (2 * 0.4 / 0.2314)` for the shared
counter-rotating four-wheel robot model. `planning_runtime_ms` is planner
computation time, not mission execution time. When official code returns a
planner runtime, the adapter uses that returned value; otherwise it measures
the planner core call and includes final path generation needed to turn a
partition into executable paths, while excluding kinematic execution-time
estimation, visualization, and metric aggregation.

`DARP-Boustrophedon-W4` is evaluated as a closed round trip: after reaching the
end of its obstacle-safe swath route, each robot retraces that route to its
start. Its reported path length is therefore exactly twice the open route.
`SCoPP-QLB` uses the same closed-round-trip convention after its monitoring
waypoints are visited. `outputs/path_closure_audit.csv` records the first/last
point and closure gap for every seed, method, robot, and path component.

The suite directly calls repository-level `mainline_tile_first_v2_3_2.py` as
`MCCA-PRO-v2.3.2`, using the same generated map and seed as every baseline.
The in-tree suite also includes the original `Your-Voronoi-Adaptive-MST` plus four
new variants: Voronoi with three tile layers, DARP partitioning with the legacy
tile set, DARP partitioning with the three-layer tile set, and DARP followed by
boustrophedon coverage with swath width `4`.

## Structure

- `mcpp_compare/grid.py`: common 4-neighbor grid model, roots, BFS, components.
- `mcpp_compare/planners.py`: planner interface, the user method, DARP, tile-MST
  variants, and boustrophedon coverage.
- `mcpp_compare/metrics.py`: static partition diagnostics and final path metrics.
- `mcpp_compare/visualization.py`: same-map final-path figure output.
- `mcpp_compare/experiment.py`: static experiment suite.
- `mcpp_compare/report.py`: Markdown report generation with paper/GitHub links.
- `tests/test_smoke.py`: interface and coverage smoke test.

## Official GitHub Integration

Official repositories are vendored under `third_party/official/`:

- `reso1/LS-MCPP`
- `reso1/MSTC_Star`
- `reso1/MIP_MCPP`
- `athakapo/DARP`

The experiment runner calls official MSTC*, official MFC, and official LS-MCPP
when those folders are present. LS-MCPP needed a tiny NumPy-2 compatibility
patch in its local-search sampling calls. Official MIP-MCPP is downloaded but
requires Gurobi/`gurobipy` for same-map optimization, so it is recorded in the
repository status but not reported through a local proxy row. Official DARP is
Java GUI/class code; a headless Java runner can be added as the next bridge.

## Sensor / Monitoring Baselines

The comparison now includes a monitoring baseline that is closer to the
original sensor/monitoring literature than classic visit-every-cell MCPP:

- `SCoPP-QLB`: an in-tree, clean-room grid adaptation of SCoPP's quick
  load-balanced monitoring pipeline. The common grid is treated as the
  discretized monitoring-cell set produced after SCoPP applies its UAV
  FOV/height cell-size choice; no extra 4x4 footprint expansion is applied.
  The adapter repairs detached components and routes each robot inside its own
  assigned monitoring region.

The planner outputs same-map visualizations. Its colored cells are
monitoring-cell assignments for SCoPP.

## Reproduction Notes

The comparison intentionally avoids local proxy rows for methods that should be
reported through official implementations. Those methods are reported only when
a downloaded repository implementation is directly called. The official
adapters keep downloaded GPL code in `third_party/official/` and call it through
thin wrappers instead of copying official algorithms into `mcpp_compare/`.
SCoPP is the exception: its public code is built around map-specific geographic
polygons and legacy dependencies, so this repository reports a clearly labeled
grid adaptation rather than a brittle direct demo wrapper.
