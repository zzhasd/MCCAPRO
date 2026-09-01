from __future__ import annotations

import contextlib
import importlib
import io
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterator

import networkx as nx
import numpy as np

from .grid import Coord, GridMap, choose_roots
from .planners import PlanResult


WORKSPACE = Path(__file__).resolve().parents[1]
OFFICIAL_ROOT = WORKSPACE / "third_party" / "official"


class OfficialAdapterError(RuntimeError):
    pass


@dataclass(frozen=True)
class OfficialRepoStatus:
    name: str
    path: Path
    available: bool
    note: str


def official_repo_statuses() -> list[OfficialRepoStatus]:
    repos = {
        "LS-MCPP": OFFICIAL_ROOT / "LS-MCPP",
        "MSTC_Star": OFFICIAL_ROOT / "MSTC_Star",
        "MIP_MCPP": OFFICIAL_ROOT / "MIP_MCPP",
        "DARP": OFFICIAL_ROOT / "DARP",
        "SCoPP": OFFICIAL_ROOT / "SCoPP",
    }
    statuses = []
    for name, path in repos.items():
        if name == "SCoPP":
            note = (
                "downloaded; current report uses the in-tree grid QLB adapter"
                if path.exists()
                else "not vendored; current report uses the in-tree grid QLB adapter because the official demo has map-specific inputs and legacy deps"
            )
        else:
            note = "downloaded" if path.exists() else "missing; run git clone first"
        statuses.append(
            OfficialRepoStatus(
                name=name,
                path=path,
                available=path.exists(),
                note=note,
            )
        )
    return statuses


class OfficialMSTCStarPlanner:
    name = "MSTC*-official"

    def __init__(self, cut_off_opt: bool = True) -> None:
        self.cut_off_opt = cut_off_opt

    def plan(
        self,
        grid: GridMap,
        robot_count: int,
        weights: list[float] | np.ndarray,
        previous: dict[Coord, int] | None = None,
    ) -> PlanResult:
        adapter_started = perf_counter()
        repo = _require_repo("MSTC_Star")
        roots = choose_roots(grid, robot_count)
        graph = _grid_to_nx_graph(grid)
        with _isolated_sys_path(repo, ("mcpp", "utils")):
            module = importlib.import_module("mcpp.mstc_star_planner")
            official_cls = module.MSTCStarPlanner
            core_started = perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                planner = official_cls(graph, robot_count, roots, float("inf"), self.cut_off_opt)
                plans = planner.allocate()
                paths, official_costs = planner.simulate(plans, is_print=False)
            core_runtime_s = perf_counter() - core_started
        assignments, missing = _assignments_from_robot_points(grid, list(plans.values()), roots)
        return PlanResult(
            self.name,
            assignments,
            roots,
            core_runtime_s,
            notes=f"Official reso1/MSTC_Star code. Colors are a grid-cell coverage projection, not a native partition; missing cells filled by nearest root: {missing}.",
            extra={
                "official_max_cost": float(max(official_costs) if official_costs else 0),
                "adapter_wall_runtime_s": float(perf_counter() - adapter_started),
                "runtime_source": "measured_official_constructor_allocate_simulate",
                "allocation_missing_cells": missing,
                "projection_kind": "coverage_projection",
                "projection_components": _component_count(grid, assignments),
            },
            paths=paths,
        )


class OfficialMFCPlanner:
    name = "MFC-official"

    def plan(
        self,
        grid: GridMap,
        robot_count: int,
        weights: list[float] | np.ndarray,
        previous: dict[Coord, int] | None = None,
    ) -> PlanResult:
        adapter_started = perf_counter()
        repo = _require_repo("MSTC_Star")
        roots = choose_roots(grid, robot_count)
        graph = _grid_to_nx_graph(grid)
        with _isolated_sys_path(repo, ("mcpp", "utils")):
            module = importlib.import_module("mcpp.mfc_planner")
            official_cls = module.MFCPlanner
            core_started = perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                planner = official_cls(graph, robot_count, roots, float("inf"))
                plans = planner.allocate(epsilon=1.0, debug=False)
                paths, official_costs = planner.simulate(plans)
            core_runtime_s = perf_counter() - core_started
        assignments, missing = _assignments_from_robot_points(grid, list(plans.values()), roots)
        return PlanResult(
            self.name,
            assignments,
            roots,
            core_runtime_s,
            notes=f"Official MFC implementation from reso1/MSTC_Star. Colors are a grid-cell coverage projection, not a native partition; missing cells filled by nearest root: {missing}.",
            extra={
                "official_max_cost": float(max(official_costs) if official_costs else 0),
                "adapter_wall_runtime_s": float(perf_counter() - adapter_started),
                "runtime_source": "measured_official_constructor_allocate_simulate",
                "allocation_missing_cells": missing,
                "projection_kind": "coverage_projection",
                "projection_components": _component_count(grid, assignments),
            },
            paths=paths,
        )


class OfficialLSMCPPPlanner:
    name = "LS-MCPP-official"

    def __init__(self, iterations: int = 300, seed: int = 0) -> None:
        self.iterations = iterations
        self.seed = seed

    def plan(
        self,
        grid: GridMap,
        robot_count: int,
        weights: list[float] | np.ndarray,
        previous: dict[Coord, int] | None = None,
    ) -> PlanResult:
        adapter_started = perf_counter()
        repo = _require_repo("LS-MCPP")
        roots = choose_roots(grid, robot_count)
        with _lsmcpp_package(repo):
            planners = importlib.import_module("lsmcpp.planners")
            local_search = importlib.import_module("lsmcpp.local_search")
            pool = importlib.import_module("lsmcpp.pool")
            with contextlib.redirect_stdout(io.StringIO()):
                mcpp = _grid_to_lsmcpp_instance(grid, roots)
                init_sol = planners.MFC_planner(mcpp)
                planner = local_search.LocalSearchMCPP(
                    mcpp,
                    init_sol,
                    pool.PrioType.CompositeHeur,
                    pool.PoolType.VertexEdgewise,
                    verbose=False,
                )
                solution, official_runtime = planner.run(
                    M=self.iterations,
                    S=max(1, self.iterations // 20),
                    alpha=float(np.exp(np.log(0.2) / max(self.iterations, 1))),
                    gamma=0.01,
                    sample_type=pool.SampleType.RouletteWheel,
                    seed=self.seed,
                )
        paths = [list(path) for path in solution.Pi]
        assignments, missing = _assignments_from_robot_points(grid, paths, roots)
        return PlanResult(
            self.name,
            assignments,
            roots,
            float(official_runtime),
            notes=f"Official reso1/LS-MCPP local search, M={self.iterations}. Colors are a grid-cell coverage projection, not a native partition; missing cells filled by nearest root: {missing}.",
            extra={
                "official_runtime_s": float(official_runtime),
                "adapter_wall_runtime_s": float(perf_counter() - adapter_started),
                "runtime_source": "official_LocalSearchMCPP.run_returned_runtime",
                "official_tau": float(solution.tau),
                "allocation_missing_cells": missing,
                "projection_kind": "coverage_projection",
                "projection_components": _component_count(grid, assignments),
            },
            paths=paths,
        )


def _require_repo(name: str) -> Path:
    path = OFFICIAL_ROOT / name
    if not path.exists():
        raise OfficialAdapterError(f"official repository not found: {path}")
    return path


def _grid_to_nx_graph(grid: GridMap) -> nx.Graph:
    graph = nx.Graph()
    for cell in grid.free_cells:
        graph.add_node(cell)
    for cell in grid.free_cells:
        for nbr in grid.neighbors(cell):
            if cell < nbr:
                graph.add_edge(cell, nbr, weight=1.0)
    return graph


def _grid_to_lsmcpp_instance(grid: GridMap, roots: list[Coord]):
    instance_module = importlib.import_module("lsmcpp.benchmark.instance")
    graph = nx.Graph()
    node_id = 0
    subnodes: dict[tuple[Coord, str], int] = {}
    legacy_pos: dict[tuple[Coord, str], tuple[float, float]] = {}
    offsets = {
        "SE": (0.25, -0.25),
        "NE": (0.25, 0.25),
        "NW": (-0.25, 0.25),
        "SW": (-0.25, -0.25),
    }
    for cell in grid.free_cells:
        x, y = cell
        for label, (dx, dy) in offsets.items():
            dpos = (x + dx, y + dy)
            subnodes[(cell, label)] = node_id
            legacy_pos[(cell, label)] = dpos
            graph.add_node(node_id, pos=(2 * dpos[0] + 0.5, 2 * dpos[1] + 0.5))
            node_id += 1
    for cell in grid.free_cells:
        for first, second in [("SW", "SE"), ("SE", "NE"), ("NE", "NW"), ("NW", "SW")]:
            graph.add_edge(subnodes[(cell, first)], subnodes[(cell, second)], weight=1.0)
        x, y = cell
        east = (x + 1, y)
        north = (x, y + 1)
        if grid.is_free(east):
            graph.add_edge(subnodes[(cell, "SE")], subnodes[(east, "SW")], weight=1.0)
            graph.add_edge(subnodes[(cell, "NE")], subnodes[(east, "NW")], weight=1.0)
        if grid.is_free(north):
            graph.add_edge(subnodes[(cell, "NE")], subnodes[(north, "SE")], weight=1.0)
            graph.add_edge(subnodes[(cell, "NW")], subnodes[(north, "SW")], weight=1.0)
    root_nodes = [subnodes[(root, "SE")] for root in roots]
    obstacles = [(2 * x + 0.5, 2 * y + 0.5) for x, y in grid.obstacles]
    return instance_module.MCPP(
        graph,
        root_nodes,
        f"{grid.width}x{grid.height}-generated-k{len(roots)}",
        grid.width,
        grid.height,
        obstacles,
        incomplete=bool(grid.obstacles),
        weighted=False,
    )


def _assignments_from_paths(
    grid: GridMap,
    paths: list[list[tuple[float, float]]],
    roots: list[Coord],
) -> tuple[dict[Coord, int], int]:
    assignments: dict[Coord, int] = {}
    free = set(grid.free_cells)
    for rid, path in enumerate(paths):
        for point in path:
            cell = _point_to_cell(point)
            if cell in free:
                assignments.setdefault(cell, rid)
    missing_cells = [cell for cell in grid.free_cells if cell not in assignments]
    root_distances = [grid.bfs_distances(root) for root in roots]
    for cell in missing_cells:
        rid = min(range(len(roots)), key=lambda idx: root_distances[idx].get(cell, 10**9))
        assignments[cell] = rid
    return assignments, len(missing_cells)


def _assignments_from_robot_points(
    grid: GridMap,
    robot_points: list[list[tuple[float, float]]],
    roots: list[Coord],
) -> tuple[dict[Coord, int], int]:
    votes: dict[Coord, dict[int, int]] = {cell: {} for cell in grid.free_cells}
    free = set(grid.free_cells)
    for rid, points in enumerate(robot_points):
        for point in points:
            cell = _point_to_cell(point)
            if cell in free:
                votes[cell][rid] = votes[cell].get(rid, 0) + 1

    assignments: dict[Coord, int] = {}
    for cell, cell_votes in votes.items():
        if not cell_votes:
            continue
        assignments[cell] = max(cell_votes, key=lambda rid: (cell_votes[rid], -rid))

    missing_cells = [cell for cell in grid.free_cells if cell not in assignments]
    root_distances = [grid.bfs_distances(root) for root in roots]
    for cell in missing_cells:
        rid = min(range(len(roots)), key=lambda idx: root_distances[idx].get(cell, 10**9))
        assignments[cell] = rid
    return assignments, len(missing_cells)


def _connectivity_repaired_projection(
    grid: GridMap,
    assignments: dict[Coord, int],
    roots: list[Coord],
) -> dict[Coord, int]:
    repaired = dict(assignments)
    for rid, root in enumerate(roots):
        if root in repaired:
            repaired[root] = rid

    for _ in range(len(roots) * 20):
        main_components: dict[int, set[Coord]] = {}
        detached: list[tuple[int, set[Coord]]] = []
        changed = False
        for rid in range(len(roots)):
            cells = {cell for cell, assigned in repaired.items() if assigned == rid}
            components = grid.connected_components(cells)
            if not components:
                continue
            root = roots[rid]
            main = next((component for component in components if root in component), components[0])
            main_components[rid] = main
            for component in components:
                if component is not main:
                    detached.append((rid, component))

        if not detached:
            break

        for old_rid, component in detached:
            candidates: dict[int, int] = {}
            for cell in component:
                for nbr in grid.neighbors(cell):
                    nbr_rid = repaired.get(nbr)
                    if nbr_rid is None or nbr_rid == old_rid:
                        continue
                    if nbr in main_components.get(nbr_rid, set()):
                        candidates[nbr_rid] = candidates.get(nbr_rid, 0) + 2
                    else:
                        candidates[nbr_rid] = candidates.get(nbr_rid, 0) + 1
            if not candidates:
                continue
            new_rid = max(candidates, key=lambda rid: (candidates[rid], -rid))
            for cell in component:
                repaired[cell] = new_rid
            changed = True

        if not changed:
            break
    return repaired


def _component_count(grid: GridMap, assignments: dict[Coord, int]) -> int:
    return sum(
        len(grid.connected_components([cell for cell, rid in assignments.items() if rid == robot]))
        for robot in sorted(set(assignments.values()))
    )


def _point_to_cell(point: tuple[float, float]) -> Coord:
    return int(round(point[0])), int(round(point[1]))


@contextlib.contextmanager
def _isolated_sys_path(repo: Path, prefixes: tuple[str, ...]) -> Iterator[None]:
    old_path = list(sys.path)
    saved = {name: module for name, module in sys.modules.items() if _matches_prefix(name, prefixes)}
    for name in list(sys.modules):
        if _matches_prefix(name, prefixes):
            del sys.modules[name]
    sys.path.insert(0, str(repo))
    try:
        yield
    finally:
        for name in list(sys.modules):
            if _matches_prefix(name, prefixes):
                del sys.modules[name]
        sys.modules.update(saved)
        sys.path = old_path


@contextlib.contextmanager
def _lsmcpp_package(repo: Path) -> Iterator[None]:
    prefixes = ("lsmcpp",)
    saved = {name: module for name, module in sys.modules.items() if _matches_prefix(name, prefixes)}
    for name in list(sys.modules):
        if _matches_prefix(name, prefixes):
            del sys.modules[name]
    package = types.ModuleType("lsmcpp")
    package.__path__ = [str(repo / "mcpp")]
    benchmark = types.ModuleType("lsmcpp.benchmark")
    benchmark.__path__ = [str(repo / "benchmark")]
    conflict_solver = types.ModuleType("lsmcpp.conflict_solver")
    conflict_solver.__path__ = [str(repo / "conflict_solver")]
    turn_minimization = types.ModuleType("lsmcpp.turn_minimization")
    turn_minimization.__path__ = [str(repo / "turn_minimization")]
    sys.modules["lsmcpp"] = package
    sys.modules["lsmcpp.benchmark"] = benchmark
    sys.modules["lsmcpp.conflict_solver"] = conflict_solver
    sys.modules["lsmcpp.turn_minimization"] = turn_minimization
    try:
        yield
    finally:
        for name in list(sys.modules):
            if _matches_prefix(name, prefixes):
                del sys.modules[name]
        sys.modules.update(saved)


def _matches_prefix(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
