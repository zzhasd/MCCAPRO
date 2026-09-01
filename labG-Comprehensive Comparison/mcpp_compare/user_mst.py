from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from .grid import Coord, GridMap


Point = tuple[float, float]
Tile = tuple[int, int, int, int, int]
TileLayer = tuple[int, int]
DEFAULT_TILE_LAYERS: tuple[TileLayer, ...] = (
    (4, 4),
    (4, 2),
    (2, 4),
    (2, 2),
    (2, 1),
    (1, 2),
    (1, 1),
)
THREE_TILE_LAYERS: tuple[TileLayer, ...] = ((4, 4), (2, 2), (1, 1))


@dataclass(frozen=True)
class TiledMST:
    """MST result produced by the legacy rectangular tiling routine."""

    tiles: list[Tile]
    edges_by_robot: dict[int, list[tuple[Point, Point]]]
    lengths_by_robot: dict[int, float]
    centroids_by_robot: dict[int, list[Point]]
    tile_stats: dict[int, dict[str, int]]

    @property
    def total_length(self) -> float:
        return float(sum(self.lengths_by_robot.values()))

    @property
    def tile_count(self) -> int:
        return len(self.tiles)


def compute_tiled_mst(
    grid: GridMap,
    assignments: dict[Coord, int],
    robot_count: int,
    tile_layers: Sequence[TileLayer] | None = None,
) -> TiledMST:
    """Reproduce the user's script: tile each region, connect adjacent tiles, then MST."""

    normalized_tile_layers = normalize_tile_layers(tile_layers)
    order1_x, order1_y = list(range(grid.width)), list(range(grid.height))
    order2_x, order2_y = list(range(grid.width - 1, -1, -1)), list(range(grid.height - 1, -1, -1))
    order3_x, order3_y = list(range(grid.width)), list(range(grid.height - 1, -1, -1))
    order4_x, order4_y = list(range(grid.width - 1, -1, -1)), list(range(grid.height))
    scan_orders = [(order1_x, order1_y), (order2_x, order2_y), (order3_x, order3_y), (order4_x, order4_y)]

    all_tiles: list[Tile] = []
    edges_by_robot: dict[int, list[tuple[Point, Point]]] = {rid: [] for rid in range(robot_count)}
    lengths_by_robot: dict[int, float] = {rid: 0.0 for rid in range(robot_count)}
    centroids_by_robot: dict[int, list[Point]] = {rid: [] for rid in range(robot_count)}
    tile_stat_keys = list(dict.fromkeys(f"{w}x{h}" for w, h in normalized_tile_layers))
    if "1x1" not in tile_stat_keys:
        tile_stat_keys.append("1x1")
    tile_stats = {rid: {key: 0 for key in tile_stat_keys} for rid in range(robot_count)}

    for rid in range(robot_count):
        region_mask = np.zeros((grid.width, grid.height), dtype=bool)
        for (x, y), assigned in assignments.items():
            if assigned == rid and grid.is_free((x, y)):
                region_mask[x, y] = True

        schemes = [
            _tile_in_order(region_mask, x_order, y_order, rid, grid.width, grid.height, normalized_tile_layers)
            for x_order, y_order in scan_orders
        ]
        schemes.sort(key=lambda item: item[1])
        best_tiles = schemes[0][0]
        all_tiles.extend(best_tiles)

        for x, y, width, height, _ in best_tiles:
            tile_stats[rid][f"{width}x{height}"] += 1

        if len(best_tiles) <= 1:
            if best_tiles:
                x, y, width, height, _ = best_tiles[0]
                centroids_by_robot[rid] = [(x + width / 2.0, y + height / 2.0)]
            continue

        tile_map = np.full((grid.width, grid.height), -1, dtype=int)
        centroids = np.zeros((len(best_tiles), 2), dtype=float)
        for index, (x, y, width, height, _) in enumerate(best_tiles):
            tile_map[x : x + width, y : y + height] = index
            centroids[index] = [x + width / 2.0, y + height / 2.0]
        centroids_by_robot[rid] = [(float(x), float(y)) for x, y in centroids.tolist()]

        h_edges = np.column_stack((tile_map[:-1, :].ravel(), tile_map[1:, :].ravel()))
        v_edges = np.column_stack((tile_map[:, :-1].ravel(), tile_map[:, 1:].ravel()))
        all_edges = np.vstack((h_edges, v_edges))
        mask = (all_edges[:, 0] != -1) & (all_edges[:, 1] != -1) & (all_edges[:, 0] != all_edges[:, 1])
        valid_edges = all_edges[mask]
        if len(valid_edges) == 0:
            continue

        valid_edges.sort(axis=1)
        unique_edges = np.unique(valid_edges, axis=0)
        u = unique_edges[:, 0]
        v = unique_edges[:, 1]
        weights = np.linalg.norm(centroids[u] - centroids[v], axis=1)
        graph = coo_matrix((weights, (u, v)), shape=(len(best_tiles), len(best_tiles)))
        mst = minimum_spanning_tree(graph).tocoo()
        lengths_by_robot[rid] = float(mst.data.sum())
        edges_by_robot[rid] = [
            (
                (float(centroids[row][0]), float(centroids[row][1])),
                (float(centroids[col][0]), float(centroids[col][1])),
            )
            for row, col in zip(mst.row, mst.col)
        ]

    return TiledMST(
        tiles=all_tiles,
        edges_by_robot=edges_by_robot,
        lengths_by_robot=lengths_by_robot,
        centroids_by_robot=centroids_by_robot,
        tile_stats=tile_stats,
    )


def tiled_mst_component_walks(tiled_mst: TiledMST, robot: int) -> list[list[Point]]:
    """Return DFS round-trip walks over each connected tile-MST component."""

    adjacency: dict[Point, set[Point]] = {}
    for first, second in tiled_mst.edges_by_robot.get(robot, []):
        adjacency.setdefault(first, set()).add(second)
        adjacency.setdefault(second, set()).add(first)
    for centroid in tiled_mst.centroids_by_robot.get(robot, []):
        adjacency.setdefault(centroid, set())

    seen: set[Point] = set()
    walks: list[list[Point]] = []
    for root in sorted(adjacency):
        if root in seen:
            continue
        walk = [root]
        seen.add(root)
        _append_tiled_dfs_walk(root, None, adjacency, seen, walk)
        walks.append(walk)
    return walks


def normalize_tile_layers(tile_layers: Sequence[TileLayer] | None = None) -> tuple[TileLayer, ...]:
    layers = tuple(tile_layers or DEFAULT_TILE_LAYERS)
    if not layers:
        raise ValueError("tile_layers must not be empty")
    for width, height in layers:
        if width <= 0 or height <= 0:
            raise ValueError(f"tile dimensions must be positive, got {(width, height)}")
    return layers


def _tile_in_order(
    region_mask: np.ndarray,
    x_scan_order: list[int],
    y_scan_order: list[int],
    robot_id: int,
    width: int,
    height: int,
    tile_layers: Sequence[TileLayer],
) -> tuple[list[Tile], int]:
    covered_mask = np.zeros_like(region_mask, dtype=bool)
    tiles: list[Tile] = []

    for tile_width, tile_height in tile_layers:
        for x in x_scan_order:
            for y in y_scan_order:
                if covered_mask[x, y] or not region_mask[x, y]:
                    continue
                if x + tile_width > width or y + tile_height > height:
                    continue

                can_place = True
                for px in range(x, x + tile_width):
                    for py in range(y, y + tile_height):
                        if not region_mask[px, py] or covered_mask[px, py]:
                            can_place = False
                            break
                    if not can_place:
                        break

                if can_place:
                    tiles.append((x, y, tile_width, tile_height, robot_id))
                    for px in range(x, x + tile_width):
                        for py in range(y, y + tile_height):
                            covered_mask[px, py] = True

    for x in x_scan_order:
        for y in y_scan_order:
            if not covered_mask[x, y] and region_mask[x, y]:
                tiles.append((x, y, 1, 1, robot_id))
                covered_mask[x, y] = True

    return tiles, len(tiles)


def _append_tiled_dfs_walk(
    node: Point,
    parent: Point | None,
    adjacency: dict[Point, set[Point]],
    seen: set[Point],
    path: list[Point],
) -> None:
    for nxt in sorted(adjacency[node]):
        if nxt == parent or nxt in seen:
            continue
        seen.add(nxt)
        path.append(nxt)
        _append_tiled_dfs_walk(nxt, node, adjacency, seen, path)
        path.append(node)
