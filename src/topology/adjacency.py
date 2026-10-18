"""路線 `LineString` と駅代表点 `Point` の隣接関係・トポロジー解析処理。"""

from collections.abc import Iterable, Mapping
from itertools import chain
from typing import cast

import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point

from src.domain.types import Coord
from src.geometry.geodesic import extract_coords, ordered_segments


def extract_point_coord(pt: Point) -> Coord:
    """`Point` ジオメトリから `(lon, lat)` の2次元座標タプルを抽出する。"""
    return (float(pt.x), float(pt.y))


def gen_graph_from_ls_gs(ls_gs: gpd.GeoSeries) -> nx.Graph[Coord]:
    """`LineString` の `GeoSeries` から頂点座標をノードとする無向グラフを構築する。"""
    all_segments = chain.from_iterable(ls_gs.map(ordered_segments))
    graph: nx.Graph[Coord] = nx.Graph(all_segments)
    return graph


def edge_to_adj(
    segments: Iterable[tuple[Coord, Coord] | frozenset[Coord]],
) -> dict[Coord, set[Coord]]:
    """セグメント（2頂点の組）のイテラブルから隣接リスト辞書を構築する。"""
    edges = ((s, t) for s, t in segments)
    graph: nx.Graph[Coord] = nx.Graph(edges)
    return {node: set(neighbors) for node, neighbors in graph.adj.items()}


def excluded_sta_coords(ls_gs: gpd.GeoSeries, sta_pt_gs: gpd.GeoSeries) -> set[Coord]:
    """路線の全頂点座標に含まれていない（路線から抜けてしまっている）駅代表点の座標集合を返す。"""
    all_line_coords = set(chain.from_iterable(ls_gs.map(extract_coords)))
    sta_coords = set(sta_pt_gs.map(extract_point_coord))
    return sta_coords - all_line_coords


def edge_stations(
    ls_gs: gpd.GeoSeries, sta_pt_gs: gpd.GeoSeries
) -> tuple[dict[Coord, frozenset[Coord]], dict[frozenset[Coord], frozenset[Coord]]]:
    """路線上の各非駅頂点から駅以外の頂点のみを経由して到達可能な隣接駅座標集合を算出する。

    Returns:
        - `coord_to_edge_stas`: 非駅頂点座標 -> 到達可能な駅座標の `frozenset`
        - `edge_stas_to_coords`: 到達可能な駅座標の `frozenset` -> 対応する非駅頂点座標の `frozenset`
    """
    graph = gen_graph_from_ls_gs(ls_gs)
    sta_coords = set(sta_pt_gs.map(extract_point_coord))
    non_sta_nodes = set(graph.nodes) - sta_coords

    non_sta_subgraph: nx.Graph[Coord] = graph.subgraph(non_sta_nodes)  # pyright: ignore[reportUnknownMemberType]
    components: list[set[Coord]] = list(nx.connected_components(non_sta_subgraph))  # pyright: ignore[reportUnknownMemberType]

    component_to_stas: tuple[tuple[frozenset[Coord], frozenset[Coord]], ...] = tuple(
        (
            frozenset(comp),
            frozenset(
                nbr
                for u in comp
                for nbr in graph.neighbors(u)  # pyright: ignore[reportUnknownMemberType]
                if nbr in sta_coords
            ),
        )
        for comp in components
    )

    coord_to_edge_stas: dict[Coord, frozenset[Coord]] = {coord: stas for comp, stas in component_to_stas for coord in comp}

    unique_stas_sets = {stas for _, stas in component_to_stas}
    edge_stas_to_coords: dict[frozenset[Coord], frozenset[Coord]] = {
        target_stas: frozenset(coord for comp, stas in component_to_stas if stas == target_stas for coord in comp)
        for target_stas in unique_stas_sets
    }

    return coord_to_edge_stas, edge_stas_to_coords


def adjacent_stations(
    edge_stas_to_coords: Mapping[frozenset[Coord], frozenset[Coord]],
) -> set[frozenset[Coord]]:
    """隣接駅数が2以上の駅座標集合のみを抽出する。"""
    return {stas_set for stas_set in edge_stas_to_coords if len(stas_set) >= 2}


def check_adjacent_stations_with_ls(ls_df: gpd.GeoDataFrame, sta_pt_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """各 `LineString` に属する頂点の隣接駅を判定し、`adj_stas` 列（カンマ区切り駅名または `'❗️'`）を付与する。"""
    ls_gs = cast(gpd.GeoSeries, ls_df["geometry"])
    sta_geom_gs = cast(gpd.GeoSeries, sta_pt_df["geometry"])
    coord_to_edge_stas, _ = edge_stations(ls_gs, sta_geom_gs)

    sta_coords = tuple(sta_geom_gs.map(extract_point_coord))
    sta_names = tuple(sta_pt_df["N02_005"].astype(str))
    coord_to_sta_name: dict[Coord, str] = {
        coord: name for coord, name in reversed(tuple(zip(sta_coords, sta_names, strict=True)))
    }

    def _resolve_ls_adj_stas(ls: LineString) -> str:
        edge_stas_set = {coord_to_edge_stas[coord] for coord in extract_coords(ls) if coord in coord_to_edge_stas}
        if len(edge_stas_set) == 1:
            only_edge_stas = next(iter(edge_stas_set))
            return ",".join({coord_to_sta_name[sta_coord] for sta_coord in only_edge_stas})
        return "❗️"

    return ls_df.assign(adj_stas=ls_gs.map(_resolve_ls_adj_stas))
