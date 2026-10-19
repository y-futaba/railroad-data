"""駅接続グラフの構築、直径パス反復抽出による路線パス分解、および駅順ソートキー生成。"""

from collections.abc import Callable, Sequence
from itertools import chain, pairwise
from typing import Any, cast

import networkx as nx
import pandas as pd

from src.domain.types import LineGraphSummary


def has_cycle(graph: nx.Graph[str]) -> bool:
    """無向グラフに閉路（サイクル）が存在するかを判定する。"""
    try:
        nx.find_cycle(graph)  # pyright: ignore[reportUnknownMemberType]
    except nx.NetworkXNoCycle:
        return False
    else:
        return True


def _build_ordered_subgraph(parent_graph: nx.Graph[str], nodes: set[str]) -> nx.Graph[str]:
    """指定したノード集合から、ノードおよびエッジが辞書順で並んだ新しい部分グラフを構築する。"""
    sorted_nodes = sorted(nodes)
    node_set = set(sorted_nodes)
    raw_edges = parent_graph.edges(sorted_nodes)  # pyright: ignore[reportUnknownMemberType]
    sorted_edges = sorted(
        (str(u), str(v)) if str(u) <= str(v) else (str(v), str(u)) for u, v in raw_edges if u in node_set and v in node_set
    )
    sub_g: nx.Graph[str] = nx.Graph()
    sub_g.add_nodes_from(sorted_nodes)
    sub_g.add_edges_from(sorted_edges)
    return sub_g


def find_diameter_path(sub_comp: nx.Graph[str]) -> tuple[str, ...]:
    """連結な部分グラフから直径（最長最短経路）となるパスを決定論的（辞書順最小タイブレーク）に1本抽出する。"""
    if sub_comp.number_of_nodes() < 2:
        return ()

    try:
        diameter = int(nx.diameter(sub_comp))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        ecc = cast(dict[str, int], nx.eccentricity(sub_comp))  # pyright: ignore[reportUnknownMemberType]
        source_u = min(u for u, e_u in ecc.items() if e_u == diameter)
        lengths = cast(
            dict[str, int],
            nx.single_source_shortest_path_length(sub_comp, source_u),  # pyright: ignore[reportUnknownMemberType]
        )
        target_v = min(v for v, dist in lengths.items() if dist == diameter)
        raw_path = tuple(nx.shortest_path(sub_comp, source=source_u, target=target_v))  # pyright: ignore[reportUnknownMemberType]
        if len(raw_path) >= 2 and raw_path[0] > raw_path[-1]:
            return tuple(reversed(raw_path))
        return raw_path
    except nx.NetworkXError, ValueError:
        return ()


def _decompose_component_paths(
    comp_graph: nx.Graph[str],
) -> tuple[tuple[str, ...], ...]:
    """単一の連結成分グラフから、辺がなくなるまで直径パスを再帰的に抽出・除去する。"""
    if comp_graph.number_of_edges() == 0:
        return ()

    sub_nodes_list: list[set[str]] = sorted(
        nx.connected_components(comp_graph),
        key=lambda comp: (-len(comp), min(comp)),
    )
    active_subgraphs: tuple[nx.Graph[str], ...] = tuple(
        sub_g for nodes in sub_nodes_list if (sub_g := _build_ordered_subgraph(comp_graph, nodes)).number_of_edges() > 0
    )
    if not active_subgraphs:
        return ()

    extracted_paths = tuple(path for path in map(find_diameter_path, active_subgraphs) if path)
    if not extracted_paths:
        return ()

    edges_to_remove = tuple(chain.from_iterable(map(pairwise, extracted_paths)))
    if not edges_to_remove:
        return extracted_paths

    residual_graph = comp_graph.copy()
    residual_graph.remove_edges_from(edges_to_remove)
    return (*extracted_paths, *_decompose_component_paths(residual_graph))


def extract_all_diameter_paths(
    graph: nx.Graph[str],
) -> tuple[tuple[str, ...], ...]:
    """グラフの全連結成分に対して直径パスによるパス分解を決定論的に実行する。"""
    if graph.number_of_nodes() == 0:
        return ()

    comp_nodes_list: list[set[str]] = sorted(
        nx.connected_components(graph),
        key=lambda comp: (-len(comp), min(comp)),
    )
    component_graphs: tuple[nx.Graph[str], ...] = tuple(_build_ordered_subgraph(graph, c) for c in comp_nodes_list)
    return tuple(chain.from_iterable(map(_decompose_component_paths, component_graphs)))


def _extract_first_station(adjs: tuple[str, ...]) -> str:
    """駅タプルから先頭の駅名を取得する。"""
    return adjs[0]


def analyze_line_graph(co: str, line: str, adj_stas_series: pd.Series[Any]) -> tuple[LineGraphSummary, nx.Graph[str] | None]:
    """1路線の `adj_stas` 列から駅接続グラフを構築し、閉路・連結判定およびパス分解を行う。"""
    lengths = adj_stas_series.map(len)
    edge_pairs: tuple[tuple[str, str], ...] = tuple(
        sorted({(str(pair[0]), str(pair[1])) for pair in adj_stas_series[lengths == 2].unique()})
    )

    if not edge_pairs:
        single_stations = tuple(sorted(set(adj_stas_series.map(_extract_first_station).unique())))
        fallback_path = ("*".join(single_stations),) if single_stations else ()
        return (
            LineGraphSummary(
                co=co,
                line=line,
                cycled=False,
                connected=True,
                paths=(fallback_path,) if fallback_path else (),
            ),
            None,
        )

    all_nodes = sorted({sta for pair in edge_pairs for sta in pair})
    graph: nx.Graph[str] = nx.Graph()
    graph.add_nodes_from(all_nodes)
    graph.add_edges_from(edge_pairs)
    cycled = has_cycle(graph)
    connected = bool(nx.is_connected(graph))
    paths = extract_all_diameter_paths(graph)

    return (
        LineGraphSummary(
            co=co,
            line=line,
            cycled=cycled,
            connected=connected,
            paths=paths,
        ),
        graph,
    )


def sorter_factory(
    paths_list: Sequence[Sequence[str]],
) -> Callable[[pd.Series[Any]], pd.Series[float]]:
    """抽出されたパス列から、エッジ（駅ペア）とノード（単一駅）を駅順に並べるソートキー関数を生成する。"""
    indexed_steps = tuple(enumerate(chain.from_iterable(map(pairwise, paths_list))))

    edge_to_order: dict[tuple[str, ...], float] = {tuple(sorted((u, v))): float(k) for k, (u, v) in reversed(indexed_steps)}

    node_candidates = tuple(chain.from_iterable(((u, k - 0.5), (v, k + 0.5)) for k, (u, v) in indexed_steps))
    node_to_order: dict[str, float] = {node: order for node, order in reversed(node_candidates)}

    def get_order_value(adjs: tuple[str, ...]) -> float:
        if len(adjs) == 2:
            return edge_to_order[adjs]
        if len(adjs) == 1:
            return node_to_order[adjs[0]]
        return float("inf")

    def sorter_key(adjs_s: pd.Series[Any]) -> pd.Series[float]:
        return adjs_s.map(get_order_value)

    return sorter_key
