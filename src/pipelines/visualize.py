"""駅接続グラフの描画・画像保存処理。"""

from pathlib import Path
from typing import Any, cast

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx


def _get_default_font_family() -> str:
    """Matplotlib の現在の設定から先頭のフォントファミリー名を取得する。"""
    font_family = cast(list[str] | str, plt.rcParams["font.family"])
    return font_family[0] if isinstance(font_family, list) else str(font_family)


def save_line_graph_image(graph: nx.Graph[str], output_path: Path) -> None:
    """単一路線の駅接続グラフを描画し、指定パスへ画像保存する。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(16, 16))  # pyright: ignore[reportUnknownMemberType]
    nx.draw(  # pyright: ignore[reportUnknownMemberType]
        graph,
        with_labels=True,
        node_color="white",
        node_size=2000,
        font_family=_get_default_font_family(),
        edge_color="gray",
        linewidths=1,
        edgecolors="black",
        font_size=10,
    )
    plt.savefig(output_path)  # pyright: ignore[reportUnknownMemberType]
    plt.clf()  # pyright: ignore[reportUnknownMemberType]


def show_sta_graph(
    rr_sorted: gpd.GeoDataFrame,
    co: str = "",
    line: str = "",
) -> None:
    """指定した事業者（`co`）または路線名（`line`）の駅接続グラフを色分けして描画する。"""
    is_two_sta = rr_sorted["adj_stas"].map(len) == 2

    if line == "":
        df = rr_sorted[(rr_sorted["co"] == co) & is_two_sta]
        group_col: str | None = "line"
    elif co == "":
        df = rr_sorted[(rr_sorted["line"] == line) & is_two_sta]
        group_col = "co"
    else:
        df = rr_sorted[(rr_sorted["co"] == co) & (rr_sorted["line"] == line) & is_two_sta]
        group_col = None

    adj_pairs = cast(tuple[tuple[str, str], ...], tuple(df["adj_stas"]))

    if group_col is not None:
        groups = tuple(df[group_col].astype(str))
        edges_with_attr = ((u, v, {"group": grp}) for (u, v), grp in zip(adj_pairs, groups, strict=True))
        graph: nx.Graph[str] = nx.Graph(edges_with_attr)
        cmap = plt.get_cmap("tab10")  # pyright: ignore[reportUnknownMemberType]
        unique_groups = tuple(df[group_col].astype(str).unique())
        color_map: dict[str, Any] = {grp: cmap(i) for i, grp in enumerate(unique_groups)}
        edge_colors: Any = [color_map[str(graph.edges[e]["group"])] for e in graph.edges]
    else:
        graph = nx.Graph(adj_pairs)
        edge_colors = "black"

    plt.figure(figsize=(16, 16))  # pyright: ignore[reportUnknownMemberType]
    nx.draw(  # pyright: ignore[reportUnknownMemberType]
        graph,
        with_labels=True,
        node_color="white",
        node_size=2000,
        font_family=_get_default_font_family(),
        width=3,
        edge_color=edge_colors,
        linewidths=1,
        edgecolors="gray",
        font_size=10,
    )
