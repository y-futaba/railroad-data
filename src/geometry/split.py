"""駅の代表点抽出、円形バッファ生成、端点スナップ、および駅点での路線分割処理。"""

from functools import partial
from typing import Any, cast

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, Polygon

from src.domain.types import Coord
from src.geometry.geodesic import extract_coords, get_center


def representative_station_point(row: pd.Series[Any]) -> Point:
    """駅データの1行から代表点 `Point` を算出する。

    九州旅客鉄道・鹿児島線・広木駅のみ最東端の頂点を採用し、
    それ以外の駅はすべて測地線中点を返す。
    """
    geom = cast(LineString, row["geometry"])
    is_hiroki = row["N02_005"] == "広木" and row["N02_003"] == "鹿児島線" and row["N02_004"] == "九州旅客鉄道"
    if is_hiroki:
        easternmost_coord = max(extract_coords(geom), key=lambda coord: coord[0])
        return Point(easternmost_coord)
    return get_center(geom)


def cir_buffer(pt: Point, r: float, num_points: int = 16) -> Polygon:
    """点 `pt` の周囲に半径 `r` `[deg]` の正多角形バッファ `Polygon` を生成する。"""
    rads = 2.0 * np.pi * np.arange(num_points, dtype=np.float64) / num_points
    xs = pt.x + r * np.cos(rads)
    ys = pt.y + r * np.sin(rads)
    return Polygon(tuple(zip(xs.tolist(), ys.tolist(), strict=True)))


def pt_norm_sq(p1: Point, p2: Point) -> float:
    """2つの `Point` 間の度数空間における二乗ユークリッド距離を算出する。"""
    return float(np.power(p1.x - p2.x, 2) + np.power(p1.y - p2.y, 2))


def _snap_endpoint_coord(coord: Coord, pt_gs: gpd.GeoSeries, tolerance_sq: float) -> Coord:
    """単一の端点座標について、`tolerance_sq` 以内に存在する最近傍の駅点座標へスナップする。"""
    if pt_gs.empty:
        return coord

    edge_pt = Point(coord)
    distance_gs = pt_gs.map(partial(pt_norm_sq, p2=edge_pt))
    if float(distance_gs.min()) < tolerance_sq:
        nearest_pt = cast(Point, pt_gs.iloc[int(distance_gs.argmin())])
        return (float(nearest_pt.x), float(nearest_pt.y))
    return coord


def snap_edge(ls: LineString, pt_gs: gpd.GeoSeries, tolerance: float) -> LineString:
    """`LineString` の始点および終点を、`tolerance` 以内にある `pt_gs` の最近傍点へスナップする。"""
    coords = extract_coords(ls)
    tolerance_sq = float(np.power(tolerance, 2))
    snapped_coords = (
        _snap_endpoint_coord(coords[0], pt_gs, tolerance_sq),
        *coords[1:-1],
        _snap_endpoint_coord(coords[-1], pt_gs, tolerance_sq),
    )
    return LineString(snapped_coords)


def split_line_at_point(
    ls_gdf: gpd.GeoDataFrame,
    pt_gdf: gpd.GeoDataFrame,
    tolerance: float,
    buffer_ratio: float = 0.9,
) -> gpd.GeoDataFrame:
    """`pt_gdf` の円形バッファで `ls_gdf` の路線をくり抜き、切断端点を駅点へスナップする。"""
    pt_geom_gs = cast(gpd.GeoSeries, pt_gdf["geometry"])
    buffer_radius = buffer_ratio * tolerance
    buffer_gdf = gpd.GeoDataFrame(
        geometry=pt_geom_gs.map(partial(cir_buffer, r=buffer_radius)),
        crs=pt_gdf.crs,
    )

    cut_gdf = gpd.overlay(
        df1=ls_gdf,
        df2=buffer_gdf,
        how="symmetric_difference",
        keep_geom_type=True,
    ).explode(index_parts=False)

    cut_geom_gs = cast(gpd.GeoSeries, cut_gdf["geometry"])
    snapped_geom = cut_geom_gs.map(partial(snap_edge, pt_gs=pt_geom_gs, tolerance=tolerance))
    return cut_gdf.assign(geometry=snapped_geom)
