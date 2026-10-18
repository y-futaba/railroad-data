"""Phase 1 パイプライン：原データの読み込み、路線結合、駅代表点による切断、欠落駅補完、および隣接駅検査。"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import ops
from shapely.geometry import MultiLineString, Point
from shapely.geometry.base import BaseGeometry

from src.domain.config import DEFAULT_CONFIG, PipelineConfig
from src.domain.types import Coord, ExcludedStation, LineKey
from src.geometry.split import (
    representative_station_point,
    split_line_at_point,
)
from src.topology.adjacency import (
    adjacent_stations,
    check_adjacent_stations_with_ls,
    edge_stations,
    excluded_sta_coords,
    extract_point_coord,
)


@dataclass(frozen=True, slots=True)
class Phase1Result:
    """Phase 1 パイプラインの中間・最終生成データを保持する不変コンテナ。"""

    railroads: gpd.GeoDataFrame
    merged_railroads: gpd.GeoDataFrame
    stations_ls: gpd.GeoDataFrame
    stations: gpd.GeoDataFrame
    line_keys: tuple[LineKey, ...]
    initial_split_railroads: gpd.GeoDataFrame
    initial_excluded_stations: tuple[ExcludedStation, ...]
    complemented_railroads: gpd.GeoDataFrame
    re_merged_railroads: gpd.GeoDataFrame
    re_split_railroads: gpd.GeoDataFrame
    re_excluded_stations: tuple[ExcludedStation, ...]
    sta_degree_count_df: pd.DataFrame
    adj_stas_ele_count_df: pd.DataFrame
    irregular_lines: pd.DataFrame
    irregular_sta_set_df: pd.DataFrame
    railroads_with_adj_stas: gpd.GeoDataFrame


def _merge_line_geometry(geom: BaseGeometry) -> BaseGeometry:
    """`MultiLineString` のジオメトリに `ops.linemerge` を適用する。"""
    if isinstance(geom, MultiLineString):
        return ops.linemerge(geom)
    return geom


def load_raw_railroads(config: PipelineConfig) -> gpd.GeoDataFrame:
    """国土数値情報の路線シェープファイルを読み込み、必要列のみを抽出する。"""
    raw_gdf = gpd.read_file(config.railroads_path, encoding=config.encoding)  # pyright: ignore[reportUnknownMemberType]
    return raw_gdf[["N02_003", "N02_004", "geometry"]].copy()


def merge_railroad_lines(railroads_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """事業者（`N02_004`）・路線名（`N02_003`）単位で路線を `dissolve` および `linemerge` で結合する。"""
    dissolved = railroads_gdf.dissolve(by=["N02_004", "N02_003"])  # pyright: ignore[reportUnknownMemberType]
    dissolved_geom = cast(gpd.GeoSeries, dissolved["geometry"])
    merged = dissolved.assign(geometry=dissolved_geom.map(_merge_line_geometry))
    return merged.explode(index_parts=False)


def load_and_compute_station_points(
    config: PipelineConfig,
    *,
    save_file: bool = True,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """駅シェープファイルを読み込み、各駅の代表点 `Point` を算出する。"""
    stations_ls = gpd.read_file(config.stations_path, encoding=config.encoding)  # pyright: ignore[reportUnknownMemberType]
    rep_points = cast(
        pd.Series[Any],
        stations_ls.apply(representative_station_point, axis=1),  # pyright: ignore[reportUnknownMemberType]
    )
    stations = stations_ls.assign(geometry=rep_points)
    if save_file:
        stations.to_file(config.station_pt_path, encoding=config.encoding)  # pyright: ignore[reportUnknownMemberType]
    return stations_ls, stations


def extract_line_keys(merged_railroads: gpd.GeoDataFrame) -> tuple[LineKey, ...]:
    """結合済み路線 `GeoDataFrame` のインデックスから一意な `(co, line)` キー列を抽出する。"""
    raw_index = cast(Sequence[tuple[Any, Any]], merged_railroads.index)
    unique_keys = {(str(co), str(line_name)) for co, line_name in raw_index}
    return tuple(sorted(unique_keys))


def _slice_line_gdf(gdf: gpd.GeoDataFrame, co: str, line_name: str) -> gpd.GeoDataFrame:
    """MultiIndex `(co, line)` から指定路線の `GeoDataFrame` を抽出する。"""
    return cast(gpd.GeoDataFrame, gdf.loc[co, line_name])


def _filter_stations_by_line(stations: gpd.GeoDataFrame, co: str, line_name: str) -> gpd.GeoDataFrame:
    """事業者名と路線名に一致する駅データを抽出する。"""
    mask = (stations["N02_004"] == co) & (stations["N02_003"] == line_name)
    return stations[mask].copy()


def split_all_lines_by_stations(
    merged_railroads: gpd.GeoDataFrame,
    stations: gpd.GeoDataFrame,
    line_keys: Sequence[LineKey],
    tolerance: float = 1e-5,
    buffer_ratio: float = 0.9,
) -> gpd.GeoDataFrame:
    """全路線をそれぞれの駅代表点で切断し、`(co, line)` インデックス付きの `GeoDataFrame` として統合する。"""
    splits: list[gpd.GeoDataFrame] = [
        split_line_at_point(
            _slice_line_gdf(merged_railroads, co, line_name),
            _filter_stations_by_line(stations, co, line_name),
            tolerance=tolerance,
            buffer_ratio=buffer_ratio,
        ).assign(co=co, line=line_name)
        for co, line_name in line_keys
    ]
    return pd.concat(splits).set_index(["co", "line"]).sort_index()


def find_excluded_stations(
    split_railroads: gpd.GeoDataFrame,
    stations: gpd.GeoDataFrame,
    line_keys: Sequence[LineKey],
) -> tuple[ExcludedStation, ...]:
    """切断後の路線データから頂点が欠落している駅を全路線にわたって検出する。"""

    def _excluded_for_line(co: str, line_name: str) -> tuple[ExcludedStation, ...]:
        line_geom_gs = cast(
            gpd.GeoSeries,
            _slice_line_gdf(split_railroads, co, line_name)["geometry"],
        )
        sta_df = _filter_stations_by_line(stations, co, line_name)
        sta_geom_gs = cast(gpd.GeoSeries, sta_df["geometry"])
        missing_coords = excluded_sta_coords(line_geom_gs, sta_geom_gs)
        if not missing_coords:
            return ()

        return tuple(
            ExcludedStation(
                co=co,
                line=line_name,
                station_name=str(sta_df[sta_geom_gs == Point(coord)]["N02_005"].iloc[0]),
            )
            for coord in missing_coords
        )

    return tuple(excluded for co, line_name in line_keys for excluded in _excluded_for_line(co, line_name))


def complement_missing_station_segments(
    railroads: gpd.GeoDataFrame,
    stations_ls: gpd.GeoDataFrame,
    excluded_stations: Sequence[ExcludedStation],
) -> gpd.GeoDataFrame:
    """路線データから抜けていた駅の線分ジオメトリを補完した新しい路線 `GeoDataFrame` を返す。"""
    if not excluded_stations:
        return railroads.copy()

    dissolved_stations_ls = stations_ls.dissolve(  # pyright: ignore[reportUnknownMemberType]
        by=["N02_004", "N02_003", "N02_005"]
    )

    def _build_station_segment_chunk(item: ExcludedStation) -> gpd.GeoDataFrame:
        mask = (railroads["N02_004"] == item.co) & (railroads["N02_003"] == item.line)
        template_df = railroads[mask].iloc[:1].copy()
        sta_geom = cast(
            BaseGeometry,
            dissolved_stations_ls.at[(item.co, item.line, item.station_name), "geometry"],
        )
        sta_geom_gs = gpd.GeoSeries([sta_geom], index=template_df.index, crs=railroads.crs)
        return template_df.assign(geometry=sta_geom_gs).explode(index_parts=False)

    complement_chunks: list[gpd.GeoDataFrame] = [_build_station_segment_chunk(item) for item in excluded_stations]
    return pd.concat([railroads.copy(), *complement_chunks])


def compute_station_degree_and_set_counts(
    split_railroads: gpd.GeoDataFrame,
    stations: gpd.GeoDataFrame,
    line_keys: Sequence[LineKey],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """路線ごとに駅の接続線数（次数）分布と、隣接駅集合の要素数分布を集計する。"""

    def _analyze_single_line_counts(co: str, line_name: str) -> tuple[pd.Series[Any], pd.Series[Any]]:
        line_geom_gs = cast(
            gpd.GeoSeries,
            _slice_line_gdf(split_railroads, co, line_name)["geometry"],
        )
        sta_geom_gs = cast(
            gpd.GeoSeries,
            _filter_stations_by_line(stations, co, line_name)["geometry"],
        )
        _, edge_stas_to_coords = edge_stations(line_geom_gs, sta_geom_gs)
        adj_stas = adjacent_stations(edge_stas_to_coords)
        row_label = f"{co} {line_name}"

        def _count_station_degree(pt: Point) -> int:
            coord = extract_point_coord(pt)
            return sum(1 for adj_sta in adj_stas if coord in adj_sta)

        degree_counts = sta_geom_gs.map(_count_station_degree).value_counts().sort_index().rename(row_label)
        ele_counts = pd.Series(tuple(edge_stas_to_coords.keys())).map(len).value_counts().sort_index().rename(row_label)
        return degree_counts, ele_counts

    pairs = [_analyze_single_line_counts(co, line_name) for co, line_name in line_keys]
    degree_series_list = [deg for deg, _ in pairs]
    ele_series_list = [ele for _, ele in pairs]

    sta_degree_count_df = pd.DataFrame(degree_series_list).fillna(0).astype(int).reindex(columns=[0, 1, 2, 3], fill_value=0)
    adj_stas_ele_count_df = (
        pd.DataFrame(ele_series_list).fillna(0).astype(int).reindex(columns=[0, 1, 2, 3, 4, 5, 6], fill_value=0)
    )
    return sta_degree_count_df, adj_stas_ele_count_df


def extract_irregular_lines(adj_stas_ele_count_df: pd.DataFrame) -> pd.DataFrame:
    """分岐・環状・データ不備などにより単純な一本線になっていない路線行を抽出する。"""
    non_pair_sum = (
        adj_stas_ele_count_df[0]
        + adj_stas_ele_count_df[3]
        + adj_stas_ele_count_df[4]
        + adj_stas_ele_count_df[5]
        + adj_stas_ele_count_df[6]
    )
    mask = (non_pair_sum > 0) | (adj_stas_ele_count_df[1] != 2)
    return adj_stas_ele_count_df[mask]


def build_irregular_station_sets_df(
    irregular_lines: pd.DataFrame,
    split_railroads: gpd.GeoDataFrame,
    stations: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """接続が異常な路線について、要素数が2でない隣接駅集合を抽出し `DataFrame` にまとめる。"""

    def _build_line_irregular_df(index_label: str) -> pd.DataFrame:
        co, line_name = index_label.split(maxsplit=1)
        sta_df = _filter_stations_by_line(stations, co, line_name)[["N02_005", "geometry"]]
        sta_geom_gs = cast(gpd.GeoSeries, sta_df["geometry"])
        line_geom_gs = cast(
            gpd.GeoSeries,
            _slice_line_gdf(split_railroads, co, line_name)["geometry"],
        )
        _, edge_stas_to_coords = edge_stations(line_geom_gs, sta_geom_gs)

        sta_coords = tuple(sta_geom_gs.map(extract_point_coord))
        sta_names = tuple(sta_df["N02_005"].astype(str))
        coord_to_name: dict[Coord, str] = {
            coord: name for coord, name in reversed(tuple(zip(sta_coords, sta_names, strict=True)))
        }

        irregular_name_sets = sorted(
            {
                frozenset(coord_to_name[edge_sta] for edge_sta in edge_stas)
                for edge_stas in edge_stas_to_coords
                if len(edge_stas) != 2
            },
            key=len,
        )

        rows: list[dict[str | int, str]] = [
            {"co": co, "line": line_name} | {i: sta_name for i, sta_name in enumerate(edge_stas)}
            for edge_stas in irregular_name_sets
        ]
        return pd.DataFrame(rows)

    index_labels = [str(idx) for idx in irregular_lines.index]
    frames: list[pd.DataFrame] = [_build_line_irregular_df(label) for label in index_labels]
    if not frames:
        return pd.DataFrame(columns=["co", "line"])

    return pd.concat(frames).fillna("").sort_values(["co", "line"]).reset_index(drop=True)


def find_isolated_line_coords(
    split_railroads: gpd.GeoDataFrame,
    stations: gpd.GeoDataFrame,
    line_keys: Sequence[LineKey],
) -> dict[LineKey, frozenset[Coord]]:
    """接続している隣接駅が0個（孤立している）の頂点座標を持つ路線とその座標集合を返す。"""
    empty_key: frozenset[Coord] = frozenset()

    def _isolated_coords_for_line(co: str, line_name: str) -> frozenset[Coord] | None:
        line_geom_gs = cast(
            gpd.GeoSeries,
            _slice_line_gdf(split_railroads, co, line_name)["geometry"],
        )
        sta_geom_gs = cast(
            gpd.GeoSeries,
            _filter_stations_by_line(stations, co, line_name)["geometry"],
        )
        _, edge_stas_to_coords = edge_stations(line_geom_gs, sta_geom_gs)
        return edge_stas_to_coords.get(empty_key)

    return {
        (co, line_name): isolated
        for co, line_name in line_keys
        if (isolated := _isolated_coords_for_line(co, line_name)) is not None
    }


def assign_adjacent_stations_to_lines(
    split_railroads: gpd.GeoDataFrame,
    stations: gpd.GeoDataFrame,
    line_keys: Sequence[LineKey],
) -> gpd.GeoDataFrame:
    """全路線の各 `LineString` に隣接駅属性 `adj_stas` と連番 `id` を付与する。"""
    checked_frames: list[gpd.GeoDataFrame] = [
        check_adjacent_stations_with_ls(
            _slice_line_gdf(split_railroads, co, line_name).copy(),
            _filter_stations_by_line(stations, co, line_name),
        )
        for co, line_name in line_keys
    ]
    combined = pd.concat(checked_frames).sort_values(["co", "line"])
    out_gdf = combined.copy()
    out_gdf.insert(1, "id", np.arange(len(out_gdf)) + 1)  # pyright: ignore[reportUnknownMemberType]
    return out_gdf


def count_lines_per_route(
    railroads_with_adj_stas: gpd.GeoDataFrame,
    line_keys: Sequence[LineKey],
) -> pd.DataFrame:
    """各路線 `(co, line)` に含まれる `LineString` 行数を集計する。"""
    rows = [
        {
            "co": co,
            "line": line_name,
            "count": len(_slice_line_gdf(railroads_with_adj_stas, co, line_name)),
        }
        for co, line_name in line_keys
    ]
    return pd.DataFrame(rows).set_index(["co", "line"])


def run_phase1(
    config: PipelineConfig = DEFAULT_CONFIG,
    *,
    save_outputs: bool = True,
) -> Phase1Result:
    """Phase 1 パイプライン全体を実行し、GIS 手動編集用のシェープファイルおよび検査 CSV を出力する。"""
    railroads = load_raw_railroads(config)
    merged_railroads = merge_railroad_lines(railroads)
    stations_ls, stations = load_and_compute_station_points(config, save_file=save_outputs)
    line_keys = extract_line_keys(merged_railroads)

    initial_split = split_all_lines_by_stations(
        merged_railroads,
        stations,
        line_keys,
        tolerance=config.snap_tolerance,
        buffer_ratio=config.buffer_ratio,
    )
    initial_excluded = find_excluded_stations(initial_split, stations, line_keys)

    complemented_railroads = complement_missing_station_segments(railroads, stations_ls, initial_excluded)
    re_merged_railroads = merge_railroad_lines(complemented_railroads)
    re_split = split_all_lines_by_stations(
        re_merged_railroads,
        stations,
        line_keys,
        tolerance=config.snap_tolerance,
        buffer_ratio=config.buffer_ratio,
    )
    re_excluded = find_excluded_stations(re_split, stations, line_keys)

    sta_degree_count_df, adj_stas_ele_count_df = compute_station_degree_and_set_counts(re_split, stations, line_keys)
    irregular_lines = extract_irregular_lines(adj_stas_ele_count_df)
    irregular_sta_set_df = build_irregular_station_sets_df(irregular_lines, re_split, stations)
    railroads_with_adj_stas = assign_adjacent_stations_to_lines(re_split, stations, line_keys)

    if save_outputs:
        irregular_sta_set_df.to_csv(config.irregular_sta_set_csv_path)
        railroads_with_adj_stas.to_file(  # pyright: ignore[reportUnknownMemberType]
            config.railroad_adj_stas_path, encoding=config.encoding
        )

    return Phase1Result(
        railroads=railroads,
        merged_railroads=merged_railroads,
        stations_ls=stations_ls,
        stations=stations,
        line_keys=line_keys,
        initial_split_railroads=initial_split,
        initial_excluded_stations=initial_excluded,
        complemented_railroads=complemented_railroads,
        re_merged_railroads=re_merged_railroads,
        re_split_railroads=re_split,
        re_excluded_stations=re_excluded,
        sta_degree_count_df=sta_degree_count_df,
        adj_stas_ele_count_df=adj_stas_ele_count_df,
        irregular_lines=irregular_lines,
        irregular_sta_set_df=irregular_sta_set_df,
        railroads_with_adj_stas=railroads_with_adj_stas,
    )
