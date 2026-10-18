"""Phase 2 パイプライン：GIS 手動補正データの検証、駅接続グラフ解析、駅順ソート、通称付与、および最終出力。"""

import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry.base import BaseGeometry

from src.domain.aliases import (
    CO_ALIASES,
    LINE_ALIASES,
    SERVICE_LINE_ALIASES,
    SERVICES,
)
from src.domain.config import DEFAULT_CONFIG, PipelineConfig
from src.domain.types import LineGraphSummary, LineKey, LineStatus
from src.pipelines.visualize import save_line_graph_image
from src.topology.graph_sort import analyze_line_graph, sorter_factory


@dataclass(frozen=True, slots=True)
class Phase2Result:
    """Phase 2 パイプラインの中間・最終生成データを保持する不変コンテナ。"""

    railroads_with_adj_stas: gpd.GeoDataFrame
    rr_adjs_man: gpd.GeoDataFrame
    invalid_adj_stas_df: gpd.GeoDataFrame
    duplicate_ids: np.ndarray[Any, Any]
    mismatched_co_line_df: gpd.GeoDataFrame
    status_counts: pd.Series[int]
    sta_pt: gpd.GeoDataFrame
    sta_pt_man: gpd.GeoDataFrame
    added_stations_df: gpd.GeoDataFrame
    removed_stations_df: gpd.GeoDataFrame
    line_graph_summaries: tuple[LineGraphSummary, ...]
    sta_graph_df: pd.DataFrame
    rr_sorted: gpd.GeoDataFrame


def clean_val(x: object) -> object:
    """`None` を空文字列に変換し、文字列であれば前後の空白を除去する。"""
    return "" if x is None else (x.strip() if isinstance(x, str) else x)


def parse_adj_stas_tuple(raw: object) -> tuple[str, ...]:
    """カンマ区切りの隣接駅文字列をソート済みの駅名タプルへ変換する。"""
    text = str(raw).strip()
    if not text:
        return ()
    return tuple(sorted(text.split(",")))


def _is_valid_adj_len(adjs: tuple[str, ...]) -> bool:
    """隣接駅タプルの要素数が 1 または 2 であるかを判定する。"""
    return len(adjs) in (1, 2)


def load_and_normalize_adj_stas_gdf(path: Path, encoding: str = "utf-8") -> gpd.GeoDataFrame:
    """隣接駅付き路線シェープファイルを読み込み、属性の正規化と `adj_stas` のタプル化を行う。"""
    raw_gdf = gpd.read_file(path, encoding=encoding)  # pyright: ignore[reportUnknownMemberType]
    cleaned_df = raw_gdf.map(clean_val)
    gdf = gpd.GeoDataFrame(cleaned_df, geometry="geometry", crs=raw_gdf.crs)
    adj_tuples = gdf["adj_stas"].map(parse_adj_stas_tuple)
    return gdf.assign(adj_stas=adj_tuples)


def find_invalid_adj_stas_rows(
    rr_adjs_man: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """`adj_stas` の要素数が 1 でも 2 でもない異常行を抽出する。"""
    valid_mask = rr_adjs_man["adj_stas"].map(_is_valid_adj_len)
    return rr_adjs_man[~valid_mask]


def find_duplicate_ids(rr_adjs_man: gpd.GeoDataFrame) -> np.ndarray[Any, Any]:
    """手動編集データ内で重複している `id` の配列を返す。"""
    unique_id, dup_count = np.unique(rr_adjs_man["id"].to_numpy(), return_counts=True)
    return unique_id[dup_count != 1]


def find_mismatched_co_line_rows(
    rr_adjs_man: gpd.GeoDataFrame,
    railroads_with_adj_stas: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """手動編集前後で同じ `id` なのに `co` または `line` が変化している行を抽出する。"""
    org_by_id: dict[int, LineKey] = {
        int(row_id): (str(co), str(line))
        for row_id, co, line in zip(
            railroads_with_adj_stas["id"],
            railroads_with_adj_stas["co"],
            railroads_with_adj_stas["line"],
            strict=True,
        )
    }

    def _is_mismatched(row: pd.Series[Any]) -> bool:
        row_id = int(row["id"])
        if row_id not in org_by_id:
            return False
        org_co, org_line = org_by_id[row_id]
        return (org_co != str(row["co"])) or (org_line != str(row["line"]))

    mask = cast(
        pd.Series[bool],
        rr_adjs_man.apply(_is_mismatched, axis=1),  # pyright: ignore[reportUnknownMemberType]
    )
    return rr_adjs_man[mask]


def assign_modification_status(
    rr_adjs_man: gpd.GeoDataFrame,
    railroads_with_adj_stas: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """編集前データとの比較により、各行へ変更ステータス（`remain` / `alter` / `copy` / `new`）を付与する。"""
    org_geom_gs = cast(gpd.GeoSeries, railroads_with_adj_stas["geometry"])
    org_adj_by_id: dict[int, tuple[str, ...]] = {
        int(row_id): cast(tuple[str, ...], adjs)
        for row_id, adjs in zip(
            railroads_with_adj_stas["id"],
            railroads_with_adj_stas["adj_stas"],
            strict=True,
        )
    }

    def _determine_status(row: pd.Series[Any]) -> LineStatus:
        row_id = int(row["id"])
        if row_id == 0:
            geom = cast(BaseGeometry, row["geometry"])
            is_copied = bool(org_geom_gs.geom_equals(geom).any())  # pyright: ignore[reportUnknownMemberType]
            return "copy" if is_copied else "new"
        if row_id in org_adj_by_id:
            return "remain" if row["adj_stas"] == org_adj_by_id[row_id] else "alter"
        return ""

    statuses = cast(
        pd.Series[str],
        rr_adjs_man.apply(_determine_status, axis=1),  # pyright: ignore[reportUnknownMemberType]
    )
    return rr_adjs_man.assign(status=statuses)


def compare_station_points(
    sta_pt: gpd.GeoDataFrame,
    sta_pt_man: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """編集前後の駅代表点データを比較し、（手動で追加・変更された駅点, 手動で削除・変更された元の駅点）を返す。"""

    def _find_unmatched_rows(source_df: gpd.GeoDataFrame, target_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        target_geom_gs = cast(gpd.GeoSeries, target_df["geometry"])

        def _exists_in_target(row: pd.Series[Any]) -> bool:
            geom = cast(BaseGeometry, row["geometry"])
            geom_matches = target_geom_gs.geom_equals(geom)  # pyright: ignore[reportUnknownMemberType]
            if not bool(geom_matches.any()):
                return False
            matched_rows = target_df[geom_matches]
            return bool((matched_rows == row).all(axis=1).any())  # pyright: ignore[reportUnknownMemberType]

        matched_mask = cast(
            pd.Series[bool],
            source_df.apply(_exists_in_target, axis=1),  # pyright: ignore[reportUnknownMemberType]
        )
        return source_df[~matched_mask]

    added_in_man = _find_unmatched_rows(sta_pt_man, sta_pt)
    removed_from_org = _find_unmatched_rows(sta_pt, sta_pt_man)
    return added_in_man, removed_from_org


def build_station_graph_table(
    rr_adjs_man: gpd.GeoDataFrame,
    image_dir_path: Path | None = None,
) -> tuple[pd.DataFrame, tuple[LineGraphSummary, ...]]:
    """路線ごとに駅接続グラフを構築して直径パス分解を行い、結果の `DataFrame` とサマリ列を返す。"""
    grouped_size = rr_adjs_man.groupby(["co", "line"], as_index=False).size()  # pyright: ignore[reportUnknownMemberType]
    line_keys: tuple[LineKey, ...] = tuple(
        (str(co), str(line)) for co, line in zip(grouped_size["co"], grouped_size["line"], strict=True)
    )

    def _process_single_line(co: str, line: str) -> LineGraphSummary:
        mask = (rr_adjs_man["co"] == co) & (rr_adjs_man["line"] == line)
        adjs_series = rr_adjs_man[mask]["adj_stas"]
        summary, graph = analyze_line_graph(co, line, adjs_series)
        if image_dir_path is not None and graph is not None and (summary.cycled or not summary.connected):
            save_line_graph_image(graph, image_dir_path / f"{co}_{line}.png")
        return summary

    summaries = tuple(_process_single_line(co, line) for co, line in line_keys)

    flat_rows = [summary.to_flat_row() for summary in summaries]
    raw_df = pd.DataFrame(flat_rows)
    renamed_cols: dict[int, str | int] = {
        0: "co",
        1: "line",
        2: "cycled",
        3: "connected",
    } | {col_idx: col_idx - 2 for col_idx in range(4, raw_df.shape[1])}
    sta_graph_df = raw_df.rename(columns=renamed_cols)
    return sta_graph_df, summaries


def sort_railroads_by_station_order(
    rr_adjs_man: gpd.GeoDataFrame,
    summaries: Sequence[LineGraphSummary],
) -> gpd.GeoDataFrame:
    """抽出された各路線のパス順序に基づいて `LineString` を駅順にソートし、`line_id` と `seq_id` を採番する。"""

    def _sort_single_line(summary: LineGraphSummary) -> gpd.GeoDataFrame:
        mask = (rr_adjs_man["co"] == summary.co) & (rr_adjs_man["line"] == summary.line)
        line_df = rr_adjs_man[mask]
        adjs_sorter = sorter_factory(summary.paths)
        return line_df.sort_values(by="adj_stas", key=adjs_sorter)

    sorted_frames: list[gpd.GeoDataFrame] = [_sort_single_line(summary) for summary in summaries]
    combined = pd.concat(sorted_frames)
    grouped = combined.groupby(["co", "line"])  # pyright: ignore[reportUnknownMemberType]
    with_ids = combined.assign(
        line_id=grouped.ngroup() + 1,
        seq_id=grouped.cumcount() + 1,
    )
    return with_ids.set_index(["line_id", "seq_id"]).drop(columns=["id"])


def assign_aliases_to_sorted_railroads(
    rr_sorted: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """事業者通称（`co_alias`）、サービス名（`service`）、路線通称（`line_alias`）を付与する。"""
    co_alias_s = rr_sorted["co"].replace(dict(CO_ALIASES))
    default_service_s = co_alias_s.replace(dict(SERVICES))
    default_line_alias_s = rr_sorted["line"].replace(dict(LINE_ALIASES))

    def _resolve_service_and_line(co: str, line: str, default_srv: str, default_l_alias: str) -> tuple[str, str]:
        if (co, line) in SERVICE_LINE_ALIASES:
            return SERVICE_LINE_ALIASES[(co, line)]
        return default_srv, default_l_alias

    resolved_pairs = [
        _resolve_service_and_line(str(co), str(line), str(srv), str(l_alias))
        for co, line, srv, l_alias in zip(
            rr_sorted["co"],
            rr_sorted["line"],
            default_service_s,
            default_line_alias_s,
            strict=True,
        )
    ]
    services_col = [srv for srv, _ in resolved_pairs]
    line_aliases_col = [l_alias for _, l_alias in resolved_pairs]

    return rr_sorted.assign(
        co_alias=co_alias_s,
        service=services_col,
        line_alias=line_aliases_col,
    )


def find_duplicate_line_names(sta_graph_df: pd.DataFrame) -> pd.Series[int]:
    """複数の事業者にまたがって存在する（一意でない）路線名とその出現回数を抽出する。"""
    line_counts = sta_graph_df["line"].value_counts()
    return line_counts[line_counts > 1]


def _join_adj_stas(adjs: tuple[str, ...]) -> str:
    """駅名タプルをカンマ区切り文字列へ変換する。"""
    return ",".join(adjs)


def export_final_outputs(
    rr_sorted: gpd.GeoDataFrame,
    sta_pt_man: gpd.GeoDataFrame,
    config: PipelineConfig,
) -> None:
    """ソート・通称付与済みの路線データと駅データを Shapefile (ZIP)、TSV、GeoJSON に出力する。"""
    config.rr_sorted_dir_path.mkdir(parents=True, exist_ok=True)

    rr_sorted_out = rr_sorted.assign(adj_stas=rr_sorted["adj_stas"].map(_join_adj_stas))
    rr_sorted_out.to_file(  # pyright: ignore[reportUnknownMemberType]
        config.rr_sorted_shp_path, encoding=config.encoding
    )
    shutil.make_archive(
        config.rr_sorted_zip_stem,
        format="zip",
        root_dir=".",
        base_dir=config.rr_sorted_dir_path,
    )

    pd.DataFrame(rr_sorted_out.drop(columns="geometry")).to_csv(config.rr_sorted_tsv_path, sep="\t")

    rr_sorted.to_crs(config.target_crs).to_file(  # pyright: ignore[reportUnknownMemberType]
        config.rr_geojson_path, driver="GeoJSON"
    )
    sta_pt_man.to_crs(config.target_crs).to_file(  # pyright: ignore[reportUnknownMemberType]
        config.sta_geojson_path, driver="GeoJSON"
    )


def run_phase2(
    config: PipelineConfig = DEFAULT_CONFIG,
    *,
    save_outputs: bool = True,
    save_graph_images: bool = False,
) -> Phase2Result:
    """Phase 2 パイプライン全体を実行し、最終的な Shapefile・TSV・GeoJSON を出力する。"""
    railroads_with_adj_stas = load_and_normalize_adj_stas_gdf(config.railroad_adj_stas_path, encoding=config.encoding)
    rr_adjs_man = load_and_normalize_adj_stas_gdf(config.railroad_adj_stas_manual_path, encoding=config.encoding)

    invalid_adj_stas_df = find_invalid_adj_stas_rows(rr_adjs_man)
    duplicate_ids = find_duplicate_ids(rr_adjs_man)
    mismatched_co_line_df = find_mismatched_co_line_rows(rr_adjs_man, railroads_with_adj_stas)

    rr_adjs_man_with_status = assign_modification_status(rr_adjs_man, railroads_with_adj_stas)
    status_counts = rr_adjs_man_with_status.groupby("status").size()  # pyright: ignore[reportUnknownMemberType]

    sta_pt = gpd.read_file(config.station_pt_path, encoding=config.encoding)  # pyright: ignore[reportUnknownMemberType]
    sta_pt_man = gpd.read_file(  # pyright: ignore[reportUnknownMemberType]
        config.station_pt_manual_path, encoding=config.encoding
    )
    added_stations_df, removed_stations_df = compare_station_points(sta_pt, sta_pt_man)

    image_dir = config.image_dir_path if save_graph_images else None
    sta_graph_df, summaries = build_station_graph_table(rr_adjs_man_with_status, image_dir_path=image_dir)
    if save_outputs:
        sta_graph_df.to_csv(config.sta_graph_csv_path, sep="\t")

    rr_sorted_base = sort_railroads_by_station_order(rr_adjs_man_with_status, summaries)
    rr_sorted = assign_aliases_to_sorted_railroads(rr_sorted_base)

    if save_outputs:
        export_final_outputs(rr_sorted, sta_pt_man, config)

    return Phase2Result(
        railroads_with_adj_stas=railroads_with_adj_stas,
        rr_adjs_man=rr_adjs_man_with_status,
        invalid_adj_stas_df=invalid_adj_stas_df,
        duplicate_ids=duplicate_ids,
        mismatched_co_line_df=mismatched_co_line_df,
        status_counts=status_counts,
        sta_pt=sta_pt,
        sta_pt_man=sta_pt_man,
        added_stations_df=added_stations_df,
        removed_stations_df=removed_stations_df,
        line_graph_summaries=summaries,
        sta_graph_df=sta_graph_df,
        rr_sorted=rr_sorted,
    )
