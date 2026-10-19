"""WGS84 測地線距離計算、`LineString` からのセグメント抽出、および距離補間処理。"""

from collections.abc import Sequence
from itertools import chain, pairwise
from typing import Any, cast

import numpy as np
import pyproj
from shapely.geometry import LineString, MultiLineString, Point

from src.domain.types import Coord, Segment, UndirectedSegment

_WGS84_GEOD = pyproj.Geod(ellps="WGS84")


def extract_coords(ls: LineString) -> tuple[Coord, ...]:
    """`LineString` から2次元座標列 `(lon, lat)` を不変タプルとして抽出する。"""
    raw_coords = cast(Sequence[Sequence[float]], ls.coords)
    return tuple((float(pt[0]), float(pt[1])) for pt in raw_coords)


def ordered_segments(ls: LineString) -> tuple[Segment, ...]:
    """`LineString` の隣り合う頂点ペアを有向セグメント `(start, target)` のタプルとして抽出する。"""
    return tuple(pairwise(extract_coords(ls)))


def undirected_segments(ls: LineString) -> tuple[UndirectedSegment, ...]:
    """`LineString` の隣り合う頂点ペアを無向セグメント `frozenset` のタプルとして抽出する。"""
    return tuple(map(frozenset, pairwise(extract_coords(ls))))


def get_all_ordered_segments(mls: MultiLineString) -> tuple[Segment, ...]:
    """`MultiLineString` 内の全 `LineString` から重複を除いた有向セグメント列を出現順に取得する。"""
    geoms = cast(Sequence[LineString], mls.geoms)
    return tuple(dict.fromkeys(chain.from_iterable(map(ordered_segments, geoms))))


def get_all_undirected_segments(mls: MultiLineString) -> tuple[UndirectedSegment, ...]:
    """`MultiLineString` 内の全 `LineString` から重複を除いた無向セグメント列を取得する。"""
    geoms = cast(Sequence[LineString], mls.geoms)
    return tuple(frozenset(chain.from_iterable(map(undirected_segments, geoms))))


def segments_from_ls(ls: LineString, order: bool = False) -> list[Segment] | list[UndirectedSegment]:
    """`LineString` からセグメント列を抽出する（有向・無向の切り替えラッパー）。"""
    return list(ordered_segments(ls)) if order else list(undirected_segments(ls))


def get_all_segments(mls: MultiLineString, order: bool = False) -> list[Segment] | list[UndirectedSegment]:
    """`MultiLineString` から重複のない全セグメント列を抽出する。"""
    return list(get_all_ordered_segments(mls)) if order else list(get_all_undirected_segments(mls))


def dist_from_lonlat(p1: Coord, p2: Coord) -> float:
    """2つの `(lon, lat)` 座標間の WGS84 測地線距離 `[m]` を算出する。"""
    _, _, dist = cast(
        tuple[Any, Any, float],
        _WGS84_GEOD.inv(p1[0], p1[1], p2[0], p2[1]),  # pyright: ignore[reportUnknownMemberType]
    )
    return float(dist)


def get_lengths(segments: Sequence[Segment]) -> list[float]:
    """有向セグメント列の各測地線距離 `[m]` をリストとして算出する。"""
    return [dist_from_lonlat(s, t) for s, t in segments]


def get_point_at_length(ls: LineString, at_length: float, ratio: bool = False) -> Point | None:
    """`LineString` 上の指定した測地線距離（または全長に対する比率）に位置する `Point` を補間する。

    `at_length` が `[0, 全長]` の範囲外である場合は `None` を返す。
    """
    segments = ordered_segments(ls)
    if not segments:
        return None

    lengths = get_lengths(segments)
    cumulative_lengths = np.cumsum(lengths)
    total_length = float(cumulative_lengths[-1])
    target_length = at_length * total_length if ratio else at_length

    if target_length < 0.0 or target_length > total_length:
        return None

    idx = int(np.searchsorted(cumulative_lengths, target_length, side="left"))
    segment_end_length = float(cumulative_lengths[idx])
    start_coord, target_coord = segments[idx]
    overrun = segment_end_length - target_length

    azimuth, _, _ = cast(
        tuple[float, Any, Any],
        _WGS84_GEOD.inv(  # pyright: ignore[reportUnknownMemberType]
            target_coord[0], target_coord[1], start_coord[0], start_coord[1]
        ),
    )
    lon, lat, _ = cast(
        tuple[float, float, Any],
        _WGS84_GEOD.fwd(  # pyright: ignore[reportUnknownMemberType]
            target_coord[0], target_coord[1], azimuth, overrun
        ),
    )
    return Point(float(lon), float(lat))


def get_center(ls: LineString) -> Point:
    """空でない `LineString` の測地線中点 `Point` を返す。"""
    center = get_point_at_length(ls, 0.5, ratio=True)
    if center is None:
        msg = "空の LineString から中点を算出することはできません。"
        raise ValueError(msg)
    return center
