"""鉄道データ処理で共有する型エイリアスおよび不変の値オブジェクト定義。"""

from dataclasses import dataclass
from typing import Literal

type Coord = tuple[float, float]
type Segment = tuple[Coord, Coord]
type UndirectedSegment = frozenset[Coord]
type LineKey = tuple[str, str]
type StationKey = tuple[str, str, str]
type LineStatus = Literal["remain", "alter", "copy", "new", ""]


@dataclass(frozen=True, slots=True)
class ExcludedStation:
    """結合・切断後の路線 `LineString` から駅の線分が欠落していた駅を表す値オブジェクト。"""

    co: str
    line: str
    station_name: str


@dataclass(frozen=True, slots=True)
class LineGraphSummary:
    """単一路線の駅接続グラフ解析結果と、分解された駅順パス列を保持する値オブジェクト。"""

    co: str
    line: str
    cycled: bool
    connected: bool
    paths: tuple[tuple[str, ...], ...]

    def to_flat_row(self) -> tuple[str, str, bool, bool, *tuple[str, ...]]:
        """テーブル出力（CSV/TSV）向けにフラットなタプルへ変換する。"""
        return (
            self.co,
            self.line,
            self.cycled,
            self.connected,
            *tuple(map(",".join, self.paths)),
        )
