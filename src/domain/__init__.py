"""ドメイン層：型エイリアス、不変データモデル、通称マスタ辞書、パイプライン設定。"""

from src.domain.aliases import (
    CO_ALIASES,
    LINE_ALIASES,
    SERVICE_LINE_ALIASES,
    SERVICES,
    ResolvedLineAliases,
    resolve_line_aliases,
)
from src.domain.config import DEFAULT_CONFIG, PipelineConfig
from src.domain.types import (
    Coord,
    ExcludedStation,
    LineGraphSummary,
    LineKey,
    LineStatus,
    Segment,
    StationKey,
    UndirectedSegment,
)

__all__ = [
    "CO_ALIASES",
    "DEFAULT_CONFIG",
    "LINE_ALIASES",
    "SERVICES",
    "SERVICE_LINE_ALIASES",
    "Coord",
    "ExcludedStation",
    "LineGraphSummary",
    "LineKey",
    "LineStatus",
    "PipelineConfig",
    "ResolvedLineAliases",
    "Segment",
    "StationKey",
    "UndirectedSegment",
    "resolve_line_aliases",
]
