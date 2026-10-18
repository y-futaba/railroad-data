"""Phase 1 および Phase 2 パイプラインの入出力パス・幾何パラメータを管理する不変設定。"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """パイプライン全体のファイルパスおよび幾何演算パラメータを保持する設定クラス。"""

    # 原データ入力シェープファイル
    railroads_path: Path = Path("N02-22_RailroadSection.shp")
    stations_path: Path = Path("N02-22_Station.shp")

    # Phase 1 出力
    station_pt_path: Path = Path("station_pt.shp")
    irregular_sta_set_csv_path: Path = Path("irregular_sta_set.csv")
    railroad_adj_stas_path: Path = Path("railroad_adj_stas.shp")

    # Phase 2 手動編集済み入力
    railroad_adj_stas_manual_path: Path = Path("railroad_adj_stas_manual.shp")
    station_pt_manual_path: Path = Path("station_pt_manual.shp")

    # Phase 2 出力
    sta_graph_csv_path: Path = Path("sta_graph.csv")
    image_dir_path: Path = Path("images")
    rr_sorted_dir_path: Path = Path("rr_sorted")
    rr_sorted_shp_name: str = "rr_sorted.shp"
    rr_sorted_zip_stem: str = "rr_sorted"
    rr_sorted_tsv_path: Path = Path("rr_sorted.tsv")
    rr_geojson_path: Path = Path("rr.geojson")
    sta_geojson_path: Path = Path("sta.geojson")

    # 幾何演算・入出力パラメータ
    snap_tolerance: float = 1e-5
    buffer_ratio: float = 0.9
    encoding: str = "utf-8"
    target_crs: str = "EPSG:4326"

    @property
    def rr_sorted_shp_path(self) -> Path:
        """`rr_sorted_dir_path` 内のソート済み路線シェープファイルのフルパスを返す。"""
        return self.rr_sorted_dir_path / self.rr_sorted_shp_name


DEFAULT_CONFIG = PipelineConfig()
