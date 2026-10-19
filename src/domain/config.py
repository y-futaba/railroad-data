"""Phase 1 および Phase 2 パイプラインの入出力パス・幾何パラメータを管理する不変設定。"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """パイプライン全体のディレクトリ階層・ファイルパスおよび幾何演算パラメータを保持する設定クラス。

    ディレクトリ構成は `base_data_dir / str(year) / {raw, phase1, manual, phase2}` の
    年次ファースト・4ステージ分割を採用し、`year` から原データファイル名（`N02-XX_...`）および
    全ステージの入出力パスを宣言的に導出する。
    """

    # データルートおよび対象年度
    base_data_dir: Path = Path("data")
    year: int = 2022

    # 各ステージのファイル・ディレクトリ名
    station_pt_name: str = "station_pt.shp"
    irregular_sta_set_csv_name: str = "irregular_sta_set.csv"
    railroad_adj_stas_name: str = "railroad_adj_stas.shp"

    railroad_adj_stas_manual_name: str = "railroad_adj_stas_manual.shp"
    station_pt_manual_name: str = "station_pt_manual.shp"

    sta_graph_csv_name: str = "sta_graph.csv"
    image_dir_name: str = "images"
    rr_sorted_dir_name: str = "rr_sorted"
    rr_sorted_shp_name: str = "rr_sorted.shp"
    rr_sorted_zip_stem_name: str = "rr_sorted"
    rr_sorted_tsv_name: str = "rr_sorted.tsv"
    rr_geojson_name: str = "rr.geojson"
    sta_geojson_name: str = "sta.geojson"

    # 幾何演算・入出力パラメータ
    snap_tolerance: float = 1e-5
    buffer_ratio: float = 0.9
    encoding: str = "utf-8"
    target_crs: str = "EPSG:4326"

    @property
    def year_short(self) -> str:
        """国土数値情報のファイル名に使用される西暦下2桁（例: `2022` -> `"22"`）を返す。"""
        return f"{self.year % 100:02d}"

    @property
    def year_dir(self) -> Path:
        """対象年度のルートディレクトリ（`data/<year>`）を返す。"""
        return self.base_data_dir / str(self.year)

    @property
    def raw_dir(self) -> Path:
        """原データ格納ディレクトリ（`data/<year>/raw`）を返す。"""
        return self.year_dir / "raw"

    @property
    def phase1_dir(self) -> Path:
        """Phase 1 出力ディレクトリ（`data/<year>/phase1`）を返す。"""
        return self.year_dir / "phase1"

    @property
    def manual_dir(self) -> Path:
        """GIS 手動編集済みデータ格納ディレクトリ（`data/<year>/manual`）を返す。"""
        return self.year_dir / "manual"

    @property
    def phase2_dir(self) -> Path:
        """Phase 2 出力ディレクトリ（`data/<year>/phase2`）を返す。"""
        return self.year_dir / "phase2"

    # --- 0. raw（原データ入力） ---
    @property
    def railroads_path(self) -> Path:
        """原データ路線シェープファイル（`data/<year>/raw/N02-XX_RailroadSection.shp`）のパスを返す。"""
        return self.raw_dir / f"N02-{self.year_short}_RailroadSection.shp"

    @property
    def stations_path(self) -> Path:
        """原データ駅シェープファイル（`data/<year>/raw/N02-XX_Station.shp`）のパスを返す。"""
        return self.raw_dir / f"N02-{self.year_short}_Station.shp"

    # --- 1. phase1（自動前処理・検査出力） ---
    @property
    def station_pt_path(self) -> Path:
        """駅代表点シェープファイル（`data/<year>/phase1/station_pt.shp`）のパスを返す。"""
        return self.phase1_dir / self.station_pt_name

    @property
    def irregular_sta_set_csv_path(self) -> Path:
        """不整合隣接駅一覧 CSV（`data/<year>/phase1/irregular_sta_set.csv`）のパスを返す。"""
        return self.phase1_dir / self.irregular_sta_set_csv_name

    @property
    def railroad_adj_stas_path(self) -> Path:
        """隣接駅付与済み路線シェープファイル（`data/<year>/phase1/railroad_adj_stas.shp`）のパスを返す。"""
        return self.phase1_dir / self.railroad_adj_stas_name

    # --- 2. manual（GIS 手動編集済み入力） ---
    @property
    def railroad_adj_stas_manual_path(self) -> Path:
        """GIS 手動補正後の路線シェープファイル（`data/<year>/manual/railroad_adj_stas_manual.shp`）のパスを返す。"""
        return self.manual_dir / self.railroad_adj_stas_manual_name

    @property
    def station_pt_manual_path(self) -> Path:
        """GIS 手動補正後の駅代表点シェープファイル（`data/<year>/manual/station_pt_manual.shp`）のパスを返す。"""
        return self.manual_dir / self.station_pt_manual_name

    # --- 3. phase2（検証・ソート・通称一次付与出力） ---
    @property
    def sta_graph_csv_path(self) -> Path:
        """駅接続グラフ解析結果 CSV/TSV（`data/<year>/phase2/sta_graph.csv`）のパスを返す。"""
        return self.phase2_dir / self.sta_graph_csv_name

    @property
    def image_dir_path(self) -> Path:
        """非連結・閉路グラフ画像の保存先ディレクトリ（`data/<year>/phase2/images`）を返す。"""
        return self.phase2_dir / self.image_dir_name

    @property
    def rr_sorted_dir_path(self) -> Path:
        """ソート済みシェープファイル格納ディレクトリ（`data/<year>/phase2/rr_sorted`）を返す。"""
        return self.phase2_dir / self.rr_sorted_dir_name

    @property
    def rr_sorted_shp_path(self) -> Path:
        """ソート済み路線シェープファイル（`data/<year>/phase2/rr_sorted/rr_sorted.shp`）のパスを返す。"""
        return self.rr_sorted_dir_path / self.rr_sorted_shp_name

    @property
    def rr_sorted_zip_stem(self) -> str:
        """ZIP アーカイブ出力時のステムパス文字列（`data/<year>/phase2/rr_sorted`）を返す。"""
        return str(self.phase2_dir / self.rr_sorted_zip_stem_name)

    @property
    def rr_sorted_tsv_path(self) -> Path:
        """通称の手動再付与・調整用の属性テーブル TSV（`data/<year>/phase2/rr_sorted.tsv`）のパスを返す。"""
        return self.phase2_dir / self.rr_sorted_tsv_name

    @property
    def rr_geojson_path(self) -> Path:
        """地図アプリ動作確認用の仮路線 GeoJSON（`data/<year>/phase2/rr.geojson`）のパスを返す。"""
        return self.phase2_dir / self.rr_geojson_name

    @property
    def sta_geojson_path(self) -> Path:
        """地図アプリ動作確認用の仮駅代表点 GeoJSON（`data/<year>/phase2/sta.geojson`）のパスを返す。"""
        return self.phase2_dir / self.sta_geojson_name


DEFAULT_CONFIG = PipelineConfig()
