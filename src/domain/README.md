# `railroad_data.domain`

鉄道データ処理パイプライン全体で共有される**型エイリアス・不変データモデル（値オブジェクト）・通称マスタ定義・パイプライン設定**を提供するドメイン層パッケージです。すべてのデータクラスは `@dataclass(frozen=True, slots=True)` により不変（イミュータブル）として定義されています。

---

## モジュール構成

- `types.py`: 幾何・トポロジー処理の基本型エイリアスと、解析結果を表す値オブジェクト
- `aliases.py`: 事業者名・運行系統（サービス名）・路線通称の不変マスタ辞書と名前解決関数
- `config.py`: 入出力ファイルパスおよび幾何パラメータを保持する設定クラス

---

## 1. `types.py`

### 型エイリアス

| 名称 | 型定義 | 説明 |
| :--- | :--- | :--- |
| `Coord` | `tuple[float, float]` | `(lon, lat)` または `(x, y)` の2次元座標ペア |
| `Segment` | `tuple[Coord, Coord]` | 向きを持つ2点間の線分（始点・終点のペア） |
| `UndirectedSegment` | `frozenset[Coord]` | 向きを持たない2点間の線分 |
| `LineKey` | `tuple[str, str]` | `(co, line)`（事業者名・路線名）の複合キー |
| `StationKey` | `tuple[str, str, str]` | `(co, line, station_name)`（事業者名・路線名・駅名）の複合キー |
| `LineStatus` | `Literal["remain", "alter", "copy", "new", ""]` | GIS手動編集前後の `LineString` 変更ステータス |

### クラス

#### `ExcludedStation`
結合・駅点切断後の路線 `LineString` から駅の線分が欠落していた駅を表す値オブジェクトです。

- **メンバ変数**
  - `co: str` — 事業者名（`N02_004`）
  - `line: str` — 路線名（`N02_003`）
  - `station_name: str` — 駅名（`N02_005`）

#### `LineGraphSummary`
1路線ごとの駅接続グラフ解析結果と、直径パス反復抽出によって分解された駅順パス列を保持する値オブジェクトです。

- **メンバ変数**
  - `co: str` — 事業者名
  - `line: str` — 路線名
  - `cycled: bool` — 駅接続グラフに閉路（ループ）が存在するか
  - `connected: bool` — 駅接続グラフが単一の連結成分として繋がっているか
  - `paths: tuple[tuple[str, ...], ...]` — 抽出された駅名タプル（パス）のタプル
- **メソッド**
  - `to_flat_row() -> tuple[str, str, bool, bool, *tuple[str, ...]]`: テーブル出力（`sta_graph.csv` 等）向けに、`(co, line, cycled, connected, "駅A,駅B,...", ...)` 形式のフラットなタプルへ変換します。

---

## 2. `aliases.py`

### 定数（不変マッピング）

- `CO_ALIASES: Mapping[str, str]` — 国土数値情報の事業者名（`co`）から事業者通称（`co_alias`、例: `"東日本旅客鉄道" -> "JR東日本"`）への辞書
- `LINE_ALIASES: Mapping[str, str]` — 路線名（`line`）から路線通称（`line_alias`、例: `"3号線銀座線" -> "銀座線"`）への辞書
- `SERVICES: Mapping[str, str]` — 事業者通称（`co_alias`）から運行サービス名（`service`、例: `"名古屋市交通局" -> "名古屋市営地下鉄"`）への辞書
- `SERVICE_LINE_ALIASES: Mapping[LineKey, tuple[str, str]]` — 路線名が他社と重複する路線や複数系統を持つ事業者について、`(co, line)` から `(service, line_alias)` を個別指定する辞書

### クラス・関数

#### `ResolvedLineAliases`
1路線に対して解決された通称群を保持する値オブジェクトです。

- **メンバ変数**
  - `co_alias: str` — 事業者通称
  - `service: str` — 運行サービス名
  - `line_alias: str` — 路線通称
  - `service_overridden: bool` — `SERVICE_LINE_ALIASES` によって `SERVICES` のデフォルト値から `service` が上書きされたかどうか（デフォルト: `False`）

#### `resolve_line_aliases(co: str, line: str) -> ResolvedLineAliases`
`co -> service -> line -> (co, line)` の優先順で通称を解決し、`ResolvedLineAliases` を返す純粋関数です。

---

## 3. `config.py`

### クラス

#### `PipelineConfig`
Phase 1・Phase 2 の入出力パスおよび幾何処理パラメータを一元管理する不変設定クラスです。

- **メンバ変数（入力ファイル）**
  - `railroads_path: Path` — 原データ路線シェープファイル（デフォルト: `Path("N02-22_RailroadSection.shp")`）
  - `stations_path: Path` — 原データ駅シェープファイル（デフォルト: `Path("N02-22_Station.shp")`）
- **メンバ変数（Phase 1 出力）**
  - `station_pt_path: Path` — 駅代表点シェープファイル（デフォルト: `Path("station_pt.shp")`）
  - `irregular_sta_set_csv_path: Path` — 不整合隣接駅一覧 CSV（デフォルト: `Path("irregular_sta_set.csv")`）
  - `railroad_adj_stas_path: Path` — 隣接駅付与済み路線シェープファイル（デフォルト: `Path("railroad_adj_stas.shp")`）
- **メンバ変数（Phase 2 手動編集入力）**
  - `railroad_adj_stas_manual_path: Path` — GIS手動補正後の路線シェープファイル（デフォルト: `Path("railroad_adj_stas_manual.shp")`）
  - `station_pt_manual_path: Path` — GIS手動補正後の駅代表点シェープファイル（デフォルト: `Path("station_pt_manual.shp")`）
- **メンバ変数（Phase 2 出力）**
  - `sta_graph_csv_path: Path` — 駅接続グラフ解析結果 CSV/TSV（デフォルト: `Path("sta_graph.csv")`）
  - `image_dir_path: Path` — 非連結・閉路グラフ画像の保存先ディレクトリ（デフォルト: `Path("images")`）
  - `rr_sorted_dir_path: Path` — ソート済みシェープファイル格納ディレクトリ（デフォルト: `Path("rr_sorted")`）
  - `rr_sorted_shp_name: str` — ソート済みシェープファイル名（デフォルト: `"rr_sorted.shp"`）
  - `rr_sorted_zip_stem: str` — ZIP アーカイブ出力時のベース名（デフォルト: `"rr_sorted"`）
  - `rr_sorted_tsv_path: Path` — 属性テーブル TSV パス（デフォルト: `Path("rr_sorted.tsv")`）
  - `rr_geojson_path: Path` — 路線 GeoJSON パス（デフォルト: `Path("rr.geojson")`）
  - `sta_geojson_path: Path` — 駅代表点 GeoJSON パス（デフォルト: `Path("sta.geojson")`）
- **メンバ変数（幾何・I/O パラメータ）**
  - `snap_tolerance: float` — 駅点での路線切断・スナップ時の許容誤差 `[deg]`（デフォルト: `1e-5`）
  - `buffer_ratio: float` — くり抜き円バッファ半径の `snap_tolerance` に対する比率（デフォルト: `0.9`）
  - `encoding: str` — ファイル入出力時の文字エンコーディング（デフォルト: `"utf-8"`）
  - `target_crs: str` — GeoJSON 出力時の座標参照系（デフォルト: `"EPSG:4326"`）
- **プロパティ（メソッド）**
  - `rr_sorted_shp_path -> Path`: `rr_sorted_dir_path / rr_sorted_shp_name` を結合したシェープファイルのフルパスを返します。

### 定数

- `DEFAULT_CONFIG: PipelineConfig` — デフォルトパラメータで初期化された `PipelineConfig` のシングルトンインスタンス
