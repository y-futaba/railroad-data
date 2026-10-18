# `railroad_data.pipelines`

GIS（QGIS 等）での手動編集を境界とする **Phase 1（自動前処理・トポロジー検査）** と **Phase 2（手動補正データの検証・駅順ソート・通称付与・最終出力）**、および駅接続グラフの可視化処理を提供するパイプライン層パッケージです。

---

## モジュール構成

- `phase1_prepare.py`: 原データの読み込み、路線結合、駅代表点算出、駅点での路線切断、欠落駅線分の補完、隣接駅トポロジー検査と中間シェープファイル出力
- `phase2_finalize.py`: GIS 手動編集済みデータの読み込み・整合性検証、変更ステータス付与、直径パス分解による駅順ソート、事業者・路線通称の付与、Shapefile / TSV / GeoJSON エクスポート
- `visualize.py`: NetworkX と Matplotlib を用いた路線・事業者別の駅接続グラフ描画および画像保存

---

## 1. `phase1_prepare.py`

### クラス

#### `Phase1Result`
Phase 1 パイプラインの各ステップで生成される中間・最終 `GeoDataFrame` / `DataFrame` を保持する不変データクラスです。

- **メンバ変数**
  - `railroads: gpd.GeoDataFrame` — 原データから必要列を抽出した路線データ
  - `merged_railroads: gpd.GeoDataFrame` — 事業者・路線名単位で結合（`dissolve` + `linemerge`）した路線データ
  - `stations_ls: gpd.GeoDataFrame` — 原データの駅 `LineString` データ
  - `stations: gpd.GeoDataFrame` — 各駅の代表点 `Point` データ
  - `line_keys: tuple[LineKey, ...]` — 全路線の `(co, line)` キー列
  - `initial_split_railroads: gpd.GeoDataFrame` — 初回の駅点切断後の路線データ
  - `initial_excluded_stations: tuple[ExcludedStation, ...]` — 初回切断時に路線から欠落していた駅一覧
  - `complemented_railroads: gpd.GeoDataFrame` — 欠落駅の線分を補完した路線データ
  - `re_merged_railroads: gpd.GeoDataFrame` — 補完後に再結合した路線データ
  - `re_split_railroads: gpd.GeoDataFrame` — 補完後に再切断した路線データ
  - `re_excluded_stations: tuple[ExcludedStation, ...]` — 再切断後の欠落駅一覧（空タプルになることを確認する用途）
  - `sta_degree_count_df: pd.DataFrame` — 路線ごとの駅の接続線数（次数 0〜3）別カウント集計表
  - `adj_stas_ele_count_df: pd.DataFrame` — 路線ごとの隣接駅集合の要素数（0〜6）別カウント集計表
  - `irregular_lines: pd.DataFrame` — 分岐・環状・不備により単純な一本線になっていない路線一覧
  - `irregular_sta_set_df: pd.DataFrame` — 異常路線における要素数 2 以外の隣接駅集合一覧
  - `railroads_with_adj_stas: gpd.GeoDataFrame` — 各 `LineString` に `adj_stas` 列と連番 `id` を付与した路線データ

### 関数

- `load_raw_railroads(config: PipelineConfig) -> gpd.GeoDataFrame`
  - 原データ路線シェープファイルを読み込み、`N02_003`, `N02_004`, `geometry` 列を抽出します。
- `merge_railroad_lines(railroads_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame`
  - 事業者・路線名単位で `dissolve` および `ops.linemerge` を適用し、マルチパートを `explode` で単一 `LineString` に分解します。
- `load_and_compute_station_points(config: PipelineConfig, *, save_file: bool = True) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]`
  - 駅シェープファイルを読み込み、各駅の代表点 `Point` を算出して `(stations_ls, stations)` を返します。
- `extract_line_keys(merged_railroads: gpd.GeoDataFrame) -> tuple[LineKey, ...]`
  - 結合済み路線データから一意な `(co, line)` タプル列を取得します。
- `split_all_lines_by_stations(merged_railroads: gpd.GeoDataFrame, stations: gpd.GeoDataFrame, line_keys: Sequence[LineKey], tolerance: float = 1e-5, buffer_ratio: float = 0.9) -> gpd.GeoDataFrame`
  - 全路線をそれぞれの駅代表点で切断し、`(co, line)` インデックス付きの `GeoDataFrame` に統合します。
- `find_excluded_stations(split_railroads: gpd.GeoDataFrame, stations: gpd.GeoDataFrame, line_keys: Sequence[LineKey]) -> tuple[ExcludedStation, ...]`
  - 路線データから頂点が抜けてしまっている駅を検出し、`ExcludedStation` のタプルとして返します。
- `complement_missing_station_segments(railroads: gpd.GeoDataFrame, stations_ls: gpd.GeoDataFrame, excluded_stations: Sequence[ExcludedStation]) -> gpd.GeoDataFrame`
  - 欠落していた駅の `LineString` を路線データへ補完した新しい `GeoDataFrame` を返します。
- `compute_station_degree_and_set_counts(split_railroads: gpd.GeoDataFrame, stations: gpd.GeoDataFrame, line_keys: Sequence[LineKey]) -> tuple[pd.DataFrame, pd.DataFrame]`
  - 駅の次数別カウント表と、隣接駅集合の要素数別カウント表のペアを返します。
- `extract_irregular_lines(adj_stas_ele_count_df: pd.DataFrame) -> pd.DataFrame`
  - 要素数が 1 の集合がちょうど 2 個でない路線、または要素数 0, 3〜6 の集合が存在する路線を抽出します。
- `build_irregular_station_sets_df(irregular_lines: pd.DataFrame, split_railroads: gpd.GeoDataFrame, stations: gpd.GeoDataFrame) -> pd.DataFrame`
  - 異常路線のうち要素数が 2 でない隣接駅集合を抽出し、テーブル形式の `DataFrame` にまとめます。
- `find_isolated_line_coords(split_railroads: gpd.GeoDataFrame, stations: gpd.GeoDataFrame, line_keys: Sequence[LineKey]) -> dict[LineKey, frozenset[Coord]]`
  - 隣接駅数が 0 個（孤立状態）の頂点を持つ路線とその座標集合を辞書として返します。
- `assign_adjacent_stations_to_lines(split_railroads: gpd.GeoDataFrame, stations: gpd.GeoDataFrame, line_keys: Sequence[LineKey]) -> gpd.GeoDataFrame`
  - 全路線の各 `LineString` に `adj_stas` 列と 1 始まりの連番 `id` を付与します。
- `count_lines_per_route(railroads_with_adj_stas: gpd.GeoDataFrame, line_keys: Sequence[LineKey]) -> pd.DataFrame`
  - 路線ごとの `LineString` 行数を集計します。
- `run_phase1(config: PipelineConfig = DEFAULT_CONFIG, *, save_outputs: bool = True) -> Phase1Result`
  - Phase 1 の全ステップを順次実行し、`Phase1Result` を返します。

---

## 2. `phase2_finalize.py`

### クラス

#### `Phase2Result`
Phase 2 パイプラインの検証結果・中間グラフデータ・最終ソート済みデータを保持する不変データクラスです。

- **メンバ変数**
  - `railroads_with_adj_stas: gpd.GeoDataFrame` — Phase 1 で出力された手動編集前の路線データ
  - `rr_adjs_man: gpd.GeoDataFrame` — 手動編集後の路線データ（`status` 列付与済み）
  - `invalid_adj_stas_df: gpd.GeoDataFrame` — 手動編集後に `adj_stas` の要素数が 1・2 以外になっている異常行
  - `duplicate_ids: np.ndarray` — 手動編集データ内で重複している `id` 一覧
  - `mismatched_co_line_df: gpd.GeoDataFrame` — 編集前後で同じ `id` なのに `co` または `line` が異なる行
  - `status_counts: pd.Series[int]` — 変更ステータス（`remain` / `alter` / `copy` / `new`）別の件数
  - `sta_pt: gpd.GeoDataFrame` — 編集前の駅代表点データ
  - `sta_pt_man: gpd.GeoDataFrame` — 手動編集後の駅代表点データ
  - `added_stations_df: gpd.GeoDataFrame` — 手動編集で追加・変更された駅点
  - `removed_stations_df: gpd.GeoDataFrame` — 手動編集で削除・変更された元の駅点
  - `line_graph_summaries: tuple[LineGraphSummary, ...]` — 全路線の駅接続グラフ解析・パス分解結果
  - `sta_graph_df: pd.DataFrame` — 路線ごとの閉路・連結判定と抽出パス列のテーブル
  - `rr_sorted: gpd.GeoDataFrame` — 駅順ソート・`line_id` / `seq_id` 採番・通称付与済みの最終路線データ

### 関数

- `clean_val(x: object) -> object` / `parse_adj_stas_tuple(raw: object) -> tuple[str, ...]`
  - 属性値の文字列トリムと、カンマ区切り駅名からソート済みタプルへの変換を行います。
- `load_and_normalize_adj_stas_gdf(path: Path, encoding: str = "utf-8") -> gpd.GeoDataFrame`
  - 路線シェープファイルを読み込み、`clean_val` と `parse_adj_stas_tuple` で正規化した `GeoDataFrame` を返します。
- `find_invalid_adj_stas_rows(rr_adjs_man: gpd.GeoDataFrame) -> gpd.GeoDataFrame`
  - `adj_stas` の要素数が 1 または 2 でない行を抽出します。
- `find_duplicate_ids(rr_adjs_man: gpd.GeoDataFrame) -> np.ndarray`
  - `id` 列の重複値を抽出します。
- `find_mismatched_co_line_rows(rr_adjs_man: gpd.GeoDataFrame, railroads_with_adj_stas: gpd.GeoDataFrame) -> gpd.GeoDataFrame`
  - 同一 `id` で事業者名または路線名が編集前と食い違っている行を抽出します。
- `assign_modification_status(rr_adjs_man: gpd.GeoDataFrame, railroads_with_adj_stas: gpd.GeoDataFrame) -> gpd.GeoDataFrame`
  - 編集前データとの比較により `status` 列（`remain` / `alter` / `copy` / `new`）を付与します。
- `compare_station_points(sta_pt: gpd.GeoDataFrame, sta_pt_man: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]`
  - 編集前後の駅点データを比較し、追加・変更分と削除・変更分のペアを返します。
- `build_station_graph_table(rr_adjs_man: gpd.GeoDataFrame, image_dir_path: Path | None = None) -> tuple[pd.DataFrame, tuple[LineGraphSummary, ...]]`
  - 全路線の駅接続グラフ構築と直径パス分解を行い、テーブル `DataFrame` と `LineGraphSummary` タプル列を返します。
- `sort_railroads_by_station_order(rr_adjs_man: gpd.GeoDataFrame, summaries: Sequence[LineGraphSummary]) -> gpd.GeoDataFrame`
  - パス分解結果から生成したソートキーで各路線の `LineString` を駅順に並び替え、`line_id` と `seq_id` をインデックスに設定します。
- `assign_aliases_to_sorted_railroads(rr_sorted: gpd.GeoDataFrame) -> gpd.GeoDataFrame`
  - `co_alias`、`service`、`line_alias` 列を解決して付与します。
- `find_duplicate_line_names(sta_graph_df: pd.DataFrame) -> pd.Series[int]`
  - 複数事業者で重複している路線名を集計します。
- `export_final_outputs(rr_sorted: gpd.GeoDataFrame, sta_pt_man: gpd.GeoDataFrame, config: PipelineConfig) -> None`
  - 最終データを Shapefile (ZIP)、TSV、GeoJSON（`EPSG:4326`）として出力します。
- `run_phase2(config: PipelineConfig = DEFAULT_CONFIG, *, save_outputs: bool = True, save_graph_images: bool = False) -> Phase2Result`
  - Phase 2 の全ステップを順次実行し、`Phase2Result` を返します。

---

## 3. `visualize.py`

### 関数

- `save_line_graph_image(graph: nx.Graph[str], output_path: Path) -> None`
  - 単一路線の駅接続グラフを描画し、指定パスに画像として保存します。
- `show_sta_graph(rr_sorted: gpd.GeoDataFrame, co: str = "", line: str = "") -> None`
  - 事業者名のみ指定時は路線ごと、路線名のみ指定時は事業者ごとにエッジを色分けし、両方指定時は単一路線の駅接続グラフを描画します。

