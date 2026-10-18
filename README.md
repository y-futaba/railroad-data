# railroad-data

国土交通省「国土数値情報（鉄道データ N02）」の路線・駅シェープファイルを入力とし、路線（`LineString`）の駅代表点（`Point`）による分割、トポロジー解析、GIS（QGIS 等）での手動補正データの検証、駅順ソート、および事業者・路線通称の付与を行い、地図アプリケーション向けの GeoJSON / Shapefile / TSV を生成するパッケージおよびスクリプトです。

---

## 必要環境・依存パッケージ

- **Python**: `>= 3.14`
- **主な依存ライブラリ** (`pyproject.toml`):
  - `geopandas >= 1.1.2`
  - `shapely >= 2.1.2`
  - `pyproj >= 3.7.2`
  - `networkx >= 3.6.1`
  - `matplotlib >= 3.11.0`
  - `pandas`, `numpy`

---

## パッケージ構成（`railroad_data`）

処理責務ごとに以下の 4 サブパッケージと CLI エントリポイントへ分割されています（各ディレクトリの詳細はそれぞれの `README.md` を参照）。

```text
railroad-data/
├── railroad.py                     # ステップ実行・対話確認用スクリプト（# %% セル構造）
└── railroad_data/
    ├── domain/                     # 型エイリアス・不変データ構造・通称辞書・パイプライン設定
    │   ├── types.py                # Coord, Segment, LineKey, StationKey, ExcludedStation, LineGraphSummary
    │   ├── aliases.py              # CO_ALIASES, LINE_ALIASES, SERVICES, resolve_line_aliases
    │   └── config.py               # PipelineConfig, DEFAULT_CONFIG
    ├── geometry/                   # 測地線計算・駅代表点算出・駅点による路線 LineString 分割
    │   ├── geodesic.py             # extract_coords, dist_from_lonlat, get_lengths, get_center
    │   └── split.py                # representative_station_point, snap_edge, split_line_at_point
    ├── topology/                   # NetworkX による駅隣接探索・直径パス分解・駅順ソーター生成
    │   ├── adjacency.py            # gen_graph_from_ls_gs, edge_stations, adjacent_stations
    │   └── graph_sort.py           # extract_all_diameter_paths, analyze_line_graph, sorter_factory
    ├── pipelines/                  # Phase 1 / Phase 2 のパイプライン処理とグラフ可視化
    │   ├── phase1_prepare.py       # 路線結合・駅点抽出・路線切断・欠落駅補完・隣接駅検査（run_phase1）
    │   ├── phase2_finalize.py      # 手動補正検証・パス分解・駅順ソート・通称付与・出力（run_phase2）
    │   └── visualize.py            # 駅接続グラフの描画・画像保存（show_sta_graph）
    ├── cli.py                      # CLI エントリポイント
    └── __main__.py                 # python -m railroad_data 実行サポート
```

---

## 実行方法

### 1. コマンドライン（CLI）からの一括実行

```bash
# Phase 1: 原データの路線結合・駅代表点算出・路線切断・隣接駅検査
python -m railroad_data phase1

# Phase 2: GIS 手動補正データの検証・駅順ソート・通称付与・最終ファイル出力
python -m railroad_data phase2 --save-graph-images

# 特定の事業者・路線の駅接続グラフを描画して確認
python -m railroad_data plot --co 東日本旅客鉄道 --line 東海道線
```

- ファイル書き出しを行わずに検証のみ行う場合は `--no-save` オプションを指定できます。

### 2. インタラクティブ実行（`railroad.py`）

`railroad.py` は VS Code Interactive Window 等で `# %%` セル単位でステップ実行できるよう構成されており、各工程の中間 `GeoDataFrame` や検証結果 `DataFrame` を対話的に確認できます。

---

## 処理パイプライン

大きく **Phase 1（GIS 手動編集前の自動前処理・検査）** と **Phase 2（GIS 手動編集後の検証・ソート・最終出力）** の 2 フェーズに分かれています。

### Phase 1: 駅代表点の抽出・路線の切断・隣接駅検査（`railroad_data.pipelines.phase1_prepare`）

1. **路線データの結合（`load_raw_railroads`, `merge_railroad_lines`）**
   - `N02-22_RailroadSection.shp` を読み込み、事業者名（`N02_004`）・路線名（`N02_003`）単位で `dissolve` および `shapely.ops.linemerge` により可能な限り単一の `LineString` に結合します。
2. **駅の代表点算出（`load_and_compute_station_points`）**
   - `N02-22_Station.shp` の各駅 `LineString` から測地線長に基づく中点（`get_center`、例外として九州旅客鉄道・鹿児島線・広木駅のみ最東端点）を代表点 `Point`（`representative_station_point`）として算出し、`station_pt.shp` に出力します。
3. **駅代表点による路線の切断と欠落区間の補完（`split_all_lines_by_stations`, `find_excluded_stations`, `complement_missing_station_segments`）**
   - 駅代表点の周囲に微小円バッファ（許容誤差 `1e-5` 度 ≒ 約 1m）を生成して路線をくり抜き、切断端点を駅代表点へスナップします（`split_line_at_point`）。
   - 路線データから駅の線分が欠落している駅（`excluded_sta_coords`）を特定し、駅の `LineString` を路線側に補完した上で再度 `dissolve`・`linemerge`・駅点切断を実行します。
4. **隣接駅のトポロジー解析と検査出力（`compute_station_degree_and_set_counts`, `build_irregular_station_sets_df`, `assign_adjacent_stations_to_lines`）**
   - 路線の各座標から駅以外の頂点を探索して到達可能な隣接駅集合を算出します（`edge_stations`）。
   - 分岐・環状・データ不備などにより隣接駅数が 2 にならない不整合箇所を抽出し、`irregular_sta_set.csv` に出力します。
   - 各 `LineString` に隣接駅属性 `adj_stas` と一意な `id` を付与した `railroad_adj_stas.shp` を出力します。

---

### Manual Step: QGIS 等による手動編集

Phase 1 で出力された以下のファイルを GIS 上で編集し、すべての線分の隣接駅数が 1 または 2 になるよう路線線分や駅点の修正・分割・追加を行います。

- `railroad_adj_stas.shp` → 手動編集後 **`railroad_adj_stas_manual.shp`** として保存
- `station_pt.shp` → 手動編集後 **`station_pt_manual.shp`** として保存

---

### Phase 2: 手動編集データの検証・駅順ソート・通称付与・最終出力（`railroad_data.pipelines.phase2_finalize`）

1. **手動編集データの整合性チェックと変更ステータスの付与（`find_invalid_adj_stas_rows`, `find_duplicate_ids`, `assign_modification_status`, `compare_station_points`）**
   - `railroad_adj_stas_manual.shp` の `adj_stas` 要素数や `id` 重複を検査し、編集前データとの差分から `status` 列（`remain` / `alter` / `copy` / `new`）を付与します。
   - 駅データについても `station_pt.shp` と `station_pt_manual.shp` の差分を確認します。
2. **駅接続グラフの構築とパス分解（`build_station_graph_table`）**
   - `NetworkX` で路線ごとに駅間の接続グラフを構築し、閉路（`cycled`）や非連結（`connected`）の有無を判定・画像出力します。
   - 各連結成分から直径パス（最長最短経路）を反復的に抽出・辺削除することで路線を順序付きパス列へ分解し、`sta_graph.csv` に出力します。
3. **駅順による `LineString` の並び替え（`sort_railroads_by_station_order`）**
   - 抽出されたパス列からエッジとノードの順序キーを生成（`sorter_factory`）し、各路線の `LineString` を始点から終点の駅順にソートして `line_id` と `seq_id` を採番します。
4. **事業者・運行系統・路線通称の付与とエクスポート（`assign_aliases_to_sorted_railroads`, `export_final_outputs`）**
   - 事業者略称（`co_alias`）、サービス名（`service`）、路線通称（`line_alias`）を辞書マッピングで付与します。
   - 最終結果を `rr_sorted/rr_sorted.shp`（`rr_sorted.zip`）、`rr_sorted.tsv`、および Web 地図アプリ用の `rr.geojson`・`sta.geojson`（`EPSG:4326`）として出力します。

---

## 入出力ファイル一覧

| 種別 | ファイル名 | 説明 |
| :--- | :--- | :--- |
| **入力** | `N02-22_RailroadSection.shp` | 国土数値情報 鉄道データ（路線 `LineString`） |
| **入力** | `N02-22_Station.shp` | 国土数値情報 鉄道データ（駅 `LineString`） |
| **中間出力** | `station_pt.shp` | 駅の代表点（`Point`）シェープファイル |
| **中間出力** | `irregular_sta_set.csv` | 隣接駅数が 2 でない不整合箇所の一覧 |
| **中間出力** | `railroad_adj_stas.shp` | 駅点で切断し隣接駅属性 `adj_stas` を付与した路線データ |
| **手動入力** | `railroad_adj_stas_manual.shp` | `railroad_adj_stas.shp` を GIS で手動補正した路線データ |
| **手動入力** | `station_pt_manual.shp` | `station_pt.shp` を GIS で手動補正した駅代表点データ |
| **中間出力** | `sta_graph.csv` | 路線ごとの閉路・連結判定と抽出された駅パス列（TSV 形式） |
| **最終出力** | `rr_sorted/rr_sorted.shp` (`rr_sorted.zip`) | 駅順ソート・通称付与済みの路線シェープファイル |
| **最終出力** | `rr_sorted.tsv` | `rr_sorted` の属性テーブル（ジオメトリ除外） |
| **最終出力** | `rr.geojson` | Web 地図アプリ用の路線 GeoJSON（`EPSG:4326`） |
| **最終出力** | `sta.geojson` | Web 地図アプリ用の駅代表点 GeoJSON（`EPSG:4326`） |