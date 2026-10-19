# railroad-data

国土交通省「国土数値情報（鉄道データ N02）」の路線・駅シェープファイルを入力とし、路線（`LineString`）の駅代表点（`Point`）による分割、トポロジー解析、GIS（QGIS 等）での手動補正データの検証、駅順ソート、および事業者・路線通称の自動一次付与を行い、通称の手動再付与用データ（`rr_sorted.tsv` / `rr_sorted.shp`）や地図アプリケーション向けの仮 GeoJSON を生成するパッケージおよびスクリプトです。

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

## ディレクトリ構成（コード・データ階層）

ソースコードは `src/` 配下に責務ごとの 4 サブパッケージとして整理されており、入出力データは `data/<year>/{raw,phase1,manual,phase2}/` の **年次ファースト・4ステージ構造** で一元管理されます（`src/domain/config.py` の `PipelineConfig` により `year` から自動導出）。

```text
railroad-data/
├── railroad.py                                 # ステップ実行・対話確認用スクリプト（# %% セル構造）
├── src/
│   ├── domain/                                 # 型エイリアス・不変データ構造・通称マスタ辞書・パイプライン設定
│   │   ├── types.py                            # Coord, Segment, LineKey, StationKey, ExcludedStation, LineGraphSummary
│   │   ├── aliases.py                          # CO_ALIASES, LINE_ALIASES, SERVICES, resolve_line_aliases
│   │   └── config.py                           # PipelineConfig, DEFAULT_CONFIG
│   ├── geometry/                               # 測地線計算・駅代表点算出・駅点による路線 LineString 分割
│   │   ├── geodesic.py                         # extract_coords, dist_from_lonlat, get_lengths, get_center
│   │   └── split.py                            # representative_station_point, snap_edge, split_line_at_point
│   ├── topology/                               # NetworkX による駅隣接探索・直径パス分解・駅順ソーター生成
│   │   ├── adjacency.py                        # gen_graph_from_ls_gs, edge_stations, adjacent_stations
│   │   └── graph_sort.py                       # extract_all_diameter_paths, analyze_line_graph, sorter_factory
│   ├── pipelines/                              # Phase 1 / Phase 2 のパイプライン処理とグラフ可視化
│   │   ├── phase1_prepare.py                   # 路線結合・駅点抽出・路線切断・欠落駅補完・隣接駅検査（run_phase1）
│   │   ├── phase2_finalize.py                  # 手動補正検証・パス分解・駅順ソート・通称の自動一次付与・出力（run_phase2）
│   │   └── visualize.py                        # 駅接続グラフの描画・画像保存（show_sta_graph）
│   ├── cli.py                                  # CLI エントリポイント
│   └── __main__.py                             # python -m src 実行サポート
└── data/
    └── <year>/                                 # 対象データの西暦年度（例: 2022, 2023, ...）
        ├── raw/                                # Stage 0: 国土数値情報 原データ（入力）
        │   ├── N02-<YY>_RailroadSection.shp
        │   └── N02-<YY>_Station.shp
        ├── phase1/                             # Stage 1: Phase 1 自動前処理・検査出力
        │   ├── station_pt.shp
        │   ├── irregular_sta_set.csv
        │   └── railroad_adj_stas.shp
        ├── manual/                             # Stage 2: Manual Step 1（QGIS 等での手動編集済みデータ）
        │   ├── railroad_adj_stas_manual.shp
        │   └── station_pt_manual.shp
        └── phase2/                             # Stage 3: Phase 2 検証・駅順ソート・通称一次付与出力
            ├── sta_graph.csv
            ├── images/                         # 閉路・非連結グラフの確認画像
            ├── rr_sorted/                      # ソート・通称一次付与済みシェープファイル格納ディレクトリ
            │   └── rr_sorted.shp
            ├── rr_sorted.zip
            ├── rr_sorted.tsv                   # Manual Step 2（会社名・路線通称の手動再付与用テーブル）
            ├── rr.geojson                      # 地図アプリ動作確認用の仮路線 GeoJSON
            └── sta.geojson                     # 地図アプリ動作確認用の仮駅代表点 GeoJSON
```

---

## 実行方法

### 1. コマンドライン（CLI）からの一括実行

`--year` オプション（デフォルト: `2022`）で対象年度を指定すると、`data/<year>/...` 配下の入出力パスと原データファイル名（`N02-<YY>_...`）が自動的に選択されます。

```bash
# Phase 1: 原データの路線結合・駅代表点算出・路線切断・隣接駅検査（デフォルト: 2022年度）
python -m src phase1 --year 2022

# Phase 2: GIS 手動補正データの検証・駅順ソート・通称の自動一次付与・手動再付与用 TSV/Shapefile 出力
python -m src phase2 --year 2022 --save-graph-images

# 特定の事業者・路線の駅接続グラフを描画して確認
python -m src plot --year 2022 --co 東日本旅客鉄道 --line 東海道線
```

- ファイル書き出しを行わずに検証のみ行う場合は `--no-save` オプションを指定できます。

### 2. インタラクティブ実行（`railroad.py`）

`railroad.py` は VS Code Interactive Window 等で `# %%` セル単位でステップ実行できるよう構成されており、先頭セルの `config = PipelineConfig(year=2022)` の年度を変更するだけで任意の年度のデータを対話的に処理・確認できます。

---

## 処理パイプライン

大きく **Phase 1（GIS 手動編集前の自動前処理・検査）**、**Manual Step 1（GIS による路線・駅の手動編集）**、**Phase 2（手動編集データの検証・駅順ソート・通称の自動一次付与・ファイル出力）**、および **Manual Step 2（`rr_sorted.tsv` 等での会社名・路線通称の手動再付与）** の流れで構成されています。

### Phase 1: 駅代表点の抽出・路線の切断・隣接駅検査（`src.pipelines.phase1_prepare`）

1. **路線データの結合（`load_raw_railroads`, `merge_railroad_lines`）**
   - `data/<year>/raw/N02-<YY>_RailroadSection.shp` を読み込み、事業者名（`N02_004`）・路線名（`N02_003`）単位で `dissolve` および `shapely.ops.linemerge` により可能な限り単一の `LineString` に結合します。
2. **駅の代表点算出（`load_and_compute_station_points`）**
   - `data/<year>/raw/N02-<YY>_Station.shp` の各駅 `LineString` から測地線長に基づく中点（`get_center`、例外として九州旅客鉄道・鹿児島線・広木駅のみ最東端点）を代表点 `Point`（`representative_station_point`）として算出し、`data/<year>/phase1/station_pt.shp` に出力します。
3. **駅代表点による路線の切断と欠落区間の補完（`split_all_lines_by_stations`, `find_excluded_stations`, `complement_missing_station_segments`）**
   - 駅代表点の周囲に微小円バッファ（許容誤差 `1e-5` 度 ≒ 約 1m）を生成して路線をくり抜き、切断端点を駅代表点へスナップします（`split_line_at_point`）。
   - 路線データから駅の線分が欠落している駅（`excluded_sta_coords`）を特定し、駅の `LineString` を路線側に補完した上で再度 `dissolve`・`linemerge`・駅点切断を実行します。
4. **隣接駅のトポロジー解析と検査出力（`compute_station_degree_and_set_counts`, `build_irregular_station_sets_df`, `assign_adjacent_stations_to_lines`）**
   - 路線の各座標から駅以外の頂点を探索して到達可能な隣接駅集合を算出します（`edge_stations`）。
   - 分岐・環状・データ不備などにより隣接駅数が 2 にならない不整合箇所を抽出し、`data/<year>/phase1/irregular_sta_set.csv` に出力します。
   - 各 `LineString` に辞書順ソート済みの隣接駅属性 `adj_stas` を付与し、路線 `(co, line)` ごとに `(adj_stas, (始点座標, 終点座標, 頂点数))` で安定ソート（`mergesort`）した上で一意な連番 `id` を採番し、`data/<year>/phase1/railroad_adj_stas.shp` を出力します。

---

### Manual Step 1: QGIS 等による路線・駅代表点の手動編集

Phase 1 で `data/<year>/phase1/` に出力された以下のファイルを GIS 上で編集し、すべての線分の隣接駅数が 1 または 2 になるよう路線線分や駅点の修正・分割・追加を行い、`data/<year>/manual/` に保存します。

- `data/<year>/phase1/railroad_adj_stas.shp` → 手動編集後 **`data/<year>/manual/railroad_adj_stas_manual.shp`** として保存
- `data/<year>/phase1/station_pt.shp` → 手動編集後 **`data/<year>/manual/station_pt_manual.shp`** として保存

---

### Phase 2: 手動編集データの検証・駅順ソート・通称の自動一次付与・出力（`src.pipelines.phase2_finalize`）

1. **手動編集データの整合性チェックと変更ステータスの付与（`find_invalid_adj_stas_rows`, `find_duplicate_ids`, `assign_modification_status`, `compare_station_points`）**
   - `data/<year>/manual/railroad_adj_stas_manual.shp` の `adj_stas` 要素数や `id` 重複を検査し、編集前データとの差分から `status` 列（`remain` / `alter` / `copy` / `new`）を付与します。
   - 駅データについても `station_pt.shp` と `station_pt_manual.shp` の差分を確認します。
2. **駅接続グラフの構築と決定論的パス分解（`build_station_graph_table`）**
   - `NetworkX` で路線ごとに駅間の接続グラフ（ノード・エッジを辞書順に整列）を構築し、閉路（`cycled`）や非連結（`connected`）の有無を判定・画像出力（`data/<year>/phase2/images/`）します。
   - 各連結成分を `(-頂点数, 最小駅名)` 順に走査し、直径パス（最長最短経路）の両端駅を辞書順最小でタイブレークして反復的に抽出・辺削除（パスの向きも先頭駅名 `<=` 末尾駅名に正規化）することで、実行ごとに変動しない一意な順序付きパス列へ分解し、`data/<year>/phase2/sta_graph.csv` に出力します。
3. **駅順による `LineString` の決定論的並び替え（`sort_railroads_by_station_order`）**
   - 抽出されたパス列からエッジ（整数順序 `k`）と単一駅ノード（半整数順序 `k - 0.5` / `k + 0.5`）の順序キーを生成（`sorter_factory`）します。
   - 同一の `adj_stas` を持つ複数 `LineString` 間の順序も固定するため、第1キー `adj_stas` 順序スコア、第2キー `id`、第3キー `(始点座標, 終点座標, 頂点数)` の複合キーと安定ソート（`mergesort`）によって並び替え、常に同じ線分に同じ `line_id` と `seq_id` を採番します。
4. **事業者・運行系統・路線通称の自動一次付与とエクスポート（`assign_aliases_to_sorted_railroads`, `export_final_outputs`）**
   - マスタ辞書（`src.domain.aliases` の `CO_ALIASES`, `SERVICES`, `LINE_ALIASES`, `SERVICE_LINE_ALIASES`）に基づき、事業者略称（`co_alias`）、サービス名（`service`）、路線通称（`line_alias`）をおおまかに自動付与します（`railroad.py` のセクション 14）。
   - その後、区間ごとに路線愛称・系統名が分かれる路線など、辞書だけでは一意に決まらない通称を手動で再付与・調整するため、`data/<year>/phase2/rr_sorted.tsv` および `data/<year>/phase2/rr_sorted/rr_sorted.shp`（`rr_sorted.zip`）として出力します（併せて、地図アプリの動作確認用に仮の `rr.geojson`・`sta.geojson` を出力します）。

---

### Manual Step 2: `rr_sorted.tsv` / `rr_sorted.shp` での会社名・路線通称の手動再付与

Phase 2 で `data/<year>/phase2/` に自動一次付与された `rr_sorted.tsv`（または `rr_sorted/rr_sorted.shp`）を開き、同一の `(co, line)` 内で区間（`seq_id`）によって異なる愛称や運行系統を持つ路線などについて、`co_alias`・`service`・`line_alias` を手動で再付与・調整します。

---

## 入出力ファイル一覧（`data/<year>/` 配下）

| ステージ | 相対パス（`data/<year>/` 基準） | 説明 |
| :--- | :--- | :--- |
| **Stage 0: `raw/`（原データ入力）** | `raw/N02-<YY>_RailroadSection.shp` | 国土数値情報 鉄道データ（路線 `LineString`、`<YY>` は西暦下2桁） |
| **Stage 0: `raw/`（原データ入力）** | `raw/N02-<YY>_Station.shp` | 国土数値情報 鉄道データ（駅 `LineString`、`<YY>` は西暦下2桁） |
| **Stage 1: `phase1/`（Phase 1 出力）** | `phase1/station_pt.shp` | 駅の代表点（`Point`）シェープファイル |
| **Stage 1: `phase1/`（Phase 1 出力）** | `phase1/irregular_sta_set.csv` | 隣接駅数が 2 でない不整合箇所の一覧 |
| **Stage 1: `phase1/`（Phase 1 出力）** | `phase1/railroad_adj_stas.shp` | 駅点で切断し隣接駅属性 `adj_stas` と連番 `id` を付与した路線データ |
| **Stage 2: `manual/`（Phase 2 手動入力）** | `manual/railroad_adj_stas_manual.shp` | `phase1/railroad_adj_stas.shp` を GIS で手動補正した路線データ |
| **Stage 2: `manual/`（Phase 2 手動入力）** | `manual/station_pt_manual.shp` | `phase1/station_pt.shp` を GIS で手動補正した駅代表点データ |
| **Stage 3: `phase2/`（Phase 2 中間出力）** | `phase2/sta_graph.csv` | 路線ごとの閉路・連結判定と抽出された駅パス列（TSV 形式） |
| **Stage 3: `phase2/`（Phase 2 中間出力）** | `phase2/images/<co>_<line>.png` | 閉路または非連結と判定された路線の駅接続グラフ画像 |
| **Stage 3: `phase2/`（手動再付与用出力）** | `phase2/rr_sorted/rr_sorted.shp` (`phase2/rr_sorted.zip`) | 駅順ソート・通称の自動一次付与済みの路線シェープファイル（手動での通称追加・調整用） |
| **Stage 3: `phase2/`（手動再付与用出力）** | `phase2/rr_sorted.tsv` | `rr_sorted` の属性テーブル（ジオメトリ除外、会社名・路線通称の手動再付与用） |
| **Stage 3: `phase2/`（仮出力）** | `phase2/rr.geojson` | Web 地図アプリの動作確認用の仮路線 GeoJSON（`EPSG:4326`） |
| **Stage 3: `phase2/`（仮出力）** | `phase2/sta.geojson` | Web 地図アプリの動作確認用の仮駅代表点 GeoJSON（`EPSG:4326`） |