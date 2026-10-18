# `railroad_data.topology`

路線 `LineString` と駅代表点 `Point` のトポロジー解析（隣接駅の探索・欠落駅の検出）、および手動補正後の駅接続グラフ構築・直径パス反復抽出によるパス分解・駅順ソートキー生成を提供するトポロジー解析層パッケージです。

---

## モジュール構成

- `adjacency.py`: 頂点グラフ構築、路線から欠落した駅座標の検出、非駅頂点連結成分からの隣接駅探索、各 `LineString` への `adj_stas` 付与
- `graph_sort.py`: 駅接続グラフの閉路・連結判定、直径（最長最短経路）パスの再帰的抽出、エッジ・ノード混在列の駅順ソート関数生成

---

## 1. `adjacency.py`

### 関数

- `extract_point_coord(pt: Point) -> Coord`
  - `Point` ジオメトリから `(lon, lat)` の2次元座標タプルを抽出します。
- `gen_graph_from_ls_gs(ls_gs: gpd.GeoSeries) -> nx.Graph[Coord]`
  - `LineString` の `GeoSeries` から全セグメントを抽出し、頂点座標をノードとする無向グラフ `nx.Graph[Coord]` を構築します。
- `edge_to_adj(segments: Iterable[tuple[Coord, Coord] | frozenset[Coord]]) -> dict[Coord, set[Coord]]`
  - セグメント列から各頂点の隣接頂点集合の辞書を構築します。
- `excluded_sta_coords(ls_gs: gpd.GeoSeries, sta_pt_gs: gpd.GeoSeries) -> set[Coord]`
  - 路線の全頂点座標集合に含まれていない（路線から抜けてしまっている）駅代表点の座標集合を返します。
- `edge_stations(ls_gs: gpd.GeoSeries, sta_pt_gs: gpd.GeoSeries) -> tuple[dict[Coord, frozenset[Coord]], dict[frozenset[Coord], frozenset[Coord]]]`
  - 頂点グラフから駅座標を除いた部分グラフの連結成分（`nx.connected_components`）を求め、各連結成分に隣接する駅座標集合（隣接駅）を算出します。
  - 第1返り値は「非駅頂点座標 → 隣接駅座標の `frozenset`」、第2返り値は「隣接駅座標の `frozenset` → 非駅頂点座標の `frozenset`」です。
- `adjacent_stations(edge_stas_to_coords: Mapping[frozenset[Coord], frozenset[Coord]]) -> set[frozenset[Coord]]`
  - 要素数が 2 以上の隣接駅座標集合のみを抽出します。
- `check_adjacent_stations_with_ls(ls_df: gpd.GeoDataFrame, sta_pt_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame`
  - 各 `LineString` に含まれる非駅頂点の隣接駅集合が一意に定まる場合はカンマ区切りの駅名文字列、複数または0個で定まらない場合は `'❗️'` を格納した `adj_stas` 列を付与して返します。

---

## 2. `graph_sort.py`

### 関数

- `has_cycle(graph: nx.Graph[str]) -> bool`
  - 駅接続グラフに閉路（サイクル）が存在するかを `nx.find_cycle` により判定します。
- `find_diameter_path(sub_comp: nx.Graph[str]) -> tuple[str, ...]`
  - 連結な部分グラフから、離心率（`nx.eccentricity`）がグラフ直径（`nx.diameter`）に一致する2ノード間の最短経路（直径パス）を1本抽出します。
- `extract_all_diameter_paths(graph: nx.Graph[str]) -> tuple[tuple[str, ...], ...]`
  - 各連結成分に対して `find_diameter_path` で直径パスを抽出し、その辺を除去した残余グラフへ再帰的に同じ処理を適用することで、すべての辺を順序付きパス列へ分解します。
- `analyze_line_graph(co: str, line: str, adj_stas_series: pd.Series[Any]) -> tuple[LineGraphSummary, nx.Graph[str] | None]`
  - 1路線の `adj_stas` シリーズから2駅間のエッジを抽出して駅接続グラフを構築し、閉路有無・連結判定・直径パス分解の結果をまとめた `LineGraphSummary` とグラフ本体を返します。
- `sorter_factory(paths_list: Sequence[Sequence[str]]) -> Callable[[pd.Series[Any]], pd.Series[float]]`
  - パス列内の各ステップ `k` に対して、エッジ `(u, v)` に整数順序 `float(k)`、始点ノード `u` に `k - 0.5`、終点ノード `v` に `k + 0.5`（いずれも初出優先）を割り当て、`DataFrame.sort_values(by="adj_stas", key=...)` に渡すソートキー関数を返します。
