# `railroad_data.geometry`

測地線（WGS84 楕円体）に基づく距離・中点計算、`LineString` からのセグメント抽出、駅の代表点算出、および駅代表点バッファによる路線の切断・端点スナップを提供する幾何演算層パッケージです。すべて副作用のない純粋関数として実装されています。

---

## モジュール構成

- `geodesic.py`: 座標・セグメント抽出、WGS84 測地線距離計算、`LineString` 上の距離補間・中点算出
- `split.py`: 駅代表点の抽出、円形バッファ生成、端点スナップ、駅点での路線分割

---

## 1. `geodesic.py`

### 定数

- `_WGS84_GEOD: pyproj.Geod` — WGS84 楕円体の測地線計算オブジェクト（モジュール内で共有）

### 関数

- `extract_coords(ls: LineString) -> tuple[Coord, ...]`
  - `LineString` の頂点列を `(lon, lat)` の不変タプル列として抽出します。
- `ordered_segments(ls: LineString) -> tuple[Segment, ...]`
  - 隣り合う頂点ペアを有向セグメント `(start, target)` のタプルとして抽出します。
- `undirected_segments(ls: LineString) -> tuple[UndirectedSegment, ...]`
  - 隣り合う頂点ペアを無向セグメント `frozenset({start, target})` のタプルとして抽出します。
- `get_all_ordered_segments(mls: MultiLineString) -> tuple[Segment, ...]` / `get_all_undirected_segments(mls: MultiLineString) -> tuple[UndirectedSegment, ...]`
  - `MultiLineString` 内の全 `LineString` から重複を除いた有向 / 無向セグメント列を取得します。
- `segments_from_ls(ls: LineString, order: bool = False) -> list[Segment] | list[UndirectedSegment]` / `get_all_segments(mls: MultiLineString, order: bool = False) -> list[Segment] | list[UndirectedSegment]`
  - 既存コード互換のセグメント取得ラッパー関数です。
- `dist_from_lonlat(p1: Coord, p2: Coord) -> float`
  - 2点間の WGS84 測地線距離 `[m]` を返します。
- `get_lengths(segments: Sequence[Segment]) -> list[float]`
  - セグメント列の各測地線距離 `[m]` のリストを返します。
- `get_point_at_length(ls: LineString, at_length: float, ratio: bool = False) -> Point | None`
  - `LineString` の始点から指定した距離（`ratio=True` の場合は全長に対する割合 `0.0〜1.0`）に位置する `Point` を測地線補間により算出します。範囲外の場合は `None` を返します。
- `get_center(ls: LineString) -> Point`
  - `LineString` の測地線中点（`ratio=0.5`）の `Point` を返します。

---

## 2. `split.py`

### 関数

- `representative_station_point(row: pd.Series[Any]) -> Point`
  - 駅データの1行から代表点 `Point` を算出します（九州旅客鉄道・鹿児島線・広木駅のみ最東端頂点、それ以外は `get_center` による測地線中点）。
- `cir_buffer(pt: Point, r: float, num_points: int = 16) -> Polygon`
  - 点 `pt` を中心とする半径 `r` `[deg]` の正 `num_points` 角形バッファ `Polygon` を生成します。
- `pt_norm_sq(p1: Point, p2: Point) -> float`
  - 2つの `Point` 間の度数空間における二乗ユークリッド距離を返します。
- `snap_edge(ls: LineString, pt_gs: gpd.GeoSeries, tolerance: float) -> LineString`
  - `LineString` の始点・終点が `pt_gs` 内のいずれかの点から許容誤差 `tolerance` `[deg]` 以内にある場合、最も近い点の座標へ端点をスナップした新しい `LineString` を返します。
- `split_line_at_point(ls_gdf: gpd.GeoDataFrame, pt_gdf: gpd.GeoDataFrame, tolerance: float, buffer_ratio: float = 0.9) -> gpd.GeoDataFrame`
  - 引数の `GeoDataFrame` を破壊的に変更せず、駅点の円形バッファ（半径 `buffer_ratio * tolerance`）で路線をくり抜き（`symmetric_difference`）、マルチパートを分解（`explode`）した上で、切断端点を `snap_edge` で駅点へスナップした新しい `GeoDataFrame` を返します。
