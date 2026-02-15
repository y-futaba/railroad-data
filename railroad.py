"""
# 定義、import
---

Point、点：(x,y) あるいは (lon, lat) の実数値のペアで座標が指定される点\
LineString、線分：単一の座標列からなる分岐や輪を含まない線\
MultiLineString：LineString の集合\
セグメント：LineString 内の任意の隣り合う 2 点のみから構成される、線の最小単位

---

!pip install graphillion

## import
"""

!pip install japanize-matplotlib

import geopandas as gpd
from shapely import ops
from shapely.geometry import Point, LineString, MultiLineString, Polygon
import pyproj
import pandas as pd
import numpy as np
# from graphillion import GraphSet as gs
import itertools
import networkx as nx
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import japanize_matplotlib
import shutil

def segments_from_ls(ls: LineString, order=False):
    starts = ls.coords[:-1]
    targets = ls.coords[1:]
    if order:
        return list(zip(starts, targets))
    else:
        return [frozenset(seg) for seg in zip(starts, targets)]

def get_all_segments(mls: MultiLineString, order=False):
    return list(
        {seg for ls in mls.geoms for seg in segments_from_ls(ls, order)}
    )

def dist_from_lonlat(p1, p2):
    wgs84 = pyproj.Geod(ellps="WGS84")
    azimuth, bkw_azimuth, dist = wgs84.inv(*p1, *p2)
    return dist

def get_lengths(segments):
    return [dist_from_lonlat(*seg) for seg in segments]

def get_point_at_length(ls: LineString, at_length: float, ratio=False) -> Point:
    wgs84 = pyproj.Geod(ellps="WGS84")
    segments = segments_from_ls(ls, order=True)
    lengths = get_lengths(segments)
    cumulative_lengths = np.cumsum(lengths)
    length = cumulative_lengths[-1]
    if ratio:
        at_length = at_length * length
    if (at_length < 0 or at_length > length):
        return np.nan

    i = np.searchsorted(cumulative_lengths, at_length, side='left')
    # at_length から 直前のセグメントまでの累積長を引いた距離をセグメントの始点に足す場合，i = 0 の場合に注意
    segment_end_length = cumulative_lengths[i]
    s, t = segments[i]
    overrun = segment_end_length - at_length
    azimuth, bkw_azimuth, dist = wgs84.inv(*t, *s)
    lon, lat, bkw_azimuth = wgs84.fwd(*t, azimuth, overrun)
    return Point(lon, lat)

def get_center(ls: LineString):
    return get_point_at_length(ls, 0.5, True)

def representative_station_point(row):
    if (row["N02_005"] == "広木") and (row["N02_003"] == "鹿児島線") and (row["N02_004"] == "九州旅客鉄道"):
        coords = row["geometry"].coords
        lons = [lon for lon, lat in coords]
        return Point(
            list(coords)[lons.index( max(lons) )]
        )
    else:
        return get_center(row["geometry"])

def cir_buffer(pt: Point, r: float):
    # r [deg]
    rads = 2 * np.pi * np.arange(16) / 16
    return Polygon(list(zip(pt.x + r * np.cos(rads), pt.y + r * np.sin(rads))))

def pt_norm_sq(p1: Point, p2: Point):
    return np.power(p1.x - p2.x, 2) + np.power(p1.y - p2.y, 2)

def snap_edge(
    ls: LineString,
    pt_gs: gpd.GeoSeries,
    tolerance: float
) -> LineString:
    new_coords = list(ls.coords)
    # ラインの終点[-1]と始点[0]を処理
    for i in [-1, 0]:
        edge_pt = Point(ls.coords[i])
        distance_gs = pt_gs.apply(pt_norm_sq, p2=edge_pt)
        # ポイントから許容誤差範囲内の端点に対して，端点の座標をポイントの座標に差し替え
        if distance_gs.min() < np.power(tolerance, 2):
            new_pt = pt_gs[distance_gs.idxmin()]
            new_coords[i] = list(new_pt.coords)[0]
    return LineString(new_coords)

def split_line_at_point(
    ls_gdf: gpd.GeoDataFrame,
    pt_gdf: gpd.GeoDataFrame,
    tolerance: float
) -> gpd.GeoDataFrame:
    # ポイントから許容誤差でバッファ
    pt_gdf["_buffer_poly"] = pt_gdf["geometry"].apply(cir_buffer, r=0.9 * tolerance)
    # ラインをバッファでくり抜き
    ls_gdf = gpd.overlay(
        df1=ls_gdf,
        df2=pt_gdf[["_buffer_poly"]].set_geometry("_buffer_poly"),
        how="symmetric_difference",
        keep_geom_type=True
    )
    ls_gdf = ls_gdf.explode(index_parts=False)
    # 端点をポイントにスナップ
    ls_gdf["geometry"] = ls_gdf["geometry"].apply(
        snap_edge,
        pt_gs=pt_gdf["geometry"],
        tolerance=tolerance
    )
    return ls_gdf

def gen_graph_from_ls_gs(ls_gs: gpd.GeoSeries):
    g = nx.Graph()
    for segments in ls_gs.apply(segments_from_ls, order=True):
        g.add_edges_from(segments)
    return g

def edge_to_adj(segments):
    adjs = defaultdict(set)
    for (s, t) in segments:
        adjs[s].add(t)
        adjs[t].add(s)
    return dict(adjs)

def excluded_sta_coords(ls_gs: gpd.GeoSeries, sta_pt_gs: gpd.GeoSeries):
    all_coords = set(ls_gs.map(lambda ls: list(ls.coords)).explode())
    sta_coords = set(sta_pt_gs.map(lambda pt: (pt.x, pt.y)))
    return sta_coords - all_coords

def edge_stations(ls_gs: gpd.GeoSeries, sta_pt_gs: gpd.GeoSeries):
    # segments_from_ls() が set を返してきても動く
    adjs = edge_to_adj(
        ls_gs.map(segments_from_ls).explode()
    )
    all_coords = set(ls_gs.map(lambda ls: list(ls.coords)).explode())
    sta_coords = set(sta_pt_gs.map(lambda pt: (pt.x, pt.y)))
    # for coord in sta_coords - all_coords:
    #     print(stations[stations["geometry"] == Point(coord)])

    coord_to_edge_stas = {}
    while sta_coords < all_coords:
        root = (all_coords - sta_coords).pop()
        stack = [root]
        visited = {root}
        edge_stas = set()
        while stack:
            current = stack.pop()
            for adj in adjs[current]:
                if adj in visited: continue
                if adj in sta_coords:
                    edge_stas.add(adj)
                    continue
                stack.append(adj)
                visited.add(adj)

        for coord in visited:
            coord_to_edge_stas[coord] = frozenset(edge_stas)
        all_coords -= visited

    edge_stas_to_coords = defaultdict(set)
    for coord, edge_stas in coord_to_edge_stas.items():
        edge_stas_to_coords[edge_stas].add(coord)

    return coord_to_edge_stas, {k: frozenset(v) for k, v in edge_stas_to_coords.items()}

def adjacent_stations(edge_stas_to_coords):
    return {stas_set for stas_set in edge_stas_to_coords.keys() if len(stas_set) >= 2}

def check_adjacent_stations_with_ls(ls_df: gpd.GeoDataFrame, sta_pt_df: gpd.GeoDataFrame):
    ls_gs = ls_df["geometry"]
    coord_to_edge_stas = edge_stations(ls_gs.copy(), sta_pt_df["geometry"].copy())[0]
    def func(row):
        ls = row["geometry"]
        # flag = 0
        # stas = set()
        edge_stas_set = set()
        for coord in ls.coords:
            if coord in coord_to_edge_stas.keys():
                edge_stas_set.add(coord_to_edge_stas[coord])
        #     else:
        #         stas.add(coord)
        if len(edge_stas_set) == 1:
            return ",".join(list({
                sta_pt_df[sta_pt_df["geometry"] == Point(edge_sta)][["N02_005"]].iloc[0,0]
                for edge_sta in edge_stas_set.pop()
            }))
        else:
            return "❗️"
        #     flag = 1
        # all_edge_stas = {edge_sta for edge_stas in edge_stas_set for edge_sta in edge_stas}
        # if (len(stas) >= 1) and len([0 for sta in stas if sta not in all_edge_stas]):
        #     print(i, "❗️隣接駅の set に含まれていない駅があります")
        #     flag = 1

    ls_df["adj_stas"] = ls_df.apply(
        func,
        axis=1
    )

    return ls_df

"""## フォント"""

!apt-get update -y
!apt-get install -y fonts-noto-cjk

# システムにあるフォント一覧を取得
import matplotlib.font_manager as fm
fonts = fm.findSystemFonts()
for font in fonts:
    print(font)

# フォントキャッシュを削除
import shutil
from pathlib import Path
shutil.rmtree(Path("~/.cache/matplotlib").expanduser())
print("フォントキャッシュを削除しました。ランタイムを再起動してください。")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# フォントのパスを指定
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

# フォントプロパティを作成
font_prop = fm.FontProperties(fname=font_path)

# matplotlib に適用
plt.rcParams["font.family"] = font_prop.get_name()
print(plt.rcParams["font.family"])

# テストプロット
plt.figure(figsize=(5,3))
plt.text(0.5, 0.5, "テスト: 日本語が表示されますか？", fontsize=15, ha='center')
plt.show()

"""# 駅で路線切断

## 路線ごとにライン結合
"""

railroads_path = "N02-22_RailroadSection.shp"
railroads_org = gpd.read_file(railroads_path, encoding="utf-8")
railroads = railroads_org[["N02_003", "N02_004", "geometry"]].copy()

# 完全には結合できない
# GeoJson は QGIS での処理速度がゴミ
dissolved_railroads = railroads.dissolve(by=["N02_004", "N02_003"])
merged_railroads = dissolved_railroads.copy()
merged_railroads["geometry"] = dissolved_railroads["geometry"].map(ops.linemerge)
merged_railroads = merged_railroads.explode(index_parts=False)

print([x for x in dissolved_railroads["geometry"][0].geoms[0].coords])

"""## 駅の代表点"""

stations_path = "N02-22_Station.shp"
stations_ls= gpd.read_file(stations_path, encoding="utf-8")

stations = stations_ls.copy()
stations["geometry"] = stations.apply(representative_station_point, axis=1)
print(stations.crs)
stations.to_file("station_pt.shp", encoding="utf-8")
stations

"""## 駅点で路線を切断"""

lines = tuple(set(merged_railroads.index))
splits = []
for co, line_name in lines:
    splits.append(split_line_at_point(
            merged_railroads.loc[co, line_name],
            # stations の index を一意でないものにすると snap_edge() 内の distance_gs.idxmin() が複数得られて機能しなくなる
            stations[(stations["N02_004"] == co) & (stations["N02_003"] == line_name)].copy(),
            # 9e-6 ° の差は北緯 25° から 45° の間では 1 m 以内
            1e-5
        ).assign(co=co, line=line_name)
    )
    print(co, line_name)
railroad_split_by_stations = pd.concat(splits).set_index(["co", "line"]).sort_index()

"""## 駅の線データと被る部分に抜けがないか確認

<img width="500" src="https://cdn.discordapp.com/attachments/1248517724351500308/1272458828323229807/image.png?ex=68d642a1&is=68d4f121&hm=77266764c0ac28df8e31f5be27bdaef71af03e738026a72977a52554a7e932f0">
"""

excluded_stas = [
    [co, line_name,
        stations[
            (stations["N02_004"] == co) &
            (stations["N02_003"] == line_name) &
            (stations["geometry"] == Point(coord))
        ].iat[0, 4]
    ]
    for co, line_name in lines
    for coord in excluded_sta_coords(
        railroad_split_by_stations.loc[co, line_name]["geometry"],
        stations[(stations["N02_004"] == co) & (stations["N02_003"] == line_name)].copy()["geometry"]
    )
]
excluded_stas

"""## 同じ操作を繰り返す
---

上の調査より、railroad から station の線分が抜けている路線があるので、その部分を補って再度 railroad を station の点で切断

---
"""

dissolved_stations_ls = stations_ls.dissolve(by=["N02_004", "N02_003", "N02_005"])
re_railroads = railroads.copy()
for co, line_name, sta_name in excluded_stas:
    tmp_df = re_railroads[(re_railroads["N02_004"] == co) & (re_railroads["N02_003"] == line_name)][:1].copy()
    tmp_df["geometry"] = dissolved_stations_ls.at[(co, line_name, sta_name), "geometry"]
    re_railroads = pd.concat([re_railroads, tmp_df.explode(index_parts=False)])

re_dissolved_railroads = re_railroads.dissolve(by=["N02_004", "N02_003"])
re_merged_railroads = re_dissolved_railroads.copy()
re_merged_railroads["geometry"] = re_dissolved_railroads["geometry"].map(ops.linemerge)
re_merged_railroads = re_merged_railroads.explode(index_parts=False)
# re_merged_railroad.to_file("line_merged_railroad.shp", encoding="utf-8")

splits = []
for co, line_name in lines:
    splits.append(split_line_at_point(
            re_merged_railroads.loc[co, line_name],
            # stations の index を一意でないものにすると snap_edge() 内の distance_gs.idxmin() が複数得られて機能しなくなる
            stations[(stations["N02_004"] == co) & (stations["N02_003"] == line_name)].copy(),
            # 9e-6 ° の差は北緯 25° から 45° の間では 1 m 以内
            1e-5
        ).assign(co=co, line=line_name)
    )
re_railroad_split_by_stations = pd.concat(splits).set_index(["co", "line"]).sort_index()
# re_railroad_split_by_stations.to_file("railroad_split_by_stations.shp", encoding="utf-8")
re_railroad_split_by_stations

re_excluded_stas = [
    [co, line_name,
        stations[
            (stations["N02_004"] == co) &
            (stations["N02_003"] == line_name) &
            (stations["geometry"] == Point(coord))
        ].iat[0, 4]
    ]
    for co, line_name in lines
    for coord in excluded_sta_coords(
        re_railroad_split_by_stations.loc[co, line_name]["geometry"],
        stations[(stations["N02_004"] == co) & (stations["N02_003"] == line_name)].copy()["geometry"]
    )
]
re_excluded_stas

"""# 隣接駅の調査・追加
---

railroad の線分を構成する点について、その点を含む線分と、その線分と接続する（端点を共有する）線分のうち、駅の点以外を再帰的に辿って着く駅を、その点の隣接駅と呼称する。\
駅を端点とする線分を駅の隣接線と呼称する。\
路線ごとに隣接駅と隣接線を調査する。

---
"""

sta_degree_counts = []
adj_stas_ele_counts = []
for co, line_name in lines:
    edge_stas_to_coords = edge_stations(
        re_railroad_split_by_stations.loc[co, line_name]["geometry"],
        stations[(stations["N02_004"] == co) & (stations["N02_003"] == line_name)]["geometry"].copy()
    )[1]
    adj_stas = adjacent_stations(edge_stas_to_coords)

    sta_degree_counts.append(
        stations[(stations["N02_004"] == co) & (stations["N02_003"] == line_name)]["geometry"]
        .map(
            lambda pt: sum( 1 if (pt.x, pt.y) in adj_sta else 0 for adj_sta in adj_stas )
        ).value_counts().sort_index().rename(f"{co} {line_name}")
    )

    adj_stas_ele_counts.append(
        pd.Series(edge_stas_to_coords.keys()).map(len).value_counts().sort_index().rename(f"{co} {line_name}")
    )
sta_degree_count_df = pd.DataFrame(sta_degree_counts).fillna(0).astype(int).reindex(columns=[0, 1, 2, 3])
adj_stas_ele_count_df = pd.DataFrame(adj_stas_ele_counts).fillna(0).astype(int).reindex(columns=[0,1,2,3,4,5,6])

"""## 駅に繋がっている線の数別にカウント
---

隣接線の個数（0, 1, 2, 3）を列として、その個数だけ隣接線を持つ駅の数を路線ごとにカウントする。

---
"""

sta_degree_count_df

"""## 隣接する駅同士をまとめた set 型の要素数ごとにカウント
---

隣接駅を set 型に入れる。\
路線ごとに全ての点について set を集計すると、路線にループがない場合 (駅の数) + 1 種類の set が得られるはずである。\
それらのうち、set 内の要素数（つまり隣接駅の数）0, 1, ..., 6 を列として set をカウントする。

---
"""

adj_stas_ele_count_df

"""## 駅同士の接続がおかしい路線とその駅
---

始点から終点まで辿ったときに全ての駅を通過できる路線である場合、要素数が 1 個である隣接駅 set が 2 つ、2 個である隣接駅 set が (全駅数) - 1 つできるはずである。\
この条件に合致しない、つまり分岐やループがあったり、LineString データに異常があったりする路線を列挙する。\
また、そのような路線のうち、要素数が 2 個でない隣接駅 set を列挙し、DataFrame にして表示し、CSV として出力する。

---
"""

irregular_lines = adj_stas_ele_count_df[(
    adj_stas_ele_count_df[0] +
    adj_stas_ele_count_df[3] +
    adj_stas_ele_count_df[4] +
    adj_stas_ele_count_df[5] +
    adj_stas_ele_count_df[6] > 0
) | (adj_stas_ele_count_df[1] != 2)]
irregular_lines

irregular_sta_sets = []
for index, values in irregular_lines.to_dict(orient="index").items():
    co, line_name = index.split()
    sta_df = stations[(stations["N02_004"] == co) & (stations["N02_003"] == line_name)][["N02_005", "geometry"]]
    edge_stas_to_coords = edge_stations(
        re_railroad_split_by_stations.loc[co, line_name]["geometry"],
        sta_df["geometry"].copy()
    )[1]
    print()
    print("-" * 20, co, line_name, "-" * 20)
    # print({k: f"{v} 個" for k, v in values.items() if v and k != 2})
    edge_stas_list = sorted(
        list({
            frozenset(
                {sta_df[sta_df["geometry"] == Point(edge_sta)].iat[0,0] for edge_sta in edge_stas}
            )
            for edge_stas in edge_stas_to_coords.keys()
            if len(edge_stas) != 2
        }),
        key=len
    )
    for edge_stas in edge_stas_list:
        print(*[sta.ljust(10, "　") for sta in edge_stas])

    irregular_sta_sets.append(pd.DataFrame([
        {"co": co, "line": line_name} | {i: edge_sta for i, edge_sta in enumerate(edge_stas) }
        for edge_stas in edge_stas_list
    ]))

irregular_sta_set_df = pd.concat(irregular_sta_sets).fillna("").sort_values(["co", "line"]).reset_index(drop=True)
irregular_sta_set_df.to_csv("irregular_sta_set.csv")
irregular_sta_set_df

"""## 接続している駅が 0 の点
---

隣接駅数が 0 個の点を列挙。

---
"""

for co, line_name in lines:
    edge_stas_to_coords = edge_stations(
        re_railroad_split_by_stations.loc[co, line_name]["geometry"],
        stations[(stations["N02_004"] == co) & (stations["N02_003"] == line_name)]["geometry"].copy()
    )[1]
    if frozenset() in edge_stas_to_coords:
        print(co, line_name)
        for coord in edge_stas_to_coords[frozenset()]:
            print(coord)
        print()

"""## 線の GeoDataFrame の属性に隣接駅を追加、出力"""

railroads_with_adj_stas = pd.concat([
    check_adjacent_stations_with_ls(
        re_railroad_split_by_stations.loc[co, line_name].copy(),
        stations[(stations["N02_004"] == co) & (stations["N02_003"] == line_name)]
    )
    for co, line_name in lines
]).sort_values(["co", "line"])
railroads_with_adj_stas.insert(1, "id", np.arange(len(railroads_with_adj_stas)) + 1)

railroads_with_adj_stas.to_file("railroad_adj_stas.shp", encoding="utf-8")
railroads_with_adj_stas

"""## 各路線の行 (LineString) の数"""

pd.DataFrame([
    [ co, line_name, len(railroads_with_adj_stas.loc[co, line_name]) ]
    for co, line_name in lines
], columns=["co", "line", "count"]).set_index(["co", "line"])

"""---
# ここで `railroad_with_adj_stas.shp` を GIS で編集して隣接駅を調整
---

# GIS で隣接駅を編集した `.shp` ファイル
---

全ての線分について、隣接駅数が 1 か 2 になるよう、路線の線分と駅点を削除・複製・移動など行った `.shp` ファイル

---

## `adj_stas` の要素数確認、異常があれば QGIS で修正
"""

railroad_with_adj_stas_manual_path = "railroad_adj_stas_manual.shp"
rr_adjs_man = gpd.read_file(railroad_with_adj_stas_manual_path, encoding="utf-8").map(lambda x: "" if type(x) is None else (x.strip() if type(x) is str else x))
rr_adjs_man["adj_stas"] = rr_adjs_man["adj_stas"].map(lambda x: tuple(sorted(x.split(","))))

railroads_with_adj_stas = gpd.read_file("railroad_adj_stas.shp", encoding="utf-8").map(lambda x: "" if x is None else (x.strip() if type(x) is str else x))
railroads_with_adj_stas["adj_stas"] = railroads_with_adj_stas["adj_stas"].map(lambda x: tuple(sorted(x.split(","))))

rr_adjs_man[~rr_adjs_man["adj_stas"].map(lambda x: len(x) in (1,2))]

"""## org と manual を比較して status を追加

### id の重複確認
"""

unique_id, dup_count = np.unique(rr_adjs_man["id"], return_counts=True)
unique_id[dup_count != 1]

"""### 同じ id で co, line が違うもの"""

def app_func(row):
    idx = railroads_with_adj_stas["id"] == row["id"]
    if idx.sum() == 0:
        return False
    b = railroads_with_adj_stas[idx].iloc[0, :]
    return (b["co"] != row["co"]) or (b["line"] != row["line"])

rr_adjs_man[rr_adjs_man.apply(app_func, axis=1)]

"""### 路線の status 追加
---

このコードは 0 以外の id が一意であることを仮定している

---
"""

def app_func(row):
    if row["id"] == 0:
        return "copy" if railroads_with_adj_stas["geometry"].geom_equals(row["geometry"]).any() else "new"
    if (idx := railroads_with_adj_stas["id"] == row["id"]).any():
        if idx.sum() > 1:
            print("🛑", row["id"])
        # return "remain" if row["adj_stas"] == railroads_with_adj_stas[idx]["adj_stas"].iat[0] else "alter"
        return "remain" if row["adj_stas"] == railroads_with_adj_stas[idx]["adj_stas"].iat[0] else "alter"
    return ""

rr_adjs_man["status"] = rr_adjs_man.apply(app_func, axis=1)
rr_adjs_man.groupby("status").size()

"""## 駅データ"""

sta_pt = gpd.read_file("station_pt.shp", encoding="utf-8")
sta_pt_man = gpd.read_file("station_pt_manual.shp", encoding="utf-8")

def app_func(row, df):
    bools = df["geometry"].geom_equals(row["geometry"])
    return bools.any() and ((df[bools] == row).all(axis=1).any())

sta_pt_man[~sta_pt_man.apply(app_func, args=(sta_pt,), axis=1)] # 元にない man の駅点

sta_pt[~sta_pt.apply(app_func, args=(sta_pt_man,), axis=1)] # man にない元の駅点

"""# 駅同士の接続をグラフで表現"""

def app_func(row):
    co, line = row["co"], row["line"]
    adjs = rr_adjs_man.query('(co == @co) and (line == @line)')["adj_stas"]
    adjs_u = adjs[adjs.map(len) == 2].unique()

    if adjs_u.size == 0:
        print("🛑", co, line, adjs)
        return pd.Series([False, True, "*".join(adjs.map(lambda x: x[0]).unique())])

    g = nx.Graph()
    g.add_edges_from(adjs_u)
    # 閉路検出
    try:
        nx.find_cycle(g)
        cycled = True
    except nx.NetworkXNoCycle:
        cycled = False
    # 連結判定
    connected = nx.is_connected(g)

    all_paths = []
    # 元のグラフを変更しないように、連結成分ごとにコピーを作るの
    try:
        component_graphs = [g.subgraph(c).copy() for c in nx.connected_components(g)]
    except nx.NetworkXError:
        # G が空だったりするとエラーになるからね
        return []

    if cycled or (not connected):
        plt.figure(figsize=(16, 16))
        nx.draw(g, with_labels=True, node_color="white", node_size=2000, font_family=plt.rcParams["font.family"][0], edge_color="gray", linewidths=1, edgecolors="black", font_size=10)
        plt.savefig(image_path / f"{co}_{line}")
        plt.clf()
    # print(co, line, cycled, len(component_graphs), g.number_of_nodes(), g.number_of_edges())

    for comp_graph in component_graphs:
        # この連結成分で、辺がなくなるまで繰り返すよ
        while comp_graph.number_of_edges() > 0:

            # パスを削除すると非連結になるかもしれないから、
            #「現在の」連結してる部分グラフごとに処理するんだよ
            current_sub_components = [
                comp_graph.subgraph(sc) for sc in nx.connected_components(comp_graph)
                if comp_graph.subgraph(sc).number_of_edges() > 0
            ]

            if not current_sub_components:
                # もう処理する辺がないみたい
                break

            edges_to_remove = []
            path_found_in_iteration = False # 無限ループ防止用

            for sub_comp in current_sub_components:
                # ノードが1個以下じゃパスは作れないでしょ
                if sub_comp.number_of_nodes() < 2:
                    continue

                try:
                    # 2. 直径となるパスを得る
                    diameter = nx.diameter(sub_comp)
                    ecc = nx.eccentricity(sub_comp) # 各ノードの離心率

                    diameter_path_nodes = []
                    found_path = False

                    # 直径長の離心率を持つノード u を探す
                    for u, e_u in ecc.items():
                        if e_u == diameter:
                            # u から距離が直径長になるノード v を探す
                            lengths = nx.shortest_path_length(sub_comp, source=u)
                            for v, length in lengths.items():
                                if length == diameter:
                                    # u-v 間の最短パスが直径パスの1つ
                                    diameter_path_nodes = nx.shortest_path(sub_comp, source=u, target=v)
                                    found_path = True
                                    break
                        if found_path:
                            break

                    if diameter_path_nodes:
                        # 4. パスのリストに追加
                        all_paths.append(diameter_path_nodes)

                        # 3. そのパスを削除 (削除する辺をマーク)
                        path_edges = list(zip(diameter_path_nodes[:-1], diameter_path_nodes[1:]))
                        edges_to_remove.extend(path_edges)
                        path_found_in_iteration = True

                except nx.NetworkXError:
                    # sub_comp が空になったり、孤立ノードだけになったりすると
                    # diameter や eccentricity が計算できなくてエラーになるから無視するの
                    pass

            if not path_found_in_iteration:
                # このイテレーションで1つもパスが見つからなかったら、もう終わり
                break

            # マークした辺をまとめて削除
            comp_graph.remove_edges_from(edges_to_remove)

    return pd.Series([cycled, connected] + [",".join(path) for path in all_paths])

sta_graph = rr_adjs_man.groupby(["co", "line"], as_index=False)
sta_graph = pd.concat([sta_graph.size().drop(columns="size"), sta_graph.size().apply(app_func, axis=1)], axis=1).rename(columns={0: "cycled", 1: "connected"})
sta_graph.to_csv("sta_graph.csv", sep="\t")
sta_graph

"""# LineString を駅順に並び替え"""

def sorter_factory(paths_list):
    """
    パスのリストを受け取って、エッジとノードが混ざったリストを
    ノードの位置も考慮してソートするための関数
    """

    # 辺の順序と、ノードの順序を別々に管理する
    edge_to_order = {}
    node_to_order = {}

    # 0, 1, 2, ... といった整数のインデックスはエッジに割り当てる
    order_index = 0

    for path in paths_list:
        # パスの最初のノードは、最初のエッジ（インデックス 0）のさらに前」から -0.5 番
        first_node = path[0]
        if first_node not in node_to_order:
            node_to_order[first_node] = order_index - 0.5

        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]

            # エッジ自体の順番は整数
            edge_key = tuple(sorted((u, v)))
            if edge_key not in edge_to_order:
                edge_to_order[edge_key] = order_index

            # 2番目以降のノード（v）は、前のエッジ（order_index）と
            # 次のエッジ（order_index + 1）の間だから、order_index + 0.5 番
            if v not in node_to_order:
                node_to_order[v] = order_index + 0.5

            order_index += 1

    # key に渡す関数
    def sorter_key(adjs_s):

        def get_order_value(adjs):
            # 要素数が 2 ならエッジ
            if len(adjs) == 2:
                return edge_to_order[adjs]
                # return edge_to_order.get(edge_key, float("inf"))
            # 要素数が 1 ならノード
            elif len(adjs) == 1:
                node_key = adjs[0]
                return node_to_order[node_key]
                # return node_to_order.get(node_key, float("inf"))

        return adjs_s.apply(get_order_value)

    return sorter_key


rr_sorted = []
for row in sta_graph.itertuples(index=False, name=None):
    df = rr_adjs_man.query('(co == @row[0]) and (line == @row[1])')
    adjs_sorter = sorter_factory([path.split(",") for path in row[4:] if type(path) is str])
    sorted_df = df.sort_values(by="adj_stas", key=adjs_sorter)
    rr_sorted.append(sorted_df)
rr_sorted = pd.concat(rr_sorted)
rr_sorted["line_id"] = rr_sorted.groupby(["co", "line"]).ngroup() + 1
rr_sorted["seq_id"] = rr_sorted.groupby(["co", "line"]).cumcount() + 1
rr_sorted.set_index(["line_id", "seq_id"], inplace=True)
rr_sorted.drop(columns=["id"], inplace=True)
rr_sorted

"""# 路線名の通称を追加

## line が一意かどうか
"""

line_counts = sta_graph["line"].value_counts()
ununique_line = set(line_counts[line_counts > 1].index)
line_counts[line_counts > 1]

"""## まずはおおまかに"""

rr_sorted.drop(columns="geometry")

# 一応更新順は (co, line) → co → line
co_aliases = {
    "東京地下鉄": "東京メトロ",
    "九州旅客鉄道": "JR九州",
    "北海道旅客鉄道": "JR北海道",
    "四国旅客鉄道": "JR四国",
    "東日本旅客鉄道": "JR東日本",
    "東海旅客鉄道": "JR東海",
    "西日本旅客鉄道": "JR西日本",
    "東海交通事業": "JR東海交通事業",
    "東京都": "東京都交通局",
    "名古屋市": "名古屋市交通局",
    "大阪市高速電気軌道": "Osaka Metro",
    "横浜市": "横浜市交通局",
    "福岡市": "福岡市交通局",
    "函館市": "函館市企業局交通部",
    "山陽電気鉄道": "山陽電車",
    "長崎電気軌道": "長崎電軌",
    "阪神電気鉄道": "阪神電車",
    "京阪電気鉄道": "京阪電車",
    "南海電気鉄道": "南海電鉄",
    "阪神電気鉄道": "阪神電車",
    "仙台市": "仙台市交通局",
    "札幌市": "札幌市交通局",
    "京都市": "京都市交通局",
    "神戸市": "神戸市交通局",
    "鹿児島市": "鹿児島市交通局",
    "熊本市": "熊本市交通局",
    "福岡市": "福岡市交通局",
    "京浜急行電鉄": "京急電鉄",
}
line_aliases = {
    # 東京メトロ
    "11号線半蔵門線": "半蔵門線",
    "13号線副都心線": "副都心線",
    "2号線日比谷線": "日比谷線",
    "3号線銀座線": "銀座線",
    "4号線丸ノ内線": "丸ノ内線",
    "5号線東西線": "東京メトロ東西線",
    "7号線南北線": "東京メトロ南北線",
    "8号線有楽町線": "有楽町線",
    "9号線千代田線": "千代田線",
    # 東京都交通局
    "10号線新宿線": "新宿線",
    "12号線大江戸線": "大江戸線",
    "1号線浅草線": "浅草線",
    "6号線三田線": "三田線",
    "上野懸垂線": "上野動物園モノレール",
    "東京臨海新交通臨海線": "ゆりかもめ",
    # 名古屋市交通局
    "1号線東山線": "東山線",
    "2号線名城線": "名城線",
    "2号線名港線": "名港線",
    "3号線鶴舞線": "鶴舞線",
    "4号線名城線": "名城線",
    "6号線桜通線": "桜通線",
    # Osaka Metro
    "1号線(御堂筋線)": "御堂筋線",
    "2号線(谷町線)": "谷町線",
    "3号線(四つ橋線)": "四つ橋線",
    "4号線(中央線)": "中央線",
    "5号線(千日前線)": "千日前線",
    "6号線(堺筋線)": "堺筋線",
    "7号線(長堀鶴見緑地線)": "長堀鶴見緑地線",
    "8号線（今里筋線）": "今里筋線",
    "南港ポートタウン線": "ニュートラム",
    # 広島高速交通
    "広島新交通1号線": "アストラムライン",
    # 福岡市交通局
    "1号線(空港線)": "福岡市地下鉄空港線",
    "2号線(箱崎線)": "箱崎線",
    "3号線(七隈線)": "七隈線",
    # 立山黒部貫光
    "無軌条電車線": "立山トンネルトロリーバス",
}
# co_alias → service
services = {
    "横浜市交通局": "横浜市営地下鉄",
    "函館市企業局交通部": "函館市電",
    "一般社団法人札幌市交通事業振興公社": "札幌市電",
    "仙台市交通局": "仙台市地下鉄",
    "札幌市交通局": "札幌市営地下鉄",
    "名古屋市交通局": "名古屋市営地下鉄",
    "京都市交通局": "京都市営地下鉄",
    "神戸市交通局": "神戸市営地下鉄",
    "鹿児島市交通局": "鹿児島市電",
    "熊本市交通局": "熊本市電",
    "福岡市交通局": "福岡市地下鉄"
}
# (co, line) → (service, line_alias)
service_line_aliases = {
    ("東京都", "10号線新宿線"): ("都営地下鉄", "新宿線"),
    ("東京都", "12号線大江戸線"): ("都営地下鉄", "大江戸線"),
    ("東京都", "1号線浅草線"): ("都営地下鉄", "浅草線"),
    ("東京都", "6号線三田線"): ("都営地下鉄", "三田線"),
    ("東京都", "荒川線"): ("東京都電車", "荒川線"),
    ("横浜市", "1号線"): ("横浜市営地下鉄", "ブルーライン"),
    ("横浜市", "3号線"): ("横浜市営地下鉄", "ブルーライン"),
    ("横浜市", "4号線"): ("横浜市営地下鉄", "グリーンライン"),
    ("京成電鉄", "本線"): ("京成電鉄", "京成本線"),
    ("京浜急行電鉄", "本線"): ("京急電鉄", "京急本線"),
    ("富山地方鉄道", "本線"): ("富山地方鉄道", "富山地方鉄道本線"),
    ("山陽電気鉄道", "本線"): ("山陽電車", "山陽電車本線"),
    ("広島電鉄", "本線"): ("広島電鉄", "広島電鉄本線"),
    ("相模鉄道", "本線"): ("相模鉄道", "相鉄本線"),
    ("近江鉄道", "本線"): ("近江鉄道", "近江鉄道本線"),
    ("長崎電気軌道", "本線"): ("長崎電軌", "長崎電軌本線"),
    ("阪神電気鉄道", "本線"): ("阪神電車", "阪神本線"),
    ("黒部峡谷鉄道", "本線"): ("黒部峡谷鉄道", "黒部峡谷鉄道本線"),
    ("京福電気鉄道", "鋼索線"): ("京福電気鉄道", "京福電気鉄道鋼索線"),
    ("京阪電気鉄道", "鋼索線"): ("京阪電車", "京阪鋼索線"),
    ("南海電気鉄道", "鋼索線"): ("南海電鉄", "南海鋼索線"),
    ("箱根登山鉄道", "鋼索線"): ("箱根登山鉄道", "箱根登山鉄道鋼索線"),
    ("能勢電鉄", "鋼索線"): ("能勢電鉄", "妙見の森ケーブル"),
    ("神戸電鉄", "神戸高速線"): ("神戸電鉄", "神戸電鉄神戸高速線"),
    ("阪急電鉄", "神戸高速線"): ("阪急電鉄", "阪急神戸高速線"),
    ("阪神電気鉄道", "神戸高速線"): ("阪神電車", "阪神神戸高速線"),
    ("仙台市", "南北線"): ("仙台市地下鉄", "仙台市地下鉄南北線"),
    ("仙台市", "東西線"): ("仙台市地下鉄", "仙台市地下鉄東西線"),
    ("北大阪急行電鉄", "南北線"): ("北大阪急行電鉄", "北大阪急行電鉄南北線"),
    ("札幌市", "南北線"): ("札幌市営地下鉄", "札幌市営地下鉄南北線"),
    ("札幌市", "東西線"): ("札幌市営地下鉄", "札幌市営地下鉄東西線"),
    ("京浜急行電鉄", "空港線"): ("京急電鉄", "京急空港線"),
    ("南海電気鉄道", "空港線"): ("南海電鉄", "南海空港線"),
    ("名古屋鉄道", "空港線"): ("名古屋鉄道", "名鉄空港線"),
    ("京都市", "東西線"): ("京都市営地下鉄", "京都市営地下鉄東西線"),
    ("甘木鉄道", "甘木線"): ("甘木鉄道", "甘木鉄道甘木線"),
    ("西日本鉄道", "甘木線"): ("西日本鉄道", "西鉄甘木線"),
    ("東日本旅客鉄道", "山手線"): ("JR東日本", "JR山手線"),
    ("神戸市", "山手線"): ("神戸市営地下鉄", "神戸市営地下鉄山手線"),
    ("東日本旅客鉄道", "山田線"): ("JR東日本", "JR山田線"),
    ("近畿日本鉄道", "山田線"): ("近畿日本鉄道", "近鉄山田線"),
    ("近畿日本鉄道", "京都線"): ("近畿日本鉄道", "近鉄京都線"),
    ("阪急電鉄", "京都線"): ("阪急電鉄", "阪急京都線"),
    ("伊予鉄道", "城北線"): ("伊予鉄道", "伊予鉄道城北線"),
    ("東海交通事業", "城北線"): ("JR東海交通事業", "JR東海交通事業城北線"),
    ("東日本旅客鉄道", "日光線"): ("JR東日本", "JR日光線"),
    ("東武鉄道", "日光線"): ("東武鉄道", "東武日光線"),
    ("西日本旅客鉄道", "奈良線"): ("JR西日本", "JR奈良線"),
    ("近畿日本鉄道", "奈良線"): ("近畿日本鉄道", "近鉄奈良線"),
    ("近畿日本鉄道", "長野線"): ("近畿日本鉄道", "近鉄長野線"),
    ("長野電鉄", "長野線"): ("長野電鉄", "長野電鉄長野線"),
    ("西日本旅客鉄道", "山口線"): ("JR西日本", "JR山口線"),
    ("西武鉄道", "山口線"): ("西武鉄道", "西武山口線"),
    ("大阪市高速電気軌道", "4号線(中央線)"): ("Osaka Metro", "Osaka Metro 中央線"),
    ("のと鉄道", "七尾線"): ("のと鉄道", "のと鉄道七尾線"),
    ("西日本旅客鉄道", "七尾線"): ("JR西日本", "JR七尾線"),
    ("京浜急行電鉄", "大師線"): ("京急電鉄", "京急大師線"),
    ("東武鉄道", "大師線"): ("東武鉄道", "東武大師線"),
    ("箱根登山鉄道", "鉄道線"): ("箱根登山鉄道", "箱根登山鉄道鉄道線"),
    ("遠州鉄道", "鉄道線"): ("遠州鉄道", "遠州鉄道鉄道線"),
    ("千葉都市モノレール", "1号線"): ("千葉都市モノレール", "千葉都市モノレール1号線"),
}

def app_func(row):
    co, service, line, line_alias = row["co"], row["service"], row["line"], row["line_alias"]
    if (co, line) in service_line_aliases.keys():
        service, line_alias = service_line_aliases[(co, line)]
        if service != row["service"]:
            print("🛑", service, row["service"], line)
    return pd.Series([service, line_alias])

# 一応更新順は co → service → line → (service, line)
rr_sorted["co_alias"] = rr_sorted["co"].replace(co_aliases)
rr_sorted["service"] = rr_sorted["co_alias"].replace(services)
rr_sorted["line_alias"] = rr_sorted["line"].replace(line_aliases)
rr_sorted[["service", "line_alias"]] = rr_sorted.apply(app_func, axis=1)

for line in line_counts[line_counts > 1].index:
    tmp_df = sta_graph[sta_graph["line"].str.contains(line)]

    for row in tmp_df.itertuples():
        if row.line in ununique_line:
            print("🛑", end=" ")
        print(f'("{row.co}", "{row.line}")')
    print()

"""## 手動で追加するために `.shp` ファイルと CSV に出力"""

rr_sorted_dir = Path("rr_sorted")
rr_sorted_dir.mkdir(exist_ok=True)

rr_sorted_out = rr_sorted.copy()
rr_sorted_out["adj_stas"] = rr_sorted_out["adj_stas"].map(lambda x: ",".join(x))
rr_sorted_out.to_file(rr_sorted_dir / "rr_sorted.shp", encoding="utf-8")
shutil.make_archive("rr_sorted", format="zip", root_dir=".", base_dir=rr_sorted_dir)

pd.DataFrame(rr_sorted_out.drop(columns="geometry")).to_csv("rr_sorted.tsv", sep="\t")

"""## 補助でグラフ出力"""

co_line_str = "東日本旅客鉄道\t東海道線" #@param {type:"string"}
def show_sta_graph(co="", line=""):
    g = nx.Graph()
    if line == "":
        df = rr_sorted.query('(co == @co)')
        df = df[df["adj_stas"].map(len) == 2]

        g.add_edges_from(df["adj_stas"])
        for row in df.itertuples():
            g.edges[*row.adj_stas]["line"] = row.line

        cmap = plt.get_cmap("tab10")
        color_map = {line: cmap(i) for i, line in enumerate(df["line"].unique())}
        edge_colors = [color_map[g.edges[e]["line"]] for e in g.edges()]
    elif co == "":
        df = rr_sorted.query('(line == @line)')
        df = df[df["adj_stas"].map(len) == 2]

        g.add_edges_from(df["adj_stas"])
        for row in df.itertuples():
            g.edges[*row.adj_stas]["co"] = row.co

        cmap = plt.get_cmap("tab10")
        color_map = {co: cmap(i) for i, co in enumerate(df["co"].unique())}
        edge_colors = [color_map[g.edges[e]["co"]] for e in g.edges()]
    else:
        df = rr_sorted.query('(co == @co) and (line == @line)')
        df = df[df["adj_stas"].map(len) == 2]
        g.add_edges_from(df["adj_stas"])
        edge_colors = "black"

    plt.figure(figsize=(16, 16))
    nx.draw(g, with_labels=True, node_color="white", node_size=2000, font_family=plt.rcParams["font.family"][0], width=3, edge_color=edge_colors, linewidths=1, edgecolors="gray", font_size=10)

show_sta_graph(*co_line_str.split("\t"))

"""# 地図アプリを形にするため仮の GeoJSON を出力"""

rr_sorted.crs
rr_sorted.to_crs("EPSG:4326").to_file("rr.geojson", driver="GeoJSON")

sta_pt_man.to_crs("EPSG:4326").to_file("sta.geojson", driver="GeoJSON")

"""# その他"""

print(1 if frozenset((1,2)) in [frozenset((2,1,))] else 0)
print(set([frozenset((1,2)), frozenset((2,1))]))
frozen_seg = frozenset({(1,2), (3,4)})
print(*frozen_seg)
print({x for l in ([1]*3, [4]*4, [2]*5) for x in l})
print([s for (s, e) in frozen_seg])

"""# NetworkX による全点調査
---

NetworkX で路線ごとに全セグメントからグラフを作成し、その路線内の全点について次数を列として集計。

---
"""

l = []
for co, line_name in lines:
    ggraph = gen_graph_from_gs_ls(railroad_split_by_stations.loc[co, line_name]["geometry"])
    l.append(
        pd.Series([ggraph.degree[v] for v in ggraph]).value_counts().sort_index().rename(f"{co} {line_name}")
    )
degree_count_df = pd.DataFrame(l).fillna(0).astype(int)

degree_count_df.sort_values(1, ascending=False).head(10)

"""# 保存用"""

# def coords_from_mls(mls: MultiLineString):
#     return list(
#         {coord for ls in mls.geoms for coord in ls.coods}
#     )

# 総当たりは時間がかかりすぎるので却下
# def get_max_route(combis, segments, lengths):
#     universe = [(*seg, lengths[seg]) for seg in segments]
#     gs.set_universe(universe)

#     max_len = 0
#     for i, pair in enumerate(combis):
#         print("pair", str(i).rjust(15, " "), pair)
#         paths = gs.paths(*pair)
#         if len(paths):
#             tmp_path = next(paths.max_iter())
#             tmp_len = sum([lengths[seg] for seg in tmp_path])
#             if max_len < tmp_len:
#                 max_len = tmp_len
#                 max_route = tmp_path
#         else:
#             print("No paths")

#     print("Start cycles")
#     cycles = gs.cycles()
#     if len(cycles):
#         tmp_cycle = next(cycles.max_iter())
#         tmp_len = sum([lengths[seg] for seg in tmp_cycle])
#         if max_len < tmp_len:
#             # max_len = tmp_len
#             max_route = tmp_cycle
#     print("Finish cycles")

#     return max_route

# これも使わない
# def get_longest_branches(m_line_str) -> list[LineString]:
#     coords = get_all_coords(m_line_str)
#     combinations = itertools.combinations(coords, 2)
#     segments = get_all_segments(m_line_str)
#     print("coords:", len(coords))
#     print("segments:", len(segments))
#     lengths = {segments[i]: length for i, length in enumerate(get_lengths(segments))}

#     branches = []
#     rest_segments = segments.copy()
#     while len(rest_segments):
#         print("Rest segments:", len(rest_segments))
#         route = get_max_route(combinations, rest_segments, lengths)
#         rest_segments = [seg for seg in rest_segments if seg not in route]
#         branches.append(LineString(route))
#     return branches

# これも使わない
# def merge_linestring(gdf_line):
#     rows = []
#     for row in gdf_line.dissolve(by=["N02_004", "N02_003"]).to_dict(orient="records"):
#         for line_str in get_longest_branches(row["geometry"]):
#             row["geometry"] = line_str
#             rows.append(row.copy())
#     return pd.DataFrame.from_dict(rows, orient="records")

# def snap_edge(
#     row: gpd.GeoSeries,
#     gs_pt: gpd.GeoSeries,
#     tolerance: float
# ):
#     xy_new = list(row["geometry"].coords)
#     # ラインの終点[-1]と始点[0]を処理
#     for i in [-1, 0]:
#         edge = Point(row["geometry"].coords[i])
#         gs_distance = gs_pt.apply(pt_dist_sq, p2=edge)
#         # ポイントから許容誤差範囲内の端点に対して，端点の座標をポイントの座標に差し替え
#         if gs_distance.min() < np.power(tolerance, 2):
#             pt_new = gs_pt[gs_distance.idxmin()]
#             xy_new[i] = list(pt_new.coords)[0]
#     return LineString(xy_new)

# def split_line_at_point(
#     gdf_line: gpd.GeoDataFrame,
#     gdf_pt: gpd.GeoDataFrame,
#     tolerance: float
# ) -> gpd.GeoDataFrame:
#     # ポイントから許容誤差でバッファ
#     gdf_pt["_buffer_poly"] = gdf_pt["geometry"].apply(cir_buffer, r=0.9 * tolerance)
#     # ラインをバッファでくり抜き
#     gdf_line = gpd.overlay(
#         df1=gdf_line,
#         df2=gdf_pt[["_buffer_poly"]].set_geometry("_buffer_poly"),
#         how="symmetric_difference",
#         keep_geom_type=True
#     )
#     gdf_line = gdf_line.explode(index_parts=False)
#     # 端点をポイントにスナップ
#     gdf_line["geometry"] = gdf_line.apply(
#         snap_edge,
#         axis=1,
#         gs_pt=gdf_pt["geometry"],
#         tolerance=tolerance
#     )
#     return gdf_line