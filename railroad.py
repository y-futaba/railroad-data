"""鉄道データ（国土数値情報 N02）のステップ実行用インタラクティブスクリプト。

VS Code Interactive Window 等で `# %%` セル単位でステップ実行するか、
コマンドラインから `python -m railroad_data phase1` / `phase2` を実行する。
"""

# %% [markdown]
# # 設定とモジュールの読み込み

# %%
import geopandas as gpd

from src.domain import DEFAULT_CONFIG
from src.pipelines import (
    assign_adjacent_stations_to_lines,
    assign_aliases_to_sorted_railroads,
    assign_modification_status,
    build_irregular_station_sets_df,
    build_station_graph_table,
    compare_station_points,
    complement_missing_station_segments,
    compute_station_degree_and_set_counts,
    count_lines_per_route,
    export_final_outputs,
    extract_irregular_lines,
    extract_line_keys,
    find_duplicate_ids,
    find_duplicate_line_names,
    find_excluded_stations,
    find_invalid_adj_stas_rows,
    find_isolated_line_coords,
    find_mismatched_co_line_rows,
    load_and_compute_station_points,
    load_and_normalize_adj_stas_gdf,
    load_raw_railroads,
    merge_railroad_lines,
    show_sta_graph,
    sort_railroads_by_station_order,
    split_all_lines_by_stations,
)

config = DEFAULT_CONFIG

# %% [markdown]
# # Phase 1: 駅で路線切断
# ## 1. 路線データの読み込みと路線ごとのライン結合

# %%
railroads = load_raw_railroads(config)
merged_railroads = merge_railroad_lines(railroads)
merged_railroads

# %% [markdown]
# ## 2. 駅の代表点の算出と保存

# %%
stations_ls, stations = load_and_compute_station_points(config, save_file=True)
stations

# %% [markdown]
# ## 3. 駅点で路線を切断

# %%
lines = extract_line_keys(merged_railroads)
railroad_split_by_stations = split_all_lines_by_stations(
    merged_railroads,
    stations,
    lines,
    tolerance=config.snap_tolerance,
    buffer_ratio=config.buffer_ratio,
)
railroad_split_by_stations

# %% [markdown]
# ## 4. 駅の線データと被る部分に抜けがないか確認

# %%
excluded_stas = find_excluded_stations(
    railroad_split_by_stations,
    stations,
    lines,
)
excluded_stas

# %% [markdown]
# ## 5. 欠落している駅の線分を補完して再度結合・駅点切断

# %%
re_railroads = complement_missing_station_segments(
    railroads,
    stations_ls,
    excluded_stas,
)
re_merged_railroads = merge_railroad_lines(re_railroads)
re_railroad_split_by_stations = split_all_lines_by_stations(
    re_merged_railroads,
    stations,
    lines,
    tolerance=config.snap_tolerance,
    buffer_ratio=config.buffer_ratio,
)
re_excluded_stas = find_excluded_stations(
    re_railroad_split_by_stations,
    stations,
    lines,
)
re_excluded_stas

# %% [markdown]
# # 隣接駅の調査・追加
# ## 6. 駅に繋がっている線の数別・隣接駅集合の要素数別カウント

# %%
sta_degree_count_df, adj_stas_ele_count_df = compute_station_degree_and_set_counts(
    re_railroad_split_by_stations,
    stations,
    lines,
)
sta_degree_count_df

# %%
adj_stas_ele_count_df

# %% [markdown]
# ## 7. 駅同士の接続がおかしい路線とその駅の抽出・CSV出力

# %%
irregular_lines = extract_irregular_lines(adj_stas_ele_count_df)
irregular_lines

# %%
irregular_sta_set_df = build_irregular_station_sets_df(
    irregular_lines,
    re_railroad_split_by_stations,
    stations,
)
irregular_sta_set_df.to_csv(config.irregular_sta_set_csv_path)
irregular_sta_set_df

# %% [markdown]
# ## 8. 接続している駅が 0 個（孤立）の頂点座標

# %%
isolated_coords = find_isolated_line_coords(
    re_railroad_split_by_stations,
    stations,
    lines,
)
isolated_coords

# %% [markdown]
# ## 9. 線の GeoDataFrame の属性に隣接駅を追加・出力、および路線ごとの行数確認

# %%
railroads_with_adj_stas = assign_adjacent_stations_to_lines(
    re_railroad_split_by_stations,
    stations,
    lines,
)
railroads_with_adj_stas.to_file(  # pyright: ignore[reportUnknownMemberType]
    config.railroad_adj_stas_path,
    encoding=config.encoding,
)
railroads_with_adj_stas

# %%
line_counts_df = count_lines_per_route(railroads_with_adj_stas, lines)
line_counts_df

# %% [markdown]
# ---
# # Phase 2: GIS で隣接駅を編集した `.shp` ファイルの検証・ソート・最終出力
# ## 10. `adj_stas` の要素数確認・ID重複確認・`(co, line)` 不整合確認

# %%
rr_adjs_man = load_and_normalize_adj_stas_gdf(
    config.railroad_adj_stas_manual_path,
    encoding=config.encoding,
)
railroads_with_adj_stas_loaded = load_and_normalize_adj_stas_gdf(
    config.railroad_adj_stas_path,
    encoding=config.encoding,
)
invalid_adj_stas_df = find_invalid_adj_stas_rows(rr_adjs_man)
invalid_adj_stas_df

# %%
duplicate_ids = find_duplicate_ids(rr_adjs_man)
duplicate_ids

# %%
mismatched_co_line_df = find_mismatched_co_line_rows(
    rr_adjs_man,
    railroads_with_adj_stas_loaded,
)
mismatched_co_line_df

# %% [markdown]
# ## 11. 路線の変更 `status` 追加と駅データの編集前後比較

# %%
rr_adjs_man = assign_modification_status(
    rr_adjs_man,
    railroads_with_adj_stas_loaded,
)
status_counts = rr_adjs_man.groupby("status").size()  # pyright: ignore[reportUnknownMemberType]
status_counts

# %%
sta_pt = gpd.read_file(  # pyright: ignore[reportUnknownMemberType]
    config.station_pt_path,
    encoding=config.encoding,
)
sta_pt_man = gpd.read_file(  # pyright: ignore[reportUnknownMemberType]
    config.station_pt_manual_path,
    encoding=config.encoding,
)
added_stations_df, removed_stations_df = compare_station_points(
    sta_pt,
    sta_pt_man,
)
added_stations_df

# %%
removed_stations_df

# %% [markdown]
# ## 12. 駅同士の接続をグラフで表現・直径パス分解

# %%
sta_graph, summaries = build_station_graph_table(
    rr_adjs_man,
    image_dir_path=config.image_dir_path,
)
sta_graph.to_csv(config.sta_graph_csv_path, sep="\t")
sta_graph

# %% [markdown]
# ## 13. `LineString` を駅順に並び替え

# %%
rr_sorted_base = sort_railroads_by_station_order(rr_adjs_man, summaries)
rr_sorted_base

# %% [markdown]
# ## 14. 路線名の重複確認と通称（`co_alias`, `service`, `line_alias`）の付与

# %%
duplicate_line_names = find_duplicate_line_names(sta_graph)
duplicate_line_names

# %%
rr_sorted = assign_aliases_to_sorted_railroads(rr_sorted_base)
rr_sorted.drop(columns="geometry")

# %% [markdown]
# ## 15. 最終出力（Shapefile ZIP / TSV / GeoJSON）

# %%
export_final_outputs(rr_sorted, sta_pt_man, config)

# %% [markdown]
# ## 16. 補助でグラフ出力

# %%
co_line_str = "東日本旅客鉄道\t東海道線"
co_arg, line_arg = co_line_str.split("\t", maxsplit=1)
show_sta_graph(rr_sorted, co=co_arg, line=line_arg)
