"""コマンドラインから Phase 1・Phase 2・駅接続グラフ描画を実行するエントリポイント。"""

import argparse
from collections.abc import Sequence

import matplotlib.pyplot as plt

from src.domain.config import DEFAULT_CONFIG
from src.pipelines.phase1_prepare import run_phase1
from src.pipelines.phase2_finalize import (
    assign_aliases_to_sorted_railroads,
    assign_modification_status,
    build_station_graph_table,
    load_and_normalize_adj_stas_gdf,
    run_phase2,
    sort_railroads_by_station_order,
)
from src.pipelines.visualize import show_sta_graph


def build_parser() -> argparse.ArgumentParser:
    """CLI 引数パーサを構築する。"""
    parser = argparse.ArgumentParser(
        prog="railroad_data",
        description="国土数値情報（鉄道データ N02）の前処理・トポロジー解析・駅順ソートパイプライン",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p1_parser = subparsers.add_parser(
        "phase1",
        help="Phase 1: 原データの路線結合・駅代表点算出・路線切断・隣接駅検査を実行する",
    )
    p1_parser.add_argument(
        "--no-save",
        action="store_true",
        help="シェープファイルや CSV のファイル書き出しを行わない",
    )

    p2_parser = subparsers.add_parser(
        "phase2",
        help="Phase 2: GIS 手動補正データの検証・駅順ソート・通称付与・最終出力を実行する",
    )
    p2_parser.add_argument(
        "--no-save",
        action="store_true",
        help="最終ファイルの書き出しを行わない",
    )
    p2_parser.add_argument(
        "--save-graph-images",
        action="store_true",
        help="閉路または非連結の路線グラフ画像を保存する",
    )

    plot_parser = subparsers.add_parser(
        "plot",
        help="指定した事業者または路線の駅接続グラフを描画する",
    )
    plot_parser.add_argument(
        "--co",
        default="",
        help="事業者名（例: 東日本旅客鉄道）",
    )
    plot_parser.add_argument(
        "--line",
        default="",
        help="路線名（例: 東海道線）",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI メイン処理を実行する。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    config = DEFAULT_CONFIG

    if args.command == "phase1":
        run_phase1(config, save_outputs=not bool(args.no_save))
    elif args.command == "phase2":
        run_phase2(
            config,
            save_outputs=not bool(args.no_save),
            save_graph_images=bool(args.save_graph_images),
        )
    elif args.command == "plot":
        railroads_with_adj_stas = load_and_normalize_adj_stas_gdf(config.railroad_adj_stas_path, encoding=config.encoding)
        rr_adjs_man = load_and_normalize_adj_stas_gdf(config.railroad_adj_stas_manual_path, encoding=config.encoding)
        rr_with_status = assign_modification_status(rr_adjs_man, railroads_with_adj_stas)
        _, summaries = build_station_graph_table(rr_with_status)
        rr_sorted = assign_aliases_to_sorted_railroads(sort_railroads_by_station_order(rr_with_status, summaries))
        show_sta_graph(rr_sorted, co=str(args.co), line=str(args.line))
        plt.show()  # pyright: ignore[reportUnknownMemberType]
