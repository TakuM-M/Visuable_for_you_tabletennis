"""複数のtrack_idが含まれるpose_data.csvから特定のtrack_idのデータのみを抽出し、
プレイヤーA/Bに集約するスクリプト

    Args:
        input_csv: 入力CSVファイルパス(既に検知し正規化していることとする．)
        output_csv: 出力CSVファイルパス
        player_a_ids: プレイヤーAのtrack_idリスト (ID 1に集約)
        player_b_ids: プレイヤーBのtrack_idリスト (ID 2に集約)

    Returns:
        抽出・集約されたデータを含むCSVファイル

Usage:
python -m src.dataset.annotation.filter_pose_data \
    -i data/detect/sample_video_01_02/all_players_pose_data.csv \
    -o data/detect/sample_video_01_02/merged_pose_data.csv \
    --player-a 1,18,21,27,50,56,64,91,112,124,137 \
    --player-b 6,33,43,97,111,115,121,130,143,148

"""
import pandas as pd
import argparse
from pathlib import Path
import sys

from src.dataset.annotation.processors import merge_player_ids

def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description='pose_data.csvから特定のtrack_idのデータを抽出し、プレイヤーA/Bに集約',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='入力CSVファイルパス'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='出力CSVファイルパス'
    )
    parser.add_argument(
        '--player-a',
        type=str,
        required=True,
        help='プレイヤーAのtrack_idリスト（カンマ区切り）例: "1,18,21"'
    )
    parser.add_argument(
        '--player-b',
        type=str,
        required=True,
        help='プレイヤーBのtrack_idリスト（カンマ区切り）例: "6,33,43"'
    )

    args = parser.parse_args()

    # 入力ファイルの存在確認
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"エラー: 入力ファイルが見つかりません: {input_path}")
        sys.exit(1)

    # track_idリストをパース
    player_a_ids = [int(x.strip()) for x in args.player_a.split(',') if x.strip()]
    player_b_ids = [int(x.strip()) for x in args.player_b.split(',') if x.strip()]

    print(f"プレイヤーA IDリスト: {player_a_ids}")
    print(f"プレイヤーB IDリスト: {player_b_ids}")

    # 出力ディレクトリを作成
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ID集約実行
    merge_player_ids(args.input, args.output, player_a_ids, player_b_ids)


if __name__ == "__main__":
    main()
