"""複数のtrack_idが含まれるpose_data.csvから特定のtrack_idのデータのみを抽出するスクリプト

    Args:
        input_csv: 入力CSVファイルパス(既に検知し正規化していることとする．)
        output_csv: 出力CSVファイルパス
        track_ids: 抽出したいtrack_idのリスト
    
    Returns:
        抽出されたtrack_idのデータを含むCSVファイル
"""
import pandas as pd
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.dataset.annotation.processors.filter_by_track_id import filter_by_track_id, show_statistics

def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description='pose_data.csvから特定のtrack_idのデータを抽出',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
--------
# 統計情報を表示
python src/data/processors/filter_pose_data.py -i output/pose_data.csv --stats

# track_id=3のデータのみを抽出
python src/data/processors/filter_pose_data.py -i output/pose_data.csv -o output/player3.csv --ids 3

# track_id=3と5のデータを抽出(カンマ区切り)
python src/data/processors/filter_pose_data.py -i data/detect/sample_video_01_03/01_all_playser_pose_data.csv -o data/detect/sample_video_01_03/players.csv --ids 1,3
        """
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
        help='出力CSVファイルパス（--idsと一緒に指定）'
    )
    parser.add_argument(
        '--ids',
        type=str,
        nargs='+',
        help='抽出するtrack_idのリスト（スペース区切りまたはカンマ区切り）例: 3 5 または "3,5"'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='統計情報のみを表示（フィルタリングは行わない）'
    )

    args = parser.parse_args()

    # 入力ファイルの存在確認
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"エラー: 入力ファイルが見つかりません: {input_path}")
        sys.exit(1)

    # 統計情報のみを表示
    if args.stats:
        show_statistics(args.input)
        return

    # フィルタリング処理
    if not args.ids or not args.output:
        print("エラー: --ids と -o の両方を指定してください")
        print("または --stats で統計情報を表示できます")
        parser.print_help()
        sys.exit(1)

    # track_idリストをパース
    track_ids = []
    for id_str in args.ids:
        if ',' in id_str:
            # カンマ区切りの場合
            track_ids.extend([int(x.strip()) for x in id_str.split(',')])
        else:
            # スペース区切りの場合
            track_ids.append(int(id_str.strip()))

    # 出力ディレクトリを作成
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # フィルタリング実行
    filter_by_track_id(args.input, args.output, track_ids)


if __name__ == "__main__":
    main()
