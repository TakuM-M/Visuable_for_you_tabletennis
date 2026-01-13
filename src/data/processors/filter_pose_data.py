"""
特定のトラッキングIDのみを抽出するスクリプト
pose_data.csvから指定したtrack_idのデータだけを抽出して新しいCSVを作成
"""
import pandas as pd
import argparse
from pathlib import Path
import sys


def filter_by_track_id(input_csv: str, output_csv: str, track_ids: list[int]):
    """
    CSVから特定のtrack_idのデータのみを抽出

    Args:
        input_csv: 入力CSVファイルパス
        output_csv: 出力CSVファイルパス
        track_ids: 抽出したいtrack_idのリスト
    """
    # CSVを読み込み
    print(f"CSVを読み込んでいます: {input_csv}")
    df = pd.read_csv(input_csv)

    print(f"  総レコード数: {len(df)}")
    print(f"  検出されたtrack_id: {sorted(df['track_id'].unique().tolist())}")

    # 指定されたtrack_idでフィルタリング
    df_filtered = df[df['track_id'].isin(track_ids)]

    print(f"\nフィルタリング結果:")
    print(f"  対象track_id: {track_ids}")
    print(f"  抽出されたレコード数: {len(df_filtered)}")

    # track_id別の統計
    for track_id in track_ids:
        count = len(df_filtered[df_filtered['track_id'] == track_id])
        if count > 0:
            print(f"    track_id={track_id}: {count}レコード")
        else:
            print(f"    track_id={track_id}: データなし（警告）")

    # 結果を保存
    df_filtered.to_csv(output_csv, index=False)
    print(f"\n保存完了: {output_csv}")


def show_statistics(csv_path: str):
    """
    CSVファイルの統計情報を表示

    Args:
        csv_path: CSVファイルパス
    """
    print(f"CSVファイルの統計情報: {csv_path}\n")

    df = pd.read_csv(csv_path)

    # 基本統計
    print(f"総レコード数: {len(df)}")
    print(f"総フレーム数: {df['frame'].nunique()}")

    # track_id別の統計
    track_ids = sorted(df['track_id'].unique().tolist())
    print(f"\n検出されたtrack_id: {track_ids}")
    print(f"\ntrack_id別の詳細:")

    for track_id in track_ids:
        df_track = df[df['track_id'] == track_id]
        print(f"\n  track_id={track_id}:")
        print(f"    レコード数: {len(df_track)}")
        print(f"    フレーム範囲: {df_track['frame'].min()} - {df_track['frame'].max()}")
        print(f"    平均信頼度: {df_track['confidence'].mean():.3f}")

        # 位置の統計
        center_x = (df_track['bbox_x1'] + df_track['bbox_x2']) / 2
        center_y = (df_track['bbox_y1'] + df_track['bbox_y2']) / 2
        print(f"    平均位置: ({center_x.mean():.1f}, {center_y.mean():.1f})")


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
