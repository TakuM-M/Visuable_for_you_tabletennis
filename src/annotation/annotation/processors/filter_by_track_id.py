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