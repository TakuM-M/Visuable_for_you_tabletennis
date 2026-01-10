"""
正規化処理のテストスクリプト（ビデオ表示なし）
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from detection.yolo_tracker import YOLOPoseTracker, PersonTrack
from preprocessing.pose_normalizer import PoseNormalizer
from utils.video_loader import VideoLoader
import csv


def test_normalization():
    """正規化処理をテスト"""

    # 動画ファイルパス
    video_path = "data/raw/sample_video_01_short_version.MOV"
    output_csv = "output/test_normalized/pose_data_normalized.csv"

    print(f"動画ファイル: {video_path}")
    print(f"出力CSV: {output_csv}")
    print("=" * 60)

    # VideoLoaderを使用
    video_loader = VideoLoader(video_path)
    if not video_loader.open():
        print("エラー: 動画を開けませんでした")
        return

    video_info = video_loader.get_info()
    print(f"\n動画情報:")
    print(f"  解像度: {video_info['width']}x{video_info['height']}")
    print(f"  FPS: {video_info['fps']:.2f}")
    print(f"  総フレーム数: {video_info['frame_count']}\n")

    # トラッカーと正規化器を初期化
    tracker = YOLOPoseTracker(conf_threshold=0.3)
    normalizer = PoseNormalizer()

    # データを収集
    normalized_data_list = []
    frame_count = 0
    max_frames = 100  # 最初の100フレームのみテスト

    print(f"処理開始（最大{max_frames}フレーム）...\n")

    try:
        while frame_count < max_frames:
            ret, frame = video_loader.read_frame()
            if not ret:
                break

            frame_count += 1
            timestamp = frame_count / video_info['fps']

            # トラッキング
            persons = tracker.track_frame(frame, persist=True)

            # 各人物について正規化
            for person in persons:
                normalized = normalizer.normalize(person, frame_count, timestamp)
                normalized_data_list.append(normalized.to_dict())

            if frame_count % 10 == 0:
                print(f"  フレーム {frame_count}/{max_frames} 処理済み, 検出人数: {len(persons)}")

    finally:
        video_loader.close()

    print(f"\n処理完了: {frame_count}フレーム処理")
    print(f"総データ数: {len(normalized_data_list)}")

    if len(normalized_data_list) == 0:
        print("警告: データが収集されませんでした")
        return

    # CSVに保存
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = PoseNormalizer.get_csv_fieldnames()

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_data_list)

    print(f"\nCSV保存完了: {output_csv}")

    # 統計情報
    valid_count = sum(1 for d in normalized_data_list if d['is_valid'])
    print(f"\n統計情報:")
    print(f"  正規化成功: {valid_count}/{len(normalized_data_list)} ({valid_count/len(normalized_data_list)*100:.1f}%)")

    # サンプルデータを表示
    if len(normalized_data_list) > 0:
        sample = normalized_data_list[0]
        print(f"\nサンプルデータ（最初のフレーム）:")
        print(f"  フレーム: {sample['frame']}")
        print(f"  トラッキングID: {sample['track_id']}")
        print(f"  腰中心: ({sample['hip_center_x']:.1f}, {sample['hip_center_y']:.1f})")
        print(f"  スケールファクター: {sample['scale_factor']:.2f}px")
        print(f"  正規化された鼻の位置: ({sample['nose_norm_x']:.3f}, {sample['nose_norm_y']:.3f})")

    print("\n" + "=" * 60)
    print("テスト完了")


if __name__ == "__main__":
    test_normalization()
