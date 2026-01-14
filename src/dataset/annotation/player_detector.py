"""プレー映像から領域内のプレイヤー骨格データを検出・トラッキングするクラス

    Args:
        映像データ(mp4)
        
    Attention:
        検出対応が見切れる可能性について考慮しない
        プレー選手のみのIDを手動で指定する必要がある

    Returns:
        プレイヤー骨格データ(検出ID,骨格座標群,信頼度)
"""
import cv2
import numpy as np
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import csv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.video_loader import VideoLoader
from src.dataset.annotation.collectors.center_playser_detector import CenterPlayerDetector
from src.dataset.annotation.exporters.pose_data_ecporter import PoseDataExporter


def main():
    """メイン処理 - 動画からプレイヤーを検出・トラッキング"""
    import argparse

    parser = argparse.ArgumentParser(
        description='画面中央のプレイヤーを全員検出し、継続的にトラッキングする'
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='入力動画ファイルパス'
    )
    parser.add_argument(
        '-o', '--output',       
        type=str,
        default=None,
        help='出力ビデオファイルパス（指定しない場合は保存しない）'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default=None,
        help='姿勢データのCSV出力パス（指定しない場合は保存しない）'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.5,
        help='検出信頼度の閾値（デフォルト: 0.5）'
    )
    parser.add_argument(
        '--center-ratio',
        type=float,
        default=0.3,
        help='中央領域の比率（デフォルト: 0.3）'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='使用デバイス（デフォルト: cpu）'
    )
    parser.add_argument(
        '--no-normalize',
        action='store_true',
        help='正規化を無効にする（絶対座標を使用）'
    )

    args = parser.parse_args()

    # 動画ファイルを開く
    print(f"動画ファイルを開いています: {args.input}...")

    # VideoLoaderを使用
    video_loader = VideoLoader(args.input)
    if not video_loader.open():
        print("エラー: 入力ソースを開けませんでした")
        return

    # フレーム情報を取得
    video_info = video_loader.get_info()
    width = video_info['width']
    height = video_info['height']
    fps = video_info['fps']

    print(f"入力情報:")
    print(f"  解像度: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  フレーム数: {video_info['frame_count']}")
    print(f"  長さ: {video_info['duration']:.2f}秒\n")

    # 検出器を初期化
    detector = CenterPlayerDetector(
        conf_threshold=args.conf,
        center_ratio=args.center_ratio,
        device=args.device
    )
    detector.set_frame_size(width, height)

    # 出力ビデオの準備
    video_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"出力ビデオ: {args.output}\n")

    # 姿勢データエクスポータを初期化
    pose_exporter = None
    if args.csv:
        use_normalization = not args.no_normalize
        pose_exporter = PoseDataExporter(use_normalization=use_normalization)
        norm_status = "有効" if use_normalization else "無効"
        print(f"姿勢データCSV: {args.csv}")
        print(f"  正規化: {norm_status}\n")

    # フレームカウント
    frame_count = 0

    print("処理開始...")
    print("  [q] キー: 終了")
    print("  [r] キー: トラッキングをリセット\n")

    try:
        while True:
            ret, frame = video_loader.read_frame()
            if not ret:
                print("動画が終了しました")
                break

            frame_count += 1

            # 常に画面中央からプレイヤーを検出（全員）
            target_persons = detector.detect_center_player(frame)

            # 姿勢データを記録（各プレイヤーごと）
            if pose_exporter and len(target_persons) > 0:
                timestamp = frame_count / fps
                for person in target_persons:
                    pose_exporter.add_frame_data(frame_count, timestamp, person)

            # 検出結果を描画（常に中央領域を表示）
            display_frame = detector.draw_results(frame, target_persons, show_center_region=True)

            # フレーム番号を表示
            cv2.putText(
                display_frame,
                f"Frame: {frame_count}",
                (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # 画面に表示
            cv2.imshow('Player Detection & Tracking', display_frame)

            # ビデオに保存
            if video_writer:
                video_writer.write(display_frame)

            # キー入力処理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("終了します...")
                break
            elif key == ord('r'):
                print("トラッキングをリセットします...")
                detector.reset()

    finally:
        # リソースを解放
        video_loader.close()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()

        print(f"\n処理完了:")
        print(f"  処理フレーム数: {frame_count}")
        if args.output:
            print(f"  出力ビデオ: {args.output}")

        # 姿勢データをCSVに保存
        if pose_exporter and args.csv:
            pose_exporter.export_csv(args.csv)

            # 統計情報を表示
            stats = pose_exporter.get_statistics()
            if stats:
                print(f"\n姿勢データ統計:")
                print(f"  トラッキングされたフレーム数: {stats['total_frames']}")
                print(f"  トラッキングID: {stats['track_ids']}")

                # 検出率が低いキーポイントを表示
                print(f"\nキーポイント検出率:")
                low_detection = []
                for kp_name, rate in stats['keypoint_detection_rates'].items():
                    if rate < 0.7:  # 70%未満のキーポイント
                        low_detection.append((kp_name, rate))

                if low_detection:
                    low_detection.sort(key=lambda x: x[1])
                    print("  検出率が低いキーポイント:")
                    for kp_name, rate in low_detection[:5]:  # 上位5つ
                        print(f"    {kp_name}: {rate*100:.1f}%")


if __name__ == "__main__":
    main()