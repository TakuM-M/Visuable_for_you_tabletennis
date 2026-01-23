"""
YOLOPose_PlayerDetector の動作確認テストコード

このスクリプトは、YOLOv11-Poseを使った人物検出とトラッキングが正しく動作することを確認します。
検出結果（バウンディングボックス、キーポイント、スケルトン）を映像上に可視化します。

使い方:
---------
# Webカメラから人物を検出
python tests/proto_type02_YOLOPose_PlayerDetector.py

# 動画ファイルから人物を検出
python tests/proto_type02_YOLOPose_PlayerDetector.py -i data/raw/sample_video_03_01.mp4

# 結果を動画として保存
python tests/proto_type02_YOLOPose_PlayerDetector.py -i data/raw/sample_video_03_01.mp4 -o output_player_detection.mp4

# YOLOモデルを指定
python tests/proto_type02_YOLOPose_PlayerDetector.py -i video.mp4 --model yolo11n-pose.pt

# 検出信頼度の閾値を調整
python tests/proto_type02_YOLOPose_PlayerDetector.py -i video.mp4 --conf 0.6

# 信頼度順にソートした上位N人のみ表示
python tests/proto_type02_YOLOPose_PlayerDetector.py -i video.mp4 --max-persons 2

機能:
-----
- YOLOPose_PlayerDetector: 人物を検出してトラッキング
- バウンディングボックスの描画
- キーポイント（17点）の描画
- スケルトンの描画
- トラッキングIDの表示
- 信頼度順のソート機能
- 検出情報の表示（人数、信頼度など）
- キーボード操作: [q]終了 / [r]トラッキングリセット
"""
import cv2
import numpy as np
import sys
from pathlib import Path
from typing import List, Optional

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.player_detector import YOLOPose_PlayerDetector
from src.detection.data_classes import PersonTrack


class PlayerDetectorVisualizer:
    """人物検出結果を可視化するクラス"""

    def __init__(self, detector: YOLOPose_PlayerDetector):
        """
        初期化

        Args:
            detector: YOLOPose_PlayerDetectorインスタンス
        """
        self.detector = detector

    def draw_detection_results(
        self,
        frame: np.ndarray,
        persons: List[PersonTrack],
        show_info: bool = True
    ) -> np.ndarray:
        """
        検出結果を描画

        Args:
            frame: 入力フレーム
            persons: 検出された人物のリスト
            show_info: 検出情報を表示するか

        Returns:
            描画後のフレーム
        """
        output = frame.copy()

        if len(persons) == 0:
            # 人物が検出されなかった場合
            if show_info:
                cv2.putText(
                    output,
                    "No Person Detected",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3
                )
            return output

        # 人物を描画
        output = self.detector.draw_tracking(
            output,
            persons,
            draw_bbox=True,
            draw_keypoints=True,
            draw_skeleton=True,
            draw_id=True
        )

        # 検出情報を表示
        if show_info:
            info_x = 20
            info_y = 40
            line_height = 35

            # 検出成功メッセージ
            cv2.putText(
                output,
                f"Detected: {len(persons)} Person(s)",
                (info_x, info_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )

            # 各人物の詳細情報
            for i, person in enumerate(persons[:3]):  # 最大3人まで表示
                y_offset = info_y + line_height * (i + 1)

                # トラッキングIDと信頼度
                cv2.putText(
                    output,
                    f"ID:{person.track_id} Conf:{person.confidence:.2f}",
                    (info_x, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

        return output


def main():
    """メイン処理 - Webカメラまたは動画から人物を検出・トラッキング"""
    import argparse

    parser = argparse.ArgumentParser(
        description='人物を検出してトラッキングし、バウンディングボックスとキーポイントを可視化する'
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        default='0',
        help='入力ソース（0: Webカメラ、またはビデオファイルパス）'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='出力ビデオファイルパス（指定しない場合は保存しない）'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='yolo11n-pose.pt',
        help='YOLOモデルのパス（デフォルト: yolo11n-pose.pt）'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.5,
        help='検出信頼度の閾値（デフォルト: 0.5）'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.7,
        help='IoU閾値（デフォルト: 0.7）'
    )
    parser.add_argument(
        '--max-persons',
        type=int,
        default=None,
        help='表示する最大人数（信頼度順、デフォルト: 制限なし）'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='使用デバイス（デフォルト: cpu）'
    )

    args = parser.parse_args()

    # 入力ソースを開く
    input_source = args.input
    if input_source.isdigit():
        input_source = int(input_source)
        print(f"Webカメラを開いています（デバイス: {input_source}）...")
    else:
        print(f"動画ファイルを開いています: {input_source}...")

    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        print("エラー: 入力ソースを開けませんでした")
        return

    # フレーム情報を取得
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if isinstance(input_source, str) else 0

    print(f"入力情報:")
    print(f"  解像度: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    if total_frames > 0:
        print(f"  総フレーム数: {total_frames}")
    print()

    # YOLOPose_PlayerDetectorを初期化
    print(f"YOLOPose_PlayerDetectorを初期化しています（モデル: {args.model}）...")
    detector = YOLOPose_PlayerDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device
    )
    visualizer = PlayerDetectorVisualizer(detector)

    # 出力ビデオの準備
    video_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"出力ビデオ: {args.output}\n")

    # 統計情報
    frame_count = 0
    detected_count = 0
    total_persons = 0

    print("処理開始...")
    print("  [q] キー: 終了")
    print("  [r] キー: トラッキングをリセット")
    print()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if isinstance(input_source, str):
                    print("動画が終了しました")
                else:
                    print("フレームの取得に失敗しました")
                break

            frame_count += 1

            # 人物を検出してトラッキング
            persons = detector.track_frame(frame, persist=True)

            # 信頼度順にソート
            persons.sort(key=lambda p: p.confidence, reverse=True)

            # 最大表示人数を制限
            if args.max_persons is not None:
                persons = persons[:args.max_persons]

            if len(persons) > 0:
                detected_count += 1
                total_persons += len(persons)

            # 検出結果を描画
            display_frame = visualizer.draw_detection_results(
                frame,
                persons,
                show_info=True
            )

            # フレーム番号を表示
            frame_text = f"Frame: {frame_count}"
            if total_frames > 0:
                frame_text += f"/{total_frames}"

            cv2.putText(
                display_frame,
                frame_text,
                (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # 画面に表示
            cv2.imshow('YOLOPose Player Detection Test', display_frame)

            # ビデオに保存
            if video_writer:
                video_writer.write(display_frame)

            # キー入力処理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("終了します...")
                break
            elif key == ord('r'):
                print(f"フレーム {frame_count}: トラッキングをリセットします...")
                detector.reset_tracker()

    finally:
        # リソースを解放
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()

        # 統計情報を表示
        print(f"\n処理完了:")
        print(f"  処理フレーム数: {frame_count}")
        print(f"  人物検出成功: {detected_count}フレーム ({detected_count/frame_count*100:.1f}%)")
        if detected_count > 0:
            print(f"  平均検出人数: {total_persons/detected_count:.1f}人/フレーム")
        if args.output:
            print(f"  出力ビデオ: {args.output}")


if __name__ == "__main__":
    main()
