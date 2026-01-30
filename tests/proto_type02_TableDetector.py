"""
卓球台検出（TableDetector）のテストコード

このスクリプトは、YOLOを使った卓球台検出が正しく動作することを確認します。
検出結果（バウンディングボックスと基本情報）を映像上に可視化します。

使い方:
---------
# 動画ファイルから卓球台を検出
python tests/proto_type02_TableDetector.py -i data/raw/sample_video_03_01.mp4

# 結果を動画として保存
python tests/proto_type02_TableDetector.py -i data/raw/sample_video_03_01.mp4 -o output_table_detection.mp4

# YOLOモデルを指定
python tests/proto_type02_TableDetector.py -i video.mp4 --model yolo11n.pt

機能:
-----
- TableDetector: 卓球台を検出し、バウンディングボックスを取得
- バウンディングボックスの描画
- 検出情報の表示（信頼度、アスペクト比など）
- キーボード操作: [q]終了 / [d]強制再検出
"""
import cv2
import numpy as np
import sys
from pathlib import Path
from typing import Optional

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.table_detector import TableDetector
from src.core.data_classes import TableInfo


class TableDetectorVisualizer:
    """卓球台検出結果を可視化するクラス"""

    def __init__(self, detector: TableDetector):
        """
        初期化

        Args:
            detector: TableDetectorインスタンス
        """
        self.detector = detector

    def draw_detection_results(
        self,
        frame: np.ndarray,
        table_info: Optional[TableInfo]
    ) -> np.ndarray:
        """
        検出結果を描画（バウンディングボックスと基本情報のみ）

        Args:
            frame: 入力フレーム
            table_info: 卓球台情報

        Returns:
            描画後のフレーム
        """
        output = frame.copy()

        if table_info is None:
            # 卓球台が検出されなかった場合
            cv2.putText(
                output,
                "Table NOT Detected",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3
            )
            return output

        # バウンディングボックスを描画
        x1, y1, x2, y2 = table_info.bbox
        cv2.rectangle(
            output,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            3
        )

        # 中心座標を描画
        center = table_info.center
        cv2.circle(output, (int(center[0]), int(center[1])), 8, (0, 255, 255), -1)
        cv2.circle(output, (int(center[0]), int(center[1])), 3, (255, 255, 255), -1)

        # 卓球台情報を表示
        info_x = 20
        info_y = 40
        line_height = 35

        # 検出成功メッセージ
        cv2.putText(
            output,
            "Table Detected!",
            (info_x, info_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

        # 信頼度
        cv2.putText(
            output,
            f"Confidence: {table_info.confidence:.2f}",
            (info_x, info_y + line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # バウンディングボックス座標
        cv2.putText(
            output,
            f"BBox: ({int(x1)}, {int(y1)}) - ({int(x2)}, {int(y2)})",
            (info_x, info_y + line_height * 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # 中心座標
        cv2.putText(
            output,
            f"Center: ({int(center[0])}, {int(center[1])})",
            (info_x, info_y + line_height * 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # アスペクト比
        cv2.putText(
            output,
            f"Aspect Ratio: {table_info.aspect_ratio:.2f}",
            (info_x, info_y + line_height * 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        return output


def main():
    """メイン処理 - 動画から卓球台を検出"""
    import argparse

    parser = argparse.ArgumentParser(
        description='卓球台を検出し、バウンディングボックスと基本情報を可視化する'
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
        help='出力動画ファイルパス（指定しない場合は保存しない）'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/proto_type02_table_detection_models/best.pt',
        help='YOLOモデルのパス（デフォルト: models/proto_type02_table_detection_models/best.pt）'
    )

    args = parser.parse_args()

    # 入力動画を開く
    print(f"動画ファイルを開いています: {args.input}...")
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print("エラー: 動画ファイルを開けませんでした")
        return

    # フレーム情報を取得
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"入力情報:")
    print(f"  解像度: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  総フレーム数: {total_frames}\n")

    # TableDetectorを初期化
    print(f"TableDetectorを初期化しています（モデル: {args.model}）...")
    detector = TableDetector(yolo_model_path=args.model)
    visualizer = TableDetectorVisualizer(detector)

    # 出力ビデオの準備
    video_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"出力ビデオ: {args.output}\n")

    # フレームカウント
    frame_count = 0
    detected_count = 0

    print("処理開始...")
    print("  [q] キー: 終了")
    print("  [d] キー: 強制再検出\n")

    force_detect = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("動画が終了しました")
                break

            frame_count += 1

            # 卓球台を検出
            table_info = detector.detect_table_from_frame(
                frame,
                frame_idx=frame_count,
                force_detect=force_detect
            )
            force_detect = False  # リセット

            if table_info is not None:
                detected_count += 1

            # 検出結果を描画
            display_frame = visualizer.draw_detection_results(
                frame,
                table_info
            )

            # フレーム番号とキャッシュ状態を表示
            cache_text = "(Cached)" if (
                table_info is not None and
                detector._cached_table_info is not None and
                frame_count != detector._cache_frame_idx
            ) else ""

            cv2.putText(
                display_frame,
                f"Frame: {frame_count}/{total_frames} {cache_text}",
                (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # 画面に表示
            cv2.imshow('Table Detection Test', display_frame)

            # ビデオに保存
            if video_writer:
                video_writer.write(display_frame)

            # キー入力処理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("終了します...")
                break
            elif key == ord('d'):
                print(f"フレーム {frame_count}: 強制再検出します...")
                force_detect = True

    finally:
        # リソースを解放
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()

        # 統計情報を表示
        print(f"\n処理完了:")
        print(f"  処理フレーム数: {frame_count}")
        print(f"  卓球台検出成功: {detected_count}フレーム ({detected_count/frame_count*100:.1f}%)")
        if args.output:
            print(f"  出力ビデオ: {args.output}")


if __name__ == "__main__":
    main()
