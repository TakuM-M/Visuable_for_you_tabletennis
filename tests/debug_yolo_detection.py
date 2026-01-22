"""
YOLO検出デバッグスクリプト

YOLOが何を検出しているかを確認するためのスクリプト
全クラスの検出結果を表示し、クラスIDとクラス名を確認できます

使い方:
---------
# 動画の最初のフレームで検出
python tests/debug_yolo_detection.py -i data/raw/sample_video_03_01.mp4

# 動画の特定のフレームで検出
python tests/debug_yolo_detection.py -i data/raw/sample_video_03_01.mp4 --frame 200

# 画像ファイルで検出
python tests/debug_yolo_detection.py -i image.jpg

# YOLOモデルを指定
python tests/debug_yolo_detection.py -i video.mp4 --model yolo11n.pt

# 検出結果を画像として保存
python tests/debug_yolo_detection.py -i data/raw/sample_video_03_01.mp4 -o debug_detection.jpg
"""
import cv2
import numpy as np
import sys
from pathlib import Path
from typing import List, Dict, Any

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO


def draw_all_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    """
    全検出結果を描画

    Args:
        frame: 入力フレーム
        detections: 検出結果のリスト

    Returns:
        描画後のフレーム
    """
    output = frame.copy()

    # 各クラスに異なる色を割り当て
    colors = [
        (0, 255, 0),    # 緑
        (255, 0, 0),    # 青
        (0, 0, 255),    # 赤
        (255, 255, 0),  # シアン
        (255, 0, 255),  # マゼンタ
        (0, 255, 255),  # 黄色
        (128, 128, 0),  # ティール
        (128, 0, 128),  # パープル
    ]

    for i, obj in enumerate(detections):
        x1, y1, x2, y2 = obj['bbox']
        class_name = obj['class_name']
        class_id = obj['class_id']
        confidence = obj['confidence']
        center = obj['center']

        # クラスIDに応じた色を選択
        color = colors[class_id % len(colors)]

        # バウンディングボックスを描画
        cv2.rectangle(
            output,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            2
        )

        # クラス名と信頼度を表示
        label = f"{class_name} ({confidence:.2f})"
        cv2.putText(
            output,
            label,
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        # 中心座標を描画
        cv2.circle(output, (int(center[0]), int(center[1])), 5, color, -1)

    return output


def detect_all_objects(yolo_model: YOLO, frame: np.ndarray) -> List[Dict[str, Any]]:
    """
    YOLOで全オブジェクトを検出

    Args:
        yolo_model: YOLOモデル
        frame: 入力フレーム

    Returns:
        検出された全オブジェクトのリスト
    """
    # YOLO検出
    results = yolo_model(frame, verbose=False)

    if len(results) == 0 or len(results[0].boxes) == 0:
        return []

    # 全検出結果を取得
    boxes = results[0].boxes
    detected_objects = []

    for box in boxes:
        # バウンディングボックス取得
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        confidence = float(box.conf[0].cpu().numpy())
        class_id = int(box.cls[0].cpu().numpy())

        # クラス名を取得
        class_name = yolo_model.names[class_id] if class_id in yolo_model.names else f"class_{class_id}"

        # 中心座標を計算
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        detected_objects.append({
            'class_id': class_id,
            'class_name': class_name,
            'bbox': (float(x1), float(y1), float(x2), float(y2)),
            'confidence': confidence,
            'center': (float(center_x), float(center_y))
        })

    return detected_objects


def load_frame(input_path: str, frame_number: int = 0) -> tuple:
    """
    動画または画像からフレームを読み込む

    Args:
        input_path: 入力ファイルパス
        frame_number: フレーム番号（動画の場合のみ）

    Returns:
        (frame, is_video, total_frames)
    """
    # 画像ファイルの拡張子
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

    input_path_obj = Path(input_path)

    # 画像ファイルかチェック
    if input_path_obj.suffix.lower() in image_extensions:
        print(f"画像ファイルを読み込んでいます: {input_path}...")
        frame = cv2.imread(input_path)
        if frame is None:
            raise ValueError(f"画像ファイルを読み込めませんでした: {input_path}")
        print(f"  解像度: {frame.shape[1]}x{frame.shape[0]}")
        return frame, False, 1

    # 動画ファイルとして扱う
    print(f"動画ファイルを開いています: {input_path}...")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"動画ファイルを開けませんでした: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 指定されたフレームに移動
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"フレーム {frame_number} を読み込めませんでした")

    print(f"フレーム {frame_number}/{total_frames} を読み込みました")
    print(f"  解像度: {frame.shape[1]}x{frame.shape[0]}")

    return frame, True, total_frames


def main():
    """メイン処理"""
    import argparse

    parser = argparse.ArgumentParser(
        description='YOLOの検出結果をデバッグする（動画・画像対応）'
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='入力ファイルパス（動画または画像）'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='出力画像ファイルパス（指定しない場合は保存しない）'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='yolo11n.pt',
        help='YOLOモデルのパス（デフォルト: yolo11n.pt）'
    )
    parser.add_argument(
        '--frame',
        type=int,
        default=0,
        help='検出するフレーム番号（動画の場合のみ、デフォルト: 0）'
    )

    args = parser.parse_args()

    try:
        # フレームを読み込む
        frame, is_video, total_frames = load_frame(args.input, args.frame)

        frame_label = f"Frame {args.frame}/{total_frames}" if is_video else "Image"

    except Exception as e:
        print(f"エラー: {e}")
        return

    # YOLOモデルを初期化
    print(f"\nYOLOモデルを初期化しています（モデル: {args.model}）...")
    try:
        yolo_model = YOLO(args.model)
    except Exception as e:
        print(f"エラー: YOLOモデルのロードに失敗しました: {e}")
        return

    # 全オブジェクトを検出
    print("\n検出を実行中...")
    detections = detect_all_objects(yolo_model, frame)

    # 検出結果を表示
    print(f"\n検出結果: {len(detections)}個のオブジェクトを検出\n")

    if len(detections) == 0:
        print("オブジェクトが検出されませんでした")
        return

    # クラスごとに集計
    class_counts = {}
    for obj in detections:
        class_name = obj['class_name']
        if class_name not in class_counts:
            class_counts[class_name] = 0
        class_counts[class_name] += 1

    print("クラス別集計:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name}: {count}個")

    print("\n詳細:")
    print("-" * 80)
    for i, obj in enumerate(detections, 1):
        print(f"{i}. クラス: {obj['class_name']} (ID: {obj['class_id']})")
        print(f"   信頼度: {obj['confidence']:.3f}")
        print(f"   位置: bbox={obj['bbox']}, center={obj['center']}")

    # 検出結果を描画
    result_frame = draw_all_detections(frame, detections)

    # タイトルを追加
    cv2.putText(
        result_frame,
        f"YOLO Detection Debug - {frame_label}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )
    cv2.putText(
        result_frame,
        f"Total: {len(detections)} objects",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # 画像を表示
    cv2.imshow('YOLO Detection Debug', result_frame)
    print(f"\n[q]キーで終了")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 画像を保存
    if args.output:
        cv2.imwrite(args.output, result_frame)
        print(f"\n検出結果を保存しました: {args.output}")


if __name__ == "__main__":
    main()
