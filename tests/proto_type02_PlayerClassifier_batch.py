"""
プレイヤー分類（PlayerClassifierBatch）のテストコード - バッチ処理版

このスクリプトは、動画全体を処理してから全データを使ってプレイヤーを特定します。

処理フロー:
1. 動画から卓球台位置を検出
2. 動画から全フレームの骨格データを収集
3. 収集完了後、全データを分析してプレイヤーを判定
4. プレイヤーIDリストを出力

使い方:
---------
# 動画からプレイヤーを検出・分類（バッチ処理）
python tests/proto_type02_PlayerClassifier_batch.py -i data/raw/sample_video_03_01.mp4

# 結果を可視化動画として保存
python tests/proto_type02_PlayerClassifier_batch.py -i data/raw/sample_video_03_01.mp4 -o output_batch.mp4

# 処理FPSを指定（デフォルト: 1fps）
python tests/proto_type02_PlayerClassifier_batch.py -i video.mp4 --fps 2

利点:
-----
- 動画全体の情報を使うため、より正確な判定
- IDが途中で変わっても、全体の傾向から判定可能
- 分析結果のサマリー表示
"""
import cv2
import numpy as np
import sys
from pathlib import Path
from typing import Optional, List, Set
import json

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.table_detector import TableDetector
from src.detection.yolopose_tracker import YOLOPose_Tracker
from src.detection.player_classifier_batch import PlayerClassifierBatch
from src.detection.data_classes import TableInfo, PersonTrack


def collect_data_from_video(
    video_path: str,
    table_detector: TableDetector,
    pose_tracker: YOLOPose_Tracker,
    player_classifier: PlayerClassifierBatch,
    process_fps: float = 1.0
) -> bool:
    """
    動画から全データを収集

    Args:
        video_path: 動画ファイルパス
        table_detector: TableDetectorインスタンス
        pose_tracker: YOLOPose_Trackerインスタンス
        player_classifier: PlayerClassifierBatchインスタンス
        process_fps: 処理フレームレート

    Returns:
        成功した場合True
    """
    print(f"\n=== フェーズ1: データ収集 ===")
    print(f"動画ファイル: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("エラー: 動画ファイルを開けませんでした")
        return False

    # 動画情報を取得
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(video_fps / process_fps)

    print(f"動画FPS: {video_fps:.2f}")
    print(f"総フレーム数: {total_frames}")
    print(f"処理FPS: {process_fps:.2f} (間隔: {frame_interval}フレーム)")

    # 卓球台を検出
    print(f"\n卓球台を検出中...")
    table_info = None
    max_detection_attempts = 100

    for attempt in range(max_detection_attempts):
        ret, frame = cap.read()
        if not ret:
            break

        table_info = table_detector.detect_table_from_frame(
            frame, frame_idx=attempt, force_detect=True
        )

        if table_info is not None:
            print(f"卓球台を検出しました（フレーム {attempt + 1}、信頼度: {table_info.confidence:.2f}）")
            break

        if (attempt + 1) % 10 == 0:
            print(f"  {attempt + 1}フレーム目まで検出試行中...")

    if table_info is None:
        print(f"エラー: 卓球台を検出できませんでした")
        cap.release()
        return False

    # 卓球台情報をセット
    player_classifier.set_table_info(table_info)

    # 動画を最初に戻す
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 全フレームを処理してデータ収集
    print(f"\n骨格データを収集中...")
    frame_count = 0
    processed_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 指定FPSで処理
        if frame_count % frame_interval != 0:
            continue

        processed_count += 1

        # 人物を検出・追跡
        persons = pose_tracker.track_frame(frame)

        # データを追加
        timestamp = frame_count / video_fps
        player_classifier.add_frame_data(frame_count, persons, timestamp)

        # 進捗表示
        if processed_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  進捗: {progress:.1f}% ({frame_count}/{total_frames}フレーム)")

    cap.release()

    print(f"\nデータ収集完了:")
    print(f"  処理フレーム数: {processed_count}")
    print(f"  検出されたtracking ID数: {len(player_classifier.tracking_data)}")

    return True


def visualize_results(
    video_path: str,
    output_path: str,
    player_ids: List[int],
    table_info: TableInfo,
    pose_tracker: YOLOPose_Tracker,
    process_fps: float = 1.0
):
    """
    結果を動画として可視化

    Args:
        video_path: 入力動画ファイルパス
        output_path: 出力動画ファイルパス
        player_ids: プレイヤーのtracking IDリスト
        table_info: 卓球台情報
        pose_tracker: YOLOPose_Trackerインスタンス
        process_fps: 処理フレームレート
    """
    print(f"\n=== 結果の可視化 ===")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("エラー: 動画ファイルを開けませんでした")
        return

    # 動画情報を取得
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(video_fps / process_fps)

    # 出力ビデオの準備
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))

    print(f"可視化中... (出力: {output_path})")

    # スケルトン接続定義
    skeleton_connections = [
        (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6),
        (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16)
    ]

    frame_count = 0
    processed_count = 0
    last_persons = []

    # トラッカーをリセット
    pose_tracker.reset_tracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 指定FPSで処理
        if frame_count % frame_interval != 0:
            persons = last_persons  # 前回の結果を使用
        else:
            processed_count += 1
            persons = pose_tracker.track_frame(frame)
            last_persons = persons

        output = frame.copy()

        # 卓球台を描画
        if table_info:
            x1, y1, x2, y2 = table_info.bbox
            cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            cv2.putText(output, "Table", (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 人物を描画
        for person in persons:
            is_player = person.track_id in player_ids
            color = (0, 255, 0) if is_player else (0, 0, 255)
            label = "PLAYER" if is_player else "Other"
            thickness = 3 if is_player else 2

            # バウンディングボックス
            x1, y1, x2, y2 = person.bbox
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(output, f"{label} ID:{person.track_id}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # スケルトン
            for connection in skeleton_connections:
                kp1_idx, kp2_idx = connection
                kp1 = person.keypoints[kp1_idx]
                kp2 = person.keypoints[kp2_idx]
                if kp1[2] > 0.5 and kp2[2] > 0.5:
                    pt1 = (int(kp1[0]), int(kp1[1]))
                    pt2 = (int(kp2[0]), int(kp2[1]))
                    cv2.line(output, pt1, pt2, color, 2)

            # キーポイント
            for kp in person.keypoints:
                if kp[2] > 0.5:
                    pt = (int(kp[0]), int(kp[1]))
                    cv2.circle(output, pt, 3, color, -1)

        # 情報表示
        cv2.putText(output, f"Frame: {frame_count}/{total_frames}",
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(output, f"Players: {len(player_ids)} (IDs: {sorted(player_ids)})",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # ビデオに保存
        video_writer.write(output)

        # 進捗表示
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  進捗: {progress:.1f}%")

    cap.release()
    video_writer.release()

    print(f"可視化完了: {output_path}")


def main():
    """メイン処理"""
    import argparse

    parser = argparse.ArgumentParser(
        description='プレイヤーをバッチ処理で検出・分類する'
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
        '--table-model',
        type=str,
        default='models/proto_type02_table_detection_models/best.pt',
        help='卓球台検出YOLOモデルのパス'
    )
    parser.add_argument(
        '--pose-model',
        type=str,
        default='yolo11n-pose.pt',
        help='YOLOv11-poseモデルのパス'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=1.0,
        help='処理フレームレート（デフォルト: 1.0fps）'
    )
    parser.add_argument(
        '--max-players',
        type=int,
        default=2,
        help='最大プレイヤー数（デフォルト: 2）'
    )

    args = parser.parse_args()

    # コンポーネントを初期化
    print("=== 初期化 ===")
    table_detector = TableDetector(yolo_model_path=args.table_model)
    pose_tracker = YOLOPose_Tracker(model_path=args.pose_model)
    player_classifier = PlayerClassifierBatch(max_players=args.max_players)

    # フェーズ1: データ収集
    success = collect_data_from_video(
        args.input,
        table_detector,
        pose_tracker,
        player_classifier,
        args.fps
    )

    if not success:
        print("エラー: データ収集に失敗しました")
        return

    # フェーズ2: 分析
    print(f"\n=== フェーズ2: データ分析 ===")
    player_ids = player_classifier.analyze_and_classify()

    print(f"\nプレイヤー検出結果:")
    print(f"  検出されたプレイヤー数: {len(player_ids)}")
    print(f"  プレイヤーID: {sorted(player_ids)}")

    # 分析サマリーを表示
    summary = player_classifier.get_analysis_summary()
    print(f"\n=== 分析サマリー ===")
    print(f"総フレーム数: {summary['total_frames']}")
    print(f"検出されたtracking ID数: {summary['total_tracking_ids']}")
    print(f"有効な候補者数: {summary['valid_candidates']}")

    print(f"\n候補者詳細（スコア順）:")
    for i, candidate in enumerate(summary['candidates'][:10], 1):
        is_player = candidate['track_id'] in player_ids
        marker = "[PLAYER]" if is_player else ""
        print(f"\n{i}. ID {candidate['track_id']} {marker}")
        print(f"   スコア: {candidate['score']:.3f}")
        print(f"   フレーム数: {candidate['total_frames']}")
        print(f"   総運動量: {candidate['total_movement']:.1f}")
        print(f"   卓球台付近比率: {candidate['near_table_ratio']:.1%}")

    # サマリーをJSONで保存
    output_json = args.input.replace('.mp4', '_analysis.json')
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump({
            'player_ids': player_ids,
            'summary': summary
        }, f, indent=2, ensure_ascii=False)
    print(f"\n分析結果を保存しました: {output_json}")

    # フェーズ3: 可視化
    if args.output:
        visualize_results(
            args.input,
            args.output,
            player_ids,
            player_classifier.table_info,
            pose_tracker,
            args.fps
        )


if __name__ == "__main__":
    main()
