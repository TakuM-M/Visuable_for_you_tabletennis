"""
プレイヤー分類（PlayerClassifier）のテストコード

このスクリプトは、YOLOv11-poseで検出された人物の中から
実際のプレイヤーを特定する機能が正しく動作することを確認します。

使い方:
---------
# 動画ファイルからプレイヤーを検出・分類
python tests/proto_type02_PlayerClassifier.py -i data/raw/sample_video_03_01.mp4

# 結果を動画として保存
python tests/proto_type02_PlayerClassifier.py -i data/raw/sample_video_03_01.mp4 -o output_player_classification.mp4 --fps 30

# 1fpsでサンプリング（デフォルト）
python tests/proto_type02_PlayerClassifier.py -i video.mp4 --fps 1

機能:
-----
- TableDetector: 卓球台を検出
- YOLOPose_Tracker: 人物の骨格データと追跡
- PlayerClassifier: プレイヤーの特定
- 可視化:
  - プレイヤー: 緑色
  - その他の人物: 赤色
  - バウンディングボックス、骨格データ、トラッキングID
  - プレイヤースコア情報
"""
import cv2
import numpy as np
import sys
from pathlib import Path
from typing import Optional, List, Set

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.table_detector import TableDetector
from src.detection.yolopose_tracker import YOLOPose_Tracker
from src.detection.player_classifier import PlayerClassifier
from src.detection.data_classes import TableInfo, PersonTrack


class PlayerClassifierVisualizer:
    """プレイヤー分類結果を可視化するクラス"""

    def __init__(
        self,
        table_detector: TableDetector,
        pose_tracker: YOLOPose_Tracker,
        player_classifier: PlayerClassifier
    ):
        """
        初期化

        Args:
            table_detector: TableDetectorインスタンス
            pose_tracker: YOLOPose_Trackerインスタンス
            player_classifier: PlayerClassifierインスタンス
        """
        self.table_detector = table_detector
        self.pose_tracker = pose_tracker
        self.player_classifier = player_classifier

    def draw_results(
        self,
        frame: np.ndarray,
        table_info: Optional[TableInfo],
        persons: List[PersonTrack],
        player_ids: Set[int]
    ) -> np.ndarray:
        """
        検出結果を描画

        Args:
            frame: 入力フレーム
            table_info: 卓球台情報
            persons: 検出された人物リスト
            player_ids: プレイヤーのtracking IDセット

        Returns:
            描画後のフレーム
        """
        output = frame.copy()

        # 卓球台を描画
        if table_info:
            x1, y1, x2, y2 = table_info.bbox
            cv2.rectangle(
                output,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 255),
                2
            )
            cv2.putText(
                output,
                "Table",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
        else:
            # 卓球台が検出できていない場合の警告表示
            cv2.putText(
                output,
                "WARNING: Table Not Detected",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3
            )
            cv2.putText(
                output,
                "Player classification is disabled",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        # スケルトン接続定義
        skeleton_connections = [
            (0, 1), (0, 2),  # 鼻 - 目
            (1, 3), (2, 4),  # 目 - 耳
            (0, 5), (0, 6),  # 鼻 - 肩
            (5, 6),          # 肩 - 肩
            (5, 7), (7, 9),  # 左腕
            (6, 8), (8, 10), # 右腕
            (5, 11), (6, 12),# 肩 - 腰
            (11, 12),        # 腰 - 腰
            (11, 13), (13, 15),  # 左脚
            (12, 14), (14, 16)   # 右脚
        ]

        # 人物を描画
        for person in persons:
            is_player = person.track_id in player_ids
            color = (0, 255, 0) if is_player else (0, 0, 255)  # 緑=プレイヤー, 赤=その他
            label = "PLAYER" if is_player else "Other"

            # バウンディングボックスを描画
            x1, y1, x2, y2 = person.bbox
            thickness = 3 if is_player else 2
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

            # ラベルとIDを描画
            text = f"{label} ID:{person.track_id}"
            cv2.putText(
                output,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            # スケルトンを描画
            for connection in skeleton_connections:
                kp1_idx, kp2_idx = connection
                kp1 = person.keypoints[kp1_idx]
                kp2 = person.keypoints[kp2_idx]

                if kp1[2] > 0.5 and kp2[2] > 0.5:
                    pt1 = (int(kp1[0]), int(kp1[1]))
                    pt2 = (int(kp2[0]), int(kp2[1]))
                    cv2.line(output, pt1, pt2, color, 2)

            # キーポイントを描画
            for kp in person.keypoints:
                if kp[2] > 0.5:
                    pt = (int(kp[0]), int(kp[1]))
                    cv2.circle(output, pt, 3, color, -1)

        return output

    def draw_candidate_info(
        self,
        frame: np.ndarray,
        player_ids: Set[int]
    ) -> np.ndarray:
        """
        候補者情報を描画

        Args:
            frame: 入力フレーム
            player_ids: プレイヤーのtracking IDセット

        Returns:
            描画後のフレーム
        """
        output = frame.copy()
        height = frame.shape[0]

        # 右上に候補情報を表示
        info_x = frame.shape[1] - 400
        info_y = 30
        line_height = 25

        cv2.putText(
            output,
            "=== Player Candidates ===",
            (info_x, info_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        y_offset = info_y + line_height

        # 候補情報を取得してスコア順にソート
        candidates = []
        for track_id, candidate in self.player_classifier.candidates.items():
            if candidate.total_frames >= self.player_classifier.min_tracking_frames:
                score = self.player_classifier._calculate_player_score(candidate)
                candidates.append((track_id, candidate, score))

        candidates.sort(key=lambda x: x[2], reverse=True)

        # 上位候補を表示
        for i, (track_id, candidate, score) in enumerate(candidates[:5]):
            is_player = track_id in player_ids
            color = (0, 255, 0) if is_player else (200, 200, 200)

            # ID情報
            text = f"ID:{track_id} {'[PLAYER]' if is_player else ''}"
            cv2.putText(
                output,
                text,
                (info_x, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )
            y_offset += line_height

            # スコア情報
            text = f"  Score: {score:.3f}"
            cv2.putText(
                output,
                text,
                (info_x, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1
            )
            y_offset += 18

            # 詳細情報
            text = f"  Frames: {candidate.total_frames}, Move: {candidate.total_movement:.1f}"
            cv2.putText(
                output,
                text,
                (info_x, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1
            )
            y_offset += 18

            text = f"  Near table: {candidate.near_table_ratio:.1%}"
            cv2.putText(
                output,
                text,
                (info_x, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1
            )
            y_offset += line_height

        return output


def main():
    """メイン処理"""
    import argparse

    parser = argparse.ArgumentParser(
        description='プレイヤーを検出・分類し、結果を可視化する'
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
        help='最大プレイヤー数（デフォルト: 5）'
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
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"入力情報:")
    print(f"  解像度: {width}x{height}")
    print(f"  FPS: {video_fps:.2f}")
    print(f"  総フレーム数: {total_frames}")
    print(f"  処理FPS: {args.fps:.2f}\n")

    # フレーム間隔を計算
    frame_interval = int(video_fps / args.fps)

    # コンポーネントを初期化
    print("コンポーネントを初期化しています...")
    table_detector = TableDetector(yolo_model_path=args.table_model)
    pose_tracker = YOLOPose_Tracker(model_path=args.pose_model)
    player_classifier = PlayerClassifier(max_players=args.max_players)
    visualizer = PlayerClassifierVisualizer(
        table_detector,
        pose_tracker,
        player_classifier
    )

    # 出力ビデオの準備
    video_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.output, fourcc, video_fps, (width, height))
        print(f"出力ビデオ: {args.output}\n")

    # フレームカウント
    frame_count = 0
    processed_count = 0
    player_ids = set()

    print("処理開始...")
    print("  [q] キー: 終了\n")

    # 卓球台を検出（見つかるまで複数フレームを試行）
    print("卓球台を検出中...")
    table_info = None
    max_detection_attempts = 100  # 最大100フレームまで試行

    for attempt in range(max_detection_attempts):
        ret, frame = cap.read()
        if not ret:
            print("エラー: 動画の終端に達しました")
            cap.release()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
            return

        table_info = table_detector.detect_table_from_frame(frame, frame_idx=attempt, force_detect=True)

        if table_info is not None:
            print(f"卓球台を検出しました（フレーム {attempt + 1}、信頼度: {table_info.confidence:.2f}）\n")
            break

        if (attempt + 1) % 10 == 0:
            print(f"  {attempt + 1}フレーム目まで検出試行中...")

    if table_info is None:
        print(f"エラー: {max_detection_attempts}フレーム試行しましたが、卓球台を検出できませんでした")
        print("以下を確認してください:")
        print("  - 動画に卓球台が映っているか")
        print("  - モデルパスが正しいか (--table-model)")
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        return

    # 動画を最初に戻す
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("動画が終了しました")
                break

            frame_count += 1

            # 指定FPSで処理
            if frame_count % frame_interval != 0:
                # 前回の結果を表示
                if processed_count > 0:
                    display_frame = visualizer.draw_results(
                        frame, table_info, last_persons, player_ids
                    )
                    display_frame = visualizer.draw_candidate_info(
                        display_frame, player_ids
                    )
                else:
                    display_frame = frame.copy()

                cv2.putText(
                    display_frame,
                    f"Frame: {frame_count}/{total_frames} (Skipped)",
                    (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (150, 150, 150),
                    2
                )

                cv2.imshow('Player Classifier Test', display_frame)
                if video_writer:
                    video_writer.write(display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("終了します...")
                    break
                continue

            processed_count += 1

            # 卓球台が検出できていない場合は再試行
            if table_info is None:
                table_info = table_detector.detect_table_from_frame(frame, frame_idx=frame_count, force_detect=True)
                if table_info is not None:
                    print(f"卓球台を検出しました（フレーム {frame_count}、信頼度: {table_info.confidence:.2f}）\n")

            # 人物を検出・追跡（卓球台フィルタリング適用）
            if table_info:
                persons = pose_tracker.track_frame_with_table_filter(frame, table_info)
            else:
                persons = pose_tracker.track_frame(frame)

            # プレイヤー分類器を更新
            if table_info and persons:
                player_classifier.update(persons, table_info, frame_count)
            elif table_info is None and processed_count == 1:
                print("警告: 卓球台が検出できていないため、プレイヤー分類は実行されません")

            # プレイヤーを分類（毎フレーム更新）
            # 重要: 古いIDをプレイヤーとして保持し続けないため、毎回更新する
            if table_info:
                player_ids = set(player_classifier.classify_players())
                # デバッグ出力は一定フレームごと
                if processed_count % 10 == 0 or processed_count == 1:
                    if player_ids:
                        print(f"Frame {frame_count}: プレイヤーID = {sorted(player_ids)}")

            # 結果を描画
            display_frame = visualizer.draw_results(
                frame, table_info, persons, player_ids
            )
            display_frame = visualizer.draw_candidate_info(
                display_frame, player_ids
            )

            # フレーム情報を表示
            cv2.putText(
                display_frame,
                f"Frame: {frame_count}/{total_frames} (Processed: {processed_count})",
                (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display_frame,
                f"Detected: {len(persons)} persons, Players: {len(player_ids)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # 画面に表示
            cv2.imshow('Player Classifier Test', display_frame)

            # ビデオに保存
            if video_writer:
                video_writer.write(display_frame)

            # 次回のスキップフレーム用に保存
            last_persons = persons

            # キー入力処理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("終了します...")
                break

    finally:
        # リソースを解放
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()

        # 最終結果を表示
        print(f"\n処理完了:")
        print(f"  処理フレーム数: {frame_count}")
        print(f"  実際に処理したフレーム数: {processed_count}")
        print(f"  検出されたプレイヤーID: {sorted(player_ids)}")
        print(f"  候補者数: {len(player_classifier.candidates)}")

        # 候補者詳細情報
        if player_classifier.candidates:
            print(f"\n=== 候補者詳細 ===")
            candidates = []
            for track_id, candidate in player_classifier.candidates.items():
                if candidate.total_frames >= player_classifier.min_tracking_frames:
                    score = player_classifier._calculate_player_score(candidate)
                    candidates.append((track_id, candidate, score))

            candidates.sort(key=lambda x: x[2], reverse=True)

            for track_id, candidate, score in candidates:
                is_player = track_id in player_ids
                print(f"\nID {track_id} {'[PLAYER]' if is_player else ''}:")
                print(f"  スコア: {score:.3f}")
                print(f"  フレーム数: {candidate.total_frames}")
                print(f"  総運動量: {candidate.total_movement:.1f}")
                print(f"  卓球台付近比率: {candidate.near_table_ratio:.1%}")

        if args.output:
            print(f"\n出力ビデオ: {args.output}")


if __name__ == "__main__":
    main()
