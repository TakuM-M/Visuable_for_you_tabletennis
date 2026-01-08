"""
プレイヤーを継続して追跡し，姿勢データを抽出・保存するスクリプト

前提条件:
- 卓球台の位置は画面の中央に位置すると仮定する
- 選手は卓球台の前に立つと仮定する
"""
import cv2
import numpy as np
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import csv

sys.path.insert(0, str(Path(__file__).parent.parent))

from detection.yolo_tracker import YOLOPoseTracker, PersonTrack, KEYPOINT_NAMES
from utils.video_loader import VideoLoader


@dataclass
class CenterRegion:
    """画面中央領域の定義"""
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def contains_point(self, x: float, y: float) -> bool:
        """座標が中央領域内にあるか判定"""
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def contains_person(self, person: PersonTrack) -> bool:
        """人物のバウンディングボックス中心が中央領域内にあるか判定"""
        center_x, center_y = person.get_center()
        return self.contains_point(center_x, center_y)


class CenterPlayerDetector:
    """画面中央のプレイヤー検出・トラッキングクラス"""

    def __init__(
        self,
        model_path: str = "yolo11n-pose.pt",
        conf_threshold: float = 0.5,
        center_ratio: float = 0.3,
        device: str = "cpu"
    ):
        """
        初期化

        Args:
            model_path: YOLOモデルのパス
            conf_threshold: 検出信頼度の閾値
            center_ratio: 中央領域の比率（画面の何%を中央とするか）
            device: 使用デバイス
        """
        self.tracker = YOLOPoseTracker(
            model_path=model_path,
            conf_threshold=conf_threshold,
            device=device
        )
        self.center_ratio = center_ratio
        self.target_track_ids: list[int] = []  # 最大2名のトラッキングID
        self.center_region: Optional[CenterRegion] = None
        self.frame_width: Optional[int] = None
        self.frame_height: Optional[int] = None
        self.max_players: int = 2  # 検出する最大人数

    def set_frame_size(self, width: int, height: int):
        """フレームサイズを設定し、中央領域を計算"""
        self.frame_width = width
        self.frame_height = height

        # 中央領域を計算（画面の中央center_ratio%の範囲）
        center_w = int(width * self.center_ratio)
        center_h = int(height * self.center_ratio)

        self.center_region = CenterRegion(
            x_min=(width - center_w) // 2,
            y_min=(height - center_h) // 2,
            x_max=(width + center_w) // 2,
            y_max=(height + center_h) // 2
        )

        print(f"中央検出領域を設定:")
        print(f"  範囲: ({self.center_region.x_min}, {self.center_region.y_min}) - "
              f"({self.center_region.x_max}, {self.center_region.y_max})")
        print(f"  サイズ: {center_w} x {center_h}\n")

    def detect_center_player(self, frame: np.ndarray) -> list[PersonTrack]:
        """
        画面中央のプレイヤーを検出（毎フレーム実行、最大2名）

        Args:
            frame: 入力フレーム

        Returns:
            検出されたプレイヤーのリスト（最大2名）
        """
        if self.center_region is None:
            self.set_frame_size(frame.shape[1], frame.shape[0])

        # 全人物を検出（トラッキング有効）
        persons = self.tracker.track_frame(frame, persist=True)

        # 中央領域内にいる人物を探す
        center_persons = [p for p in persons if self.center_region.contains_person(p)]

        if not center_persons:
            return []

        # 信頼度が高い順にソートして最大2名を選択
        center_persons.sort(key=lambda p: p.confidence, reverse=True)
        target_persons = center_persons[:self.max_players]

        # トラッキングIDを更新
        new_track_ids = [p.track_id for p in target_persons]

        # 新しいプレイヤーが検出された場合のみログ出力
        if set(new_track_ids) != set(self.target_track_ids):
            self.target_track_ids = new_track_ids
            print(f"フレーム内のプレイヤーを検出（{len(target_persons)}名）:")
            for i, person in enumerate(target_persons, 1):
                print(f"  プレイヤー{i}:")
                print(f"    トラッキングID: {person.track_id}")
                print(f"    信頼度: {person.confidence:.3f}")
                print(f"    位置: ({person.get_center()[0]:.1f}, {person.get_center()[1]:.1f})")
            print()
        else:
            self.target_track_ids = new_track_ids

        return target_persons

    def track_player(self, frame: np.ndarray) -> list[PersonTrack]:
        """
        ターゲットプレイヤーをトラッキング（画面全体、最大2名）

        Args:
            frame: 入力フレーム

        Returns:
            トラッキングされたプレイヤーのリスト
        """
        if not self.target_track_ids:
            return []

        # 全人物をトラッキング
        persons = self.tracker.track_frame(frame, persist=True)

        # ターゲットIDの人物を探す
        target_persons = []
        for person in persons:
            if person.track_id in self.target_track_ids:
                target_persons.append(person)

        return target_persons

    def draw_results(
        self,
        frame: np.ndarray,
        persons: list[PersonTrack] = None,
        show_center_region: bool = False
    ) -> np.ndarray:
        """
        検出・トラッキング結果を描画

        Args:
            frame: 入力フレーム
            persons: 描画するプレイヤーのリスト
            show_center_region: 中央領域を表示するか

        Returns:
            描画後のフレーム
        """
        output = frame.copy()

        # 中央領域を描画
        if show_center_region and self.center_region is not None:
            cv2.rectangle(
                output,
                (self.center_region.x_min, self.center_region.y_min),
                (self.center_region.x_max, self.center_region.y_max),
                (255, 255, 0),
                2
            )
            cv2.putText(
                output,
                "Center Detection Area",
                (self.center_region.x_min, self.center_region.y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

        # プレイヤーを描画
        if persons and len(persons) > 0:
            output = self.tracker.draw_tracking(
                output,
                persons,
                draw_bbox=True,
                draw_keypoints=True,
                draw_skeleton=True,
                draw_id=True
            )

            # ステータス表示
            status_text = f"Tracking {len(persons)} Player(s) (IDs: {[p.track_id for p in persons]})"
            cv2.putText(
                output,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )
        else:
            # プレイヤー未検出
            status_text = "Searching for players in center..."
            cv2.putText(
                output,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )

        return output

    def reset(self):
        """トラッキングをリセット"""
        self.target_track_ids = []
        self.tracker.reset_tracker()


class PoseDataExporter:
    """姿勢データをCSVに出力するクラス"""

    def __init__(self):
        self.pose_data = []

    def add_frame_data(self, frame_num: int, timestamp: float, person: PersonTrack):
        """
        フレームの姿勢データを追加

        Args:
            frame_num: フレーム番号
            timestamp: タイムスタンプ（秒）
            person: トラッキングされた人物
        """
        # 各キーポイントのデータを追加
        frame_data = {
            'frame': frame_num,
            'timestamp': timestamp,
            'track_id': person.track_id,
            'bbox_x1': person.bbox[0],
            'bbox_y1': person.bbox[1],
            'bbox_x2': person.bbox[2],
            'bbox_y2': person.bbox[3],
            'confidence': person.confidence
        }

        # キーポイントデータを追加
        for i, keypoint_name in enumerate(KEYPOINT_NAMES):
            kp = person.keypoints[i]
            frame_data[f'{keypoint_name}_x'] = kp[0]
            frame_data[f'{keypoint_name}_y'] = kp[1]
            frame_data[f'{keypoint_name}_conf'] = kp[2]

        self.pose_data.append(frame_data)

    def export_csv(self, output_path: str):
        """
        姿勢データをCSVファイルに出力

        Args:
            output_path: 出力ファイルパス
        """
        if not self.pose_data:
            print("警告: 出力するデータがありません")
            return

        # CSVのヘッダーを作成
        fieldnames = ['frame', 'timestamp', 'track_id', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'confidence']
        for keypoint_name in KEYPOINT_NAMES:
            fieldnames.extend([
                f'{keypoint_name}_x',
                f'{keypoint_name}_y',
                f'{keypoint_name}_conf'
            ])

        # CSVに書き込み
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.pose_data)

        print(f"姿勢データをCSVに保存しました: {output_path}")
        print(f"  総フレーム数: {len(self.pose_data)}")

    def get_statistics(self):
        """統計情報を取得"""
        if not self.pose_data:
            return {}

        total_frames = len(self.pose_data)
        track_ids = set(data['track_id'] for data in self.pose_data)

        # 各キーポイントの検出率を計算
        keypoint_detection_rates = {}
        for keypoint_name in KEYPOINT_NAMES:
            detected_count = sum(
                1 for data in self.pose_data
                if data[f'{keypoint_name}_conf'] > 0.5
            )
            keypoint_detection_rates[keypoint_name] = detected_count / total_frames

        return {
            'total_frames': total_frames,
            'track_ids': list(track_ids),
            'keypoint_detection_rates': keypoint_detection_rates
        }


def main():
    """メイン処理 - 動画からプレイヤーを検出・トラッキング"""
    import argparse

    parser = argparse.ArgumentParser(
        description='画面中央のプレイヤーを検出し、継続的にトラッキングする'
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
        pose_exporter = PoseDataExporter()
        print(f"姿勢データCSV: {args.csv}\n")

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

            # 常に画面中央からプレイヤーを検出（最大2名）
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