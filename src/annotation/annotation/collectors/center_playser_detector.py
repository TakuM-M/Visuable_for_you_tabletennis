import cv2
import numpy as np
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import csv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.detection.yolo_tracker import YOLOPoseTracker, PersonTrack, KEYPOINT_NAMES

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
    """画面中央のプレイヤー検出・トラッキングクラス（中央領域内の全員を検出）"""

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
        self.target_track_ids: list[int] = []  # トラッキングID
        self.center_region: Optional[CenterRegion] = None
        self.frame_width: Optional[int] = None
        self.frame_height: Optional[int] = None

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
        画面中央のプレイヤーを検出（毎フレーム実行、全員）

        Args:
            frame: 入力フレーム

        Returns:
            検出されたプレイヤーのリスト（中央領域内の全員）
        """
        if self.center_region is None:
            self.set_frame_size(frame.shape[1], frame.shape[0])

        # 全人物を検出（トラッキング有効）
        persons = self.tracker.track_frame(frame, persist=True)

        # 中央領域内にいる人物を探す
        center_persons = [p for p in persons if self.center_region.contains_person(p)]

        if not center_persons:
            return []

        # 信頼度が高い順にソート（制限なし）
        center_persons.sort(key=lambda p: p.confidence, reverse=True)
        target_persons = center_persons  # 全員を対象

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
        ターゲットプレイヤーをトラッキング（画面全体）

        Args:
            frame: 入力フレーム

        Returns:
            トラッキングされたプレイヤーのリスト（全員）
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
