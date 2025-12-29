"""
YOLOv11-Pose トラッキングモジュール
動画内の全人物をID付きでトラッキングし、姿勢キーポイントを取得する
"""
import cv2
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError(
        "ultralytics がインストールされていません。\n"
        "以下のコマンドでインストールしてください:\n"
        "pip install ultralytics"
    )


# COCO 17キーポイント定義
KEYPOINT_NAMES = [
    "nose",           # 0
    "left_eye",       # 1
    "right_eye",      # 2
    "left_ear",       # 3
    "right_ear",      # 4
    "left_shoulder",  # 5
    "right_shoulder", # 6
    "left_elbow",     # 7
    "right_elbow",    # 8
    "left_wrist",     # 9
    "right_wrist",    # 10
    "left_hip",       # 11
    "right_hip",      # 12
    "left_knee",      # 13
    "right_knee",     # 14
    "left_ankle",     # 15
    "right_ankle"     # 16
]


@dataclass
class PersonTrack:
    """トラッキングされた人物の情報"""
    track_id: int  # トラッキングID
    bbox: Tuple[int, int, int, int]  # バウンディングボックス (x1, y1, x2, y2)
    keypoints: np.ndarray  # キーポイント座標 (17, 3) [x, y, confidence]
    confidence: float  # 検出信頼度

    def get_center(self) -> Tuple[float, float]:
        """バウンディングボックスの中心座標を取得"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def get_keypoint(self, name: str) -> Optional[Tuple[float, float, float]]:
        """
        キーポイント名から座標を取得

        Args:
            name: キーポイント名（例: "nose", "left_shoulder"）

        Returns:
            (x, y, confidence) または None
        """
        if name not in KEYPOINT_NAMES:
            return None

        idx = KEYPOINT_NAMES.index(name)
        kp = self.keypoints[idx]
        return (float(kp[0]), float(kp[1]), float(kp[2]))

    def get_body_center_y(self) -> float:
        """
        体の中心Y座標を取得（肩と腰の中間）
        選手分類に使用

        Returns:
            体の中心Y座標
        """
        left_shoulder = self.get_keypoint("left_shoulder")
        right_shoulder = self.get_keypoint("right_shoulder")
        left_hip = self.get_keypoint("left_hip")
        right_hip = self.get_keypoint("right_hip")

        # 信頼度が高いキーポイントを優先的に使用
        y_coords = []

        if left_shoulder and left_shoulder[2] > 0.5:
            y_coords.append(left_shoulder[1])
        if right_shoulder and right_shoulder[2] > 0.5:
            y_coords.append(right_shoulder[1])
        if left_hip and left_hip[2] > 0.5:
            y_coords.append(left_hip[1])
        if right_hip and right_hip[2] > 0.5:
            y_coords.append(right_hip[1])

        if y_coords:
            return float(np.mean(y_coords))

        # フォールバック: バウンディングボックスの中心
        return float((self.bbox[1] + self.bbox[3]) / 2)


class YOLOPoseTracker:
    """YOLOv11-Pose トラッキングクラス"""

    def __init__(
        self,
        model_path: str = "yolo11n-pose.pt",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.7,
        device: str = "cpu"
    ):
        """
        YOLOトラッカーの初期化

        Args:
            model_path: YOLOモデルのパス（デフォルト: yolo11n-pose.pt）
            conf_threshold: 検出信頼度の閾値
            iou_threshold: NMS（Non-Maximum Suppression）のIoU閾値
            device: 使用デバイス（"cpu" or "cuda"）
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        # YOLOモデルをロード
        print(f"YOLOモデルをロード中: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(device)

        print(f"YOLOトラッカーを初期化しました")
        print(f"  モデル: {model_path}")
        print(f"  デバイス: {device}")
        print(f"  信頼度閾値: {conf_threshold}")

    def track_frame(
        self,
        frame: np.ndarray,
        persist: bool = True
    ) -> List[PersonTrack]:
        """
        単一フレームから人物をトラッキング

        Args:
            frame: 入力フレーム（BGR形式）
            persist: トラッキングIDを維持するか

        Returns:
            検出された人物のリスト
        """
        # YOLOで推論（トラッキング有効）
        results = self.model.track(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            persist=persist,
            verbose=False
        )

        persons = []

        if len(results) == 0:
            return persons

        result = results[0]

        # トラッキング結果が存在するか確認
        if result.boxes is None or len(result.boxes) == 0:
            return persons

        if result.keypoints is None:
            return persons

        # 各検出結果を処理
        for i in range(len(result.boxes)):
            box = result.boxes[i]
            kps = result.keypoints[i]

            # トラッキングIDを取得
            if box.id is not None:
                track_id = int(box.id.item())
            else:
                track_id = i  # IDがない場合はインデックスを使用

            # バウンディングボックス
            bbox_xyxy = box.xyxy[0].cpu().numpy()
            bbox = (
                int(bbox_xyxy[0]),
                int(bbox_xyxy[1]),
                int(bbox_xyxy[2]),
                int(bbox_xyxy[3])
            )

            # 信頼度
            confidence = float(box.conf.item())

            # キーポイント (17, 3) [x, y, confidence]
            keypoints = kps.data[0].cpu().numpy()

            persons.append(PersonTrack(
                track_id=track_id,
                bbox=bbox,
                keypoints=keypoints,
                confidence=confidence
            ))

        return persons

    def reset_tracker(self):
        """トラッキングIDをリセット"""
        # 新しいモデルインスタンスを作成してトラッカーをリセット
        self.model = YOLO(self.model_path)
        self.model.to(self.device)

    def draw_tracking(
        self,
        frame: np.ndarray,
        persons: List[PersonTrack],
        draw_bbox: bool = True,
        draw_keypoints: bool = True,
        draw_skeleton: bool = True,
        draw_id: bool = True
    ) -> np.ndarray:
        """
        フレームにトラッキング結果を描画

        Args:
            frame: 入力フレーム
            persons: トラッキング結果
            draw_bbox: バウンディングボックスを描画するか
            draw_keypoints: キーポイントを描画するか
            draw_skeleton: スケルトンを描画するか
            draw_id: トラッキングIDを描画するか

        Returns:
            描画後のフレーム
        """
        output = frame.copy()

        # スケルトンの接続定義（COCO形式）
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

        for person in persons:
            # カラーマップ（IDごとに色を変える）
            color_idx = person.track_id % 10
            colors = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255),
                (255, 255, 0), (255, 0, 255), (0, 255, 255),
                (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0)
            ]
            color = colors[color_idx]

            # バウンディングボックスを描画
            if draw_bbox:
                x1, y1, x2, y2 = person.bbox
                cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # トラッキングIDを描画
            if draw_id:
                x1, y1, _, _ = person.bbox
                label = f"ID:{person.track_id}"
                cv2.putText(output, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # スケルトンを描画
            if draw_skeleton:
                for connection in skeleton_connections:
                    kp1_idx, kp2_idx = connection
                    kp1 = person.keypoints[kp1_idx]
                    kp2 = person.keypoints[kp2_idx]

                    # 両方のキーポイントが信頼度 > 0.5 の場合のみ描画
                    if kp1[2] > 0.5 and kp2[2] > 0.5:
                        pt1 = (int(kp1[0]), int(kp1[1]))
                        pt2 = (int(kp2[0]), int(kp2[1]))
                        cv2.line(output, pt1, pt2, color, 2)

            # キーポイントを描画
            if draw_keypoints:
                for kp in person.keypoints:
                    if kp[2] > 0.5:  # 信頼度 > 0.5
                        pt = (int(kp[0]), int(kp[1]))
                        cv2.circle(output, pt, 3, color, -1)

        return output
