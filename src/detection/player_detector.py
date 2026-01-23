"""
YOLOv11-Pose を用いた人物トラッキングモジュール

"""

import cv2
import numpy as np
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.detection.data_classes import PersonTrack

from ultralytics import YOLO


class YOLOPose_PlayerDetector:
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