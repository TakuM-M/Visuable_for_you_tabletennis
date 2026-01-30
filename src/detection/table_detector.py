"""
卓球台検出モジュール
"""
import cv2
import numpy as np
from typing import Optional, Tuple, List
from collections import defaultdict

from ultralytics import YOLO

from src.core.data_classes import TableInfo


class TableDetector:
    """
    卓球台検出クラス

    - Ping Pong Table (卓球台)
    """
    TABLE_CLASS_NAME = "Ping Pong Table"

    def __init__(
        self,
        yolo_model_path: str = "models/table_detection/best.pt",
        cache_valid_frames: int = 1000,
    ):
        """
        卓球台検出器の初期化

        Args:
            yolo_model_path: YOLOモデルのパス
            cache_valid_frames: キャッシュ有効期間（フレーム数）
        """
        self.yolo_model = None
        self.class_names = {}
        self.cache_valid_frames = cache_valid_frames

        if yolo_model_path is None:
            print("警告: YOLOモデルパスが指定されていません")
        else:
            try:
                self.yolo_model = YOLO(yolo_model_path)
                self.class_names = self.yolo_model.names
                print(f"YOLOモデルをロードしました: {yolo_model_path}")
                print(f"検出可能なクラス: {list(self.class_names.values())}")
            except Exception as e:
                print(f"警告: YOLOモデルのロードに失敗しました: {e}")

        self._cached_table_info: Optional[TableInfo] = None
        self._cache_frame_idx: int = -1

    def detect_table_frame(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
        force_detect: bool = False
    ) -> Optional[TableInfo]:
        """
        単一フレームから卓球台のバウンディングボックスを検出

        Args:
            frame: 入力フレーム（BGR形式）
            frame_idx: フレーム番号
            force_detect: Trueの場合、キャッシュを無視して再検出

        Returns:
            TableInfo: 卓球台のバウンディングボックス情報、検出失敗時はNone
        """
        # キャッシュチェック
        # 過去の有効フレーム数内にあればキャッシュを返す
        if (not force_detect and
            self._cached_table_info is not None and
            abs(frame_idx - self._cache_frame_idx) < self.cache_valid_frames):
            return self._cached_table_info

        if self.yolo_model is None:
            print("警告: YOLOモデルが初期化されていません")
            return None
        
        results = self.yolo_model(frame, verbose=False)
        if len(results) == 0 or len(results[0].boxes) == 0:
            return None
        
        # 画面中央に最も近い卓球台を選択
        # 全検出結果から"Ping Pong Table"クラスのみをフィルタリング
        boxes = results[0].boxes
        frame_height, frame_width = frame.shape[:2]
        frame_center = (frame_width / 2, frame_height / 2)
        best_table = None
        min_distance = float('inf')

        for box in boxes:
            class_id = int(box.cls[0].cpu().numpy())
            class_name = self.class_names.get(class_id, "Unknown")

            if class_name != self.TABLE_CLASS_NAME:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # 距離:ユークリッド距離
            distance = np.sqrt((center_x - frame_center[0])**2 +
                             (center_y - frame_center[1])**2)

            # 最も中央に近いものを卓球台として選択
            if distance < min_distance:
                min_distance = distance
                best_table = TableInfo(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=confidence,
                    frame_idx=frame_idx
                )

        if best_table is None:
            return None

        # キャッシュに保存
        self._cached_table_info = best_table
        self._cache_frame_idx = frame_idx

        return best_table